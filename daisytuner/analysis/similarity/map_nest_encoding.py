# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import ast
import astunparse
import dace
import torch
import numpy as np
import networkx as nx
import torch_geometric as geo

from torch_geometric.utils.convert import to_networkx

from collections import Counter
from typing import Any, Dict, List
from pyvis.network import Network

from daisytuner.passes.map_expanded_form import MapExpandedForm

_CTYPES = {
    None: "void",
    int: "int",
    float: "float",
    complex: "dace::complex64",
    bool: "bool",
    np.bool_: "bool",
    np.int8: "char",
    np.int16: "short",
    np.int32: "int",
    np.int64: "long long",
    np.uint8: "unsigned char",
    np.uint16: "unsigned short",
    np.uint32: "unsigned int",
    np.uint64: "unsigned long long",
    np.float16: "dace::float16",
    np.float32: "float",
    np.float64: "double",
    np.complex64: "dace::complex64",
    np.complex128: "dace::complex128",
}

_FUNCS = [
    "+",
    "-",
    "*",
    "/",
    "sqrt",
    "exp",
    "pow",
    "tanh",
    "tan",
    "sin",
    "cos",
    "min",
    "max",
    "abs",
    "__INDIRECTION",
]


class MapNestEncoding:
    """
    Encodes map nests for usage with a graph neural network
    with the following limitations:

        1) Assumptions about symbol values must be provided
        2) It must use only default schedule, storage etc. annotations
        3) Non-affine expressions are encoded using one-hot representation.

    """

    MAX_ARRAY_DIMS = 6
    MAX_MAP_DIMS = 12

    START_ACCESS_NODE = 0
    NUM_FEATURES_ACCESS_NODE = (
        6 + len(_CTYPES) + (MAX_MAP_DIMS + 2) + 2 * MAX_ARRAY_DIMS * (MAX_MAP_DIMS + 2)
    )

    START_MAP_ENTRY = START_ACCESS_NODE + NUM_FEATURES_ACCESS_NODE
    NUM_FEATURES_MAP_ENTRY = 3 * (MAX_MAP_DIMS + 2) + 1

    START_MAP_EXIT = START_MAP_ENTRY + NUM_FEATURES_MAP_ENTRY
    NUM_FEATURES_MAP_EXIT = 1

    START_NESTED_SDFG = START_MAP_EXIT + NUM_FEATURES_MAP_EXIT
    NUM_FEATURES_NESTED_SDFG = 1

    START_TASKLET = START_NESTED_SDFG + NUM_FEATURES_NESTED_SDFG
    NUM_FEATURES_TASKLET = len(_FUNCS)

    START_MEMLET = 0
    NUM_FEATURES_MEMLET = 4 + len(_FUNCS) + MAX_ARRAY_DIMS * (3 * (MAX_MAP_DIMS + 2))

    def __init__(self, sdfg: dace.SDFG, symbol_values: Dict = None) -> None:
        assert MapNestEncoding.can_be_encoded(sdfg, symbol_values)

        self._sdfg = sdfg
        self._constants = {**sdfg.constants}
        if symbol_values is not None:
            self._constants.update(symbol_values)
        self._data = None

        # Helper structures
        self._element_table = {}
        self._nodes = []
        self._edges = [[], []]
        self._edges_attr = []
        self._data = None

        # Register buffer id
        self._buffers = []
        for arr in sdfg.arrays:
            self._buffers.append(arr)

        # Register param id
        self._params = []
        for node in sdfg.start_state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                self._params.append(node.map.params[0])

    @staticmethod
    def can_be_encoded(sdfg: dace.SDFG, symbol_values: Dict = None) -> bool:
        if not MapExpandedForm.is_expanded_form(sdfg):
            return False

        # 1. All symbols are defined
        params = set()
        for node in sdfg.start_state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                params.add(node.map.params[0])

        free_symbols = sdfg.free_symbols.difference(params)
        free_symbols = free_symbols.difference(sdfg.constants)
        if symbol_values is not None:
            free_symbols = free_symbols.difference(symbol_values)
        if len(free_symbols) > 0:
            return False

        # 2. SDFG only contains the map nest
        #   - single state
        #   - single connected-component
        #   - single top-level map
        if len(sdfg.states()) > 1:
            return False

        state = sdfg.start_state
        if len(list(nx.weakly_connected_components(state._nx))) > 1:
            return False

        top_level_maps = [
            node
            for node in sdfg.start_state
            if isinstance(node, dace.nodes.MapEntry) and state.entry_node(node) is None
        ]
        if len(top_level_maps) != 1:
            return False

        # 3. No scheduling information
        #   - storage
        #   - schedules
        for desc in sdfg.arrays.values():
            if desc.storage != dace.StorageType.Default:
                return False

        for node in sdfg.start_state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                if node.map.schedule != dace.ScheduleType.Default:
                    return False

        return True

    def visualize(self) -> None:
        nx_graph = to_networkx(self._data)
        nt = Network("500px", "500px", directed=True)
        nt.from_nx(nx_graph)
        nt.show(f"{self._sdfg.label}.html", notebook=False)

    def encode(self) -> None:
        if self._data is not None:
            return self._data

        state = self._sdfg.start_state
        for node in state.nodes():
            if isinstance(node, dace.nodes.AccessNode):
                self._encode_access_node(node)
            elif isinstance(node, dace.nodes.MapEntry):
                self._encode_map_entry(node)
            elif isinstance(node, dace.nodes.MapExit):
                self._encode_map_exit(node)
            elif isinstance(node, dace.nodes.NestedSDFG):
                self._encode_nested_sdfg(node)
            elif isinstance(node, dace.nodes.Tasklet):
                self._encode_tasklet(node)
            else:
                raise ValueError("Node not supported: ", node)

        for edge in state.edges():
            self._encode_data_edge(edge)

        nodes = np.vstack(self._nodes)
        edge_index = np.vstack(
            [
                np.array(self._edges[0], dtype=np.int64),
                np.array(self._edges[1], dtype=np.int64),
            ]
        )
        edge_attr = np.vstack(self._edges_attr)

        # Log2p transformation
        nodes = np.sign(nodes) * np.log2(np.abs(nodes) + 1.0)

        x = torch.tensor(nodes, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        self._data = geo.data.Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        return self._data

    def _encode_access_node(self, node: dace.nodes.AccessNode) -> None:
        features = np.zeros(
            (MapNestEncoding.NUM_FEATURES_ACCESS_NODE), dtype=np.float32
        )

        # Obtain buffer id
        arr = node.data
        buffer_id = self._buffers.index(arr) + 1
        desc = self._sdfg.arrays[arr]

        # Array features:
        # 0: buffer_id - assign IDs to arrays
        # 1: transient
        # 2: dims
        # 3: start_offset
        # 4: alignment
        # 5: dtype (bytes)
        # 6: dtype (one-hot)
        # 7: total_size
        features[0] = buffer_id
        features[1] = 1 if desc.transient else 0
        features[2] = len(desc.shape)
        features[3] = desc.start_offset
        if isinstance(desc, dace.data.Array):
            features[4] = desc.alignment
        features[5] = desc.dtype.bytes
        ctype_one_hot = MapNestEncoding.encode_ctype(desc.ctype)
        features[6 : len(_CTYPES) + 6] = ctype_one_hot[:]

        offset = 6 + len(_CTYPES)
        features[
            offset : offset + (MapNestEncoding.MAX_MAP_DIMS + 2)
        ] = MapNestEncoding.encode_expression(
            desc.total_size, self._constants, self._params
        )

        # 8: Shape
        offset = offset + (MapNestEncoding.MAX_MAP_DIMS + 2)
        for i, dim in enumerate(desc.shape):
            dim = MapNestEncoding.encode_expression(dim, self._constants, self._params)
            features[
                offset
                + i * (MapNestEncoding.MAX_MAP_DIMS + 2) : offset
                + (i + 1) * (MapNestEncoding.MAX_MAP_DIMS + 2)
            ] = dim

        # 9: Stride
        offset = offset + MapNestEncoding.MAX_ARRAY_DIMS * (
            MapNestEncoding.MAX_MAP_DIMS + 2
        )
        for i, stride in enumerate(desc.strides):
            stride = MapNestEncoding.encode_expression(
                stride, self._constants, self._params
            )
            features[
                offset
                + i * (MapNestEncoding.MAX_MAP_DIMS + 2) : offset
                + (i + 1) * (MapNestEncoding.MAX_MAP_DIMS + 2)
            ] = stride

        b, e = (
            MapNestEncoding.START_ACCESS_NODE,
            MapNestEncoding.START_ACCESS_NODE
            + MapNestEncoding.NUM_FEATURES_ACCESS_NODE,
        )
        full_features = np.zeros(MapNestEncoding.node_dimensions(), dtype=np.float32)
        full_features[b:e] = features[:]
        self._nodes.append(full_features)
        self._element_table[node] = len(self._nodes) - 1

    def _encode_map_entry(self, node: dace.nodes.MapEntry) -> None:
        features = np.zeros((MapNestEncoding.NUM_FEATURES_MAP_ENTRY), dtype=np.float32)

        # Obtain param id
        param = node.map.params[0]
        param_id = self._params.index(param) + 1

        begin, end, step = node.map.range[0]

        # Map features:
        # 0: param id
        # 1: begin (affine: vector, non-affine: -1)
        # 2: end (affine: vector, non-affine: -1)
        # 3: step (affine: vector, non-affine: -1)
        features[0] = param_id
        features[
            1 : (MapNestEncoding.MAX_MAP_DIMS + 2) + 1
        ] = MapNestEncoding.encode_expression(begin, self._constants, self._params)
        features[
            (MapNestEncoding.MAX_MAP_DIMS + 2)
            + 1 : 2 * (MapNestEncoding.MAX_MAP_DIMS + 2)
            + 1
        ] = MapNestEncoding.encode_expression(end, self._constants, self._params)
        features[
            2 * (MapNestEncoding.MAX_MAP_DIMS + 2)
            + 1 : 3 * (MapNestEncoding.MAX_MAP_DIMS + 2)
            + 1
        ] = MapNestEncoding.encode_expression(step, self._constants, self._params)

        b, e = (
            MapNestEncoding.START_MAP_ENTRY,
            MapNestEncoding.START_MAP_ENTRY + MapNestEncoding.NUM_FEATURES_MAP_ENTRY,
        )
        full_features = np.zeros(MapNestEncoding.node_dimensions(), dtype=np.float32)
        full_features[b:e] = features[:]
        self._nodes.append(full_features)
        self._element_table[node] = len(self._nodes) - 1

    def _encode_map_exit(self, node: dace.nodes.MapExit) -> None:
        features = np.zeros((MapNestEncoding.NUM_FEATURES_MAP_EXIT), dtype=np.float32)

        # Obtain param id
        param = node.map.params[0]
        param_id = self._params.index(param) + 1

        # Map features:
        # 0: param id
        features[0] = param_id

        b, e = (
            MapNestEncoding.START_MAP_EXIT,
            MapNestEncoding.START_MAP_EXIT + MapNestEncoding.NUM_FEATURES_MAP_EXIT,
        )
        full_features = np.zeros(MapNestEncoding.node_dimensions(), dtype=np.float32)
        full_features[b:e] = features[:]
        self._nodes.append(full_features)
        self._element_table[node] = len(self._nodes) - 1

    def _encode_nested_sdfg(self, node: dace.nodes.NestedSDFG) -> None:
        features = np.zeros(
            (MapNestEncoding.NUM_FEATURES_NESTED_SDFG), dtype=np.float32
        )

        # NestedSDFG features:
        # 0: one-hot
        features[0] = 1

        b, e = (
            MapNestEncoding.START_NESTED_SDFG,
            MapNestEncoding.START_NESTED_SDFG
            + MapNestEncoding.NUM_FEATURES_NESTED_SDFG,
        )
        full_features = np.zeros(MapNestEncoding.node_dimensions(), dtype=np.float32)
        full_features[b:e] = features[:]
        self._nodes.append(full_features)
        self._element_table[node] = len(self._nodes) - 1

    def _encode_tasklet(self, node: dace.nodes.Tasklet) -> None:
        # Tasklet histogram of ops
        features = MapNestEncoding.encode_code(
            node.code.as_string, language=node.language
        )

        b, e = (
            MapNestEncoding.START_TASKLET,
            MapNestEncoding.START_TASKLET + MapNestEncoding.NUM_FEATURES_TASKLET,
        )
        full_features = np.zeros(MapNestEncoding.node_dimensions(), dtype=np.float32)
        full_features[b:e] = features[:]
        self._nodes.append(full_features)
        self._element_table[node] = len(self._nodes) - 1

    def _encode_data_edge(self, edge) -> None:
        features = np.zeros((MapNestEncoding.NUM_FEATURES_MEMLET), dtype=np.float32)

        memlet = edge.data

        # Features:
        # 0: buffer_id
        # 1: dims
        # 2: has wcr
        features[0] = 0 if memlet.data is None else self._buffers.index(memlet.data) + 1
        features[1] = 0 if memlet.data is None else len(memlet.subset)
        features[2] = 0 if memlet.dynamic else 1
        features[3] = 0 if memlet.wcr is None else 1

        code = "" if memlet.wcr is None else memlet.wcr
        wcr_features = MapNestEncoding.encode_code(code, language=dace.Language.Python)
        features[4 : 4 + wcr_features.size] = wcr_features[:]

        if memlet.data is not None:
            offset = 4 + wcr_features.size
            stride = 3 * (MapNestEncoding.MAX_MAP_DIMS + 2)
            for i, (b, e, s) in enumerate(memlet.subset):
                b = MapNestEncoding.encode_expression(b, self._constants, self._params)
                features[
                    offset
                    + i * stride : offset
                    + i * stride
                    + (MapNestEncoding.MAX_MAP_DIMS + 2)
                ] = b

                e = MapNestEncoding.encode_expression(e, self._constants, self._params)
                features[
                    offset
                    + i * stride
                    + (MapNestEncoding.MAX_MAP_DIMS + 2) : offset
                    + i * stride
                    + 2 * (MapNestEncoding.MAX_MAP_DIMS + 2)
                ] = e

                s = MapNestEncoding.encode_expression(s, self._constants, self._params)
                features[
                    offset
                    + i * stride
                    + 2 * (MapNestEncoding.MAX_MAP_DIMS + 2) : offset
                    + i * stride
                    + 3 * (MapNestEncoding.MAX_MAP_DIMS + 2)
                ] = s

        b, e = (
            MapNestEncoding.START_MEMLET,
            MapNestEncoding.START_MEMLET + MapNestEncoding.NUM_FEATURES_MEMLET,
        )
        full_features = np.zeros(MapNestEncoding.edge_dimensions(), dtype=np.float32)
        full_features[b:e] = features[:]
        self._edges_attr.append(full_features)
        self._element_table[edge] = len(self._edges_attr) - 1

        self._edges[0].append(self._element_table[edge.src])
        self._edges[1].append(self._element_table[edge.dst])

    @staticmethod
    def encode_ctype(ctype: str):
        one_hot_encoding = np.zeros((len(_CTYPES),), dtype=np.float32)
        one_hot_encoding[list(_CTYPES.values()).index(ctype)] = 1
        return one_hot_encoding

    @staticmethod
    def node_dimensions():
        return (
            MapNestEncoding.NUM_FEATURES_ACCESS_NODE
            + MapNestEncoding.NUM_FEATURES_MAP_ENTRY
            + MapNestEncoding.NUM_FEATURES_MAP_EXIT
            + MapNestEncoding.NUM_FEATURES_NESTED_SDFG
            + MapNestEncoding.NUM_FEATURES_TASKLET
        )

    @staticmethod
    def edge_dimensions():
        return MapNestEncoding.NUM_FEATURES_MEMLET

    @staticmethod
    def encode_expression(expr: str, symbols: Dict, parameters: List[str]):
        sympy_expr = dace.symbolic.pystr_to_symbolic(expr)
        sympy_expr = sympy_expr.subs(symbols)

        features = np.zeros((MapNestEncoding.MAX_MAP_DIMS + 2), dtype=np.float32)
        affine_decomposition = MapNestEncoding.decompose_affine_expression(
            sympy_expr, parameters
        )
        if affine_decomposition is None:
            features[-1] = 1
        else:
            for param in affine_decomposition:
                if param is None:
                    features[-2] = affine_decomposition[param]
                else:
                    features[parameters.index(param)] = affine_decomposition[param]

        return features

    @staticmethod
    def decompose_affine_expression(expr, parameters: List[str]):
        sympy_expr = dace.symbolic.pystr_to_symbolic(expr)

        if sympy_expr.is_constant():
            return {None: dace.symbolic.evaluate(sympy_expr, symbols={})}
        elif str(sympy_expr) in parameters:
            return {str(sympy_expr): 1}
        elif sympy_expr.is_Mul:
            args = sympy_expr.args
            if len(args) != 2:
                return None

            arg1 = MapNestEncoding.decompose_affine_expression(
                args[0], parameters=parameters
            )
            if arg1 is None or len(arg1) != 1:
                return None
            arg2 = MapNestEncoding.decompose_affine_expression(
                args[1], parameters=parameters
            )
            if arg2 is None or len(arg2) != 1:
                return None

            decomp = {}
            if None in arg1:
                sym = next(arg2.__iter__())
                return {sym: arg1[None]}
            elif None in arg2:
                sym = next(arg1.__iter__())
                return {sym: arg2[None]}
            else:
                return None
        elif sympy_expr.is_Add:
            decomp = {}
            for arg in sympy_expr.args:
                arg_decomp = MapNestEncoding.decompose_affine_expression(
                    arg, parameters=parameters
                )
                if arg_decomp is None:
                    return None

                for param in arg_decomp:
                    if param not in decomp:
                        decomp[param] = 0
                    decomp[param] += arg_decomp[param]
            return decomp
        else:
            return None

    @staticmethod
    def encode_code(code, language: dace.Language):
        if language == dace.Language.Python:
            ctr = ArithmeticCounter()
            if isinstance(code, (tuple, list)):
                for stmt in code:
                    ctr.visit(stmt)
            elif isinstance(code, str):
                ctr.visit(ast.parse(code))
            else:
                ctr.visit(code)

            counter = ctr.count
        else:
            counter = Counter()
            for func in _FUNCS:
                for i in range(code.count(func)):
                    counter.update((func,))

        features = np.zeros((len(_FUNCS)), dtype=np.float32)
        for i, func in enumerate(_FUNCS):
            if func in counter:
                features[i] = counter[func]

        return features


class ArithmeticCounter(ast.NodeVisitor):
    def __init__(self):
        self.count = Counter()

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Add):
            self.count.update(("+",))
        elif isinstance(node.op, ast.Mult):
            self.count.update(("*",))
        elif isinstance(node.op, ast.Div):
            self.count.update(("/",))
        elif isinstance(node.op, ast.Sub):
            self.count.update(("-",))

        return self.generic_visit(node)

    def visit_UnaryOp(self, node):
        return self.generic_visit(node)

    def visit_Call(self, node):
        fname = astunparse.unparse(node.func)[:-1]
        if fname.startswith("math."):
            fname = fname[5:]

        if fname in _FUNCS:
            self.count.update((fname,))
        return self.generic_visit(node)

    def visit_Subscript(self, node) -> Any:
        if isinstance(node.slice, ast.Name):
            self.count.update(("__INDIRECTION",))
        return self.generic_visit(node)

    def visit_AugAssign(self, node):
        return self.visit_BinOp(node)

    def visit_For(self, node):
        raise NotImplementedError

    def visit_While(self, node):
        raise NotImplementedError
