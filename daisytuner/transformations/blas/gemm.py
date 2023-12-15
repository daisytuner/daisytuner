# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import re
import ast

from dace.sdfg import utils as sdutil
from dace.sdfg.state import SDFGState
from dace.properties import make_properties
from dace.transformation.transformation import SingleStateTransformation, PatternNode

from daisytuner.library.blas import Gemm


@make_properties
class GEMM(SingleStateTransformation):
    map_entry = PatternNode(dace.nodes.MapEntry)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry)]

    def can_be_applied(
        self, state: dace.SDFGState, expr_index: int, sdfg: dace.SDFG, permissive=False
    ):
        if len(self.map_entry.map.params) != 3:
            return False

        map_exit = state.exit_node(self.map_entry)
        tasklets = list(state.all_nodes_between(self.map_entry, map_exit))
        if len(tasklets) != 1:
            return False

        tasklet = tasklets[0]

        in_conns = list(tasklet.in_connectors.keys())
        out_conns = list(tasklet.out_connectors.keys())
        if len(in_conns) != 2 or len(out_conns) != 1:
            return False

        a_edge = list(state.in_edges_by_connector(tasklet, in_conns[0]))
        b_edge = list(state.in_edges_by_connector(tasklet, in_conns[1]))
        c_edge = list(state.out_edges_by_connector(tasklet, out_conns[0]))
        if len(a_edge) != 1 or len(b_edge) != 1 or len(c_edge) != 1:
            return False

        a_edge = a_edge[0]
        b_edge = b_edge[0]
        c_edge = c_edge[0]
        if (
            a_edge.data.data not in sdfg.arrays
            or b_edge.data.data not in sdfg.arrays
            or c_edge.data.data not in sdfg.arrays
        ):
            return False

        A, A_desc = (a_edge.data.data, sdfg.arrays[a_edge.data.data])
        B, B_desc = (b_edge.data.data, sdfg.arrays[b_edge.data.data])
        C, C_desc = (c_edge.data.data, sdfg.arrays[c_edge.data.data])
        if len(A_desc.shape) != 2 or len(B_desc.shape) != 2 or len(C_desc.shape) != 2:
            return False

        if A_desc.dtype != dace.float64 and A_desc.dtype != dace.float32:
            return False
        if B_desc.dtype != dace.float64 and B_desc.dtype != dace.float32:
            return False
        if C_desc.dtype != dace.float64 and C_desc.dtype != dace.float32:
            return False

        c_strides = [
            c_edge.data.get_stride(sdfg, self.map_entry.map, dim=i) for i in range(3)
        ]
        if (
            c_edge.data.wcr is None
            or "+" not in c_edge.data.wcr
            or c_strides != [C_desc.shape[1], 1, 0]
        ):
            return False

        # Determine A and B
        a_strides = [
            a_edge.data.get_stride(sdfg, self.map_entry.map, dim=i) for i in range(3)
        ]
        b_strides = [
            b_edge.data.get_stride(sdfg, self.map_entry.map, dim=i) for i in range(3)
        ]
        if a_strides == [A_desc.shape[1], 0, 1]:
            if b_strides != [0, 1, B_desc.shape[1]]:
                return False
        elif a_strides == [0, 1, A_desc.shape[1]]:
            if b_strides != [B_desc.shape[1], 0, 1]:
                return False

            tmp = (A, A_desc, a_edge)
            A, A_desc, a_edge = (B, B_desc, b_edge)
            B, B_desc, b_edge = tmp
        else:
            return False

        op = tasklet.code.as_string
        if tasklet.language == dace.Language.CPP:
            op = op.replace(";", "")

        # TODO: Does not scalar / or canonicalize tasklet codes more
        # A * B, B * A
        p1 = r"^\s*%s\s*=\s*%s\s*\*\s*%s\s*" % (
            re.escape(c_edge.src_conn),
            re.escape(a_edge.dst_conn),
            re.escape(b_edge.dst_conn),
        )
        p2 = r"^\s*%s\s*=\s*%s\s*\*\s*%s\s*" % (
            re.escape(c_edge.src_conn),
            re.escape(b_edge.dst_conn),
            re.escape(a_edge.dst_conn),
        )
        p1_ = r"^\s*%s\s*=\s*\(\s*%s\s*\*\s*%s\s*\)\s*" % (
            re.escape(c_edge.src_conn),
            re.escape(a_edge.dst_conn),
            re.escape(b_edge.dst_conn),
        )
        p2_ = r"^\s*%s\s*=\s*\(\s*%s\s*\*\s*%s\s*\)\s*" % (
            re.escape(c_edge.src_conn),
            re.escape(b_edge.dst_conn),
            re.escape(a_edge.dst_conn),
        )
        if (
            re.match(p1, op) is not None
            or re.match(p2, op) is not None
            or re.match(p2_, op) is not None
            or re.match(p1_, op) is not None
        ):
            return True

        # alpha * A * B
        p3 = r"^\s*%s\s*=\s*\(\s*%s\s*\*\s*(.*)\s*\)\s*\*\s*%s\s*" % (
            re.escape(c_edge.src_conn),
            re.escape(a_edge.dst_conn),
            re.escape(b_edge.dst_conn),
        )
        match3 = re.match(p3, op)
        if match3 is not None:
            alpha = match3.group(1)
            try:
                alpha = float(ast.literal_eval(alpha))
                return True
            except ValueError:
                return False

        p4 = r"^\s*%s\s*=\s*\(\s*%s\s*\*\s*(.*)\s*\)\s*\*\s*%s\s*" % (
            re.escape(c_edge.src_conn),
            re.escape(f"({a_edge.dst_conn})"),
            re.escape(b_edge.dst_conn),
        )
        match4 = re.match(p4, op)
        if match4 is not None:
            alpha = match4.group(1)
            try:
                alpha = float(ast.literal_eval(alpha))
                return True
            except ValueError:
                return False

        return False

    def apply(self, state: SDFGState, sdfg: dace.SDFG):
        map_exit = state.exit_node(self.map_entry)
        tasklets = list(state.all_nodes_between(self.map_entry, map_exit))
        tasklet = tasklets[0]

        in_conns = list(tasklet.in_connectors.keys())
        out_conns = list(tasklet.out_connectors.keys())

        a_edge = list(state.in_edges_by_connector(tasklet, in_conns[0]))[0]
        A, A_desc = (a_edge.data.data, sdfg.arrays[a_edge.data.data])

        b_edge = list(state.in_edges_by_connector(tasklet, in_conns[1]))[0]
        B, B_desc = (b_edge.data.data, sdfg.arrays[b_edge.data.data])

        c_edge = list(state.out_edges_by_connector(tasklet, out_conns[0]))[0]
        C, C_desc = (c_edge.data.data, sdfg.arrays[c_edge.data.data])

        # Determine A and B
        a_strides = [
            a_edge.data.get_stride(sdfg, self.map_entry.map, dim=i) for i in range(3)
        ]
        b_strides = [
            b_edge.data.get_stride(sdfg, self.map_entry.map, dim=i) for i in range(3)
        ]
        if a_strides == [0, 1, A_desc.shape[1]]:
            if b_strides != [B_desc.shape[1], 0, 1]:
                return False

            tmp = (A, A_desc, a_edge)
            A, A_desc, a_edge = (B, B_desc, b_edge)
            B, B_desc, b_edge = tmp

        A_node = list(
            state.in_edges_by_connector(self.map_entry, "IN_" + a_edge.src_conn[4:])
        )[0].src
        B_node = list(
            state.in_edges_by_connector(self.map_entry, "IN_" + b_edge.src_conn[4:])
        )[0].src
        C_node = list(
            state.out_edges_by_connector(map_exit, "OUT_" + c_edge.dst_conn[3:])
        )[0].dst

        op = tasklet.code.as_string
        if tasklet.language == dace.Language.CPP:
            op = op.replace(";", "")

        p1 = r"^\s*%s\s*=\s*%s\s*\*\s*%s\s*" % (
            re.escape(c_edge.src_conn),
            re.escape(a_edge.dst_conn),
            re.escape(b_edge.dst_conn),
        )
        p2 = r"^\s*%s\s*=\s*%s\s*\*\s*%s\s*" % (
            re.escape(c_edge.src_conn),
            re.escape(b_edge.dst_conn),
            re.escape(a_edge.dst_conn),
        )
        p1_ = r"^\s*%s\s*=\s*\(\s*%s\s*\*\s*%s\s*\)\s*" % (
            re.escape(c_edge.src_conn),
            re.escape(a_edge.dst_conn),
            re.escape(b_edge.dst_conn),
        )
        p2_ = r"^\s*%s\s*=\s*\(\s*%s\s*\*\s*%s\s*\)\s*" % (
            re.escape(c_edge.src_conn),
            re.escape(b_edge.dst_conn),
            re.escape(a_edge.dst_conn),
        )
        if (
            re.match(p1, op) is not None
            or re.match(p2, op) is not None
            or re.match(p2_, op) is not None
            or re.match(p1_, op) is not None
        ):
            alpha = 1.0
        else:
            p3 = r"^\s*%s\s*=\s*\(\s*%s\s*\*\s*(.*)\s*\)\s*\*\s*%s\s*" % (
                re.escape(c_edge.src_conn),
                re.escape(a_edge.dst_conn),
                re.escape(b_edge.dst_conn),
            )
            match3 = re.match(p3, op)
            if match3 is not None:
                alpha = match3.group(1)
                alpha = float(ast.literal_eval(alpha))

            p4 = r"^\s*%s\s*=\s*\(\s*%s\s*\*\s*(.*)\s*\)\s*\*\s*%s\s*" % (
                re.escape(c_edge.src_conn),
                re.escape(f"({a_edge.dst_conn})"),
                re.escape(b_edge.dst_conn),
            )
            match4 = re.match(p4, op)
            if match4 is not None:
                alpha = match4.group(1)
                alpha = float(ast.literal_eval(alpha))

        state.remove_node(map_exit)
        state.remove_node(self.map_entry)
        state.remove_node(tasklet)

        libnode = Gemm("_Gemm_", transA=False, transB=False, alpha=alpha, beta=0.0)
        state.add_node(libnode)

        state.add_edge(A_node, None, libnode, "_a", dace.Memlet.from_array(A, A_desc))
        state.add_edge(B_node, None, libnode, "_b", dace.Memlet.from_array(B, B_desc))
        state.add_edge(libnode, "_c", C_node, None, dace.Memlet.from_array(C, C_desc))
