# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace

from dace.sdfg import utils as sdutil
from dace.sdfg.state import SDFGState
from dace.properties import make_properties
from dace.transformation.transformation import SingleStateTransformation, PatternNode

from daisytuner.library.blas import Syr2k


@make_properties
class SYR2K(SingleStateTransformation):
    outer_map_entry = PatternNode(dace.nodes.MapEntry)
    inner_map_entry = PatternNode(dace.nodes.MapEntry)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.outer_map_entry, cls.inner_map_entry)]

    def can_be_applied(
        self, state: dace.SDFGState, expr_index: int, sdfg: dace.SDFG, permissive=False
    ):
        if (
            len(self.outer_map_entry.map.params) != 2
            or len(self.inner_map_entry.map.params) != 1
        ):
            return False

        inner_map_exit = state.exit_node(self.inner_map_entry)
        tasklets = list(state.all_nodes_between(self.inner_map_entry, inner_map_exit))
        if len(tasklets) != 1:
            return False

        tasklet = tasklets[0]

        inedges = state.in_edges(tasklet)
        array_in_conns = list()
        scalar_in_conns = list()
        for iedge in inedges:
            if sdfg.arrays[iedge.data.data].total_size == 1:
                scalar_in_conns.append(iedge.dst_conn)
                continue

            if iedge.dst_conn not in array_in_conns:
                array_in_conns.append(iedge.dst_conn)

        out_conns = list(tasklet.out_connectors.keys())
        if len(array_in_conns) != 4 or len(out_conns) != 1 or len(scalar_in_conns) > 1:
            return False

        a1_edge = list(state.in_edges_by_connector(tasklet, array_in_conns[0]))
        a2_edge = list(state.in_edges_by_connector(tasklet, array_in_conns[1]))
        b1_edge = list(state.in_edges_by_connector(tasklet, array_in_conns[2]))
        b2_edge = list(state.in_edges_by_connector(tasklet, array_in_conns[3]))
        c_edge = list(state.out_edges_by_connector(tasklet, out_conns[0]))
        if (
            len(a1_edge) != 1
            or len(a2_edge) != 1
            or len(b1_edge) != 1
            or len(b2_edge) != 1
            or len(c_edge) != 1
        ):
            return False

        a1_edge = a1_edge[0]
        a2_edge = a2_edge[0]
        b1_edge = b1_edge[0]
        b2_edge = b2_edge[0]
        c_edge = c_edge[0]

        if a1_edge.data.data != a2_edge.data.data:
            if a1_edge.data.data == b1_edge.data.data:
                tmp = a2_edge
                a2_edge = b1_edge
                b1_edge = tmp
            else:
                tmp = a2_edge
                a2_edge = b2_edge
                b2_edge = tmp

        if (
            a1_edge.data.data not in sdfg.arrays
            or b1_edge.data.data not in sdfg.arrays
            or c_edge.data.data not in sdfg.arrays
            or a1_edge.data.data != a2_edge.data.data
            or b1_edge.data.data != b2_edge.data.data
        ):
            return False

        A, A_desc = (a1_edge.data.data, sdfg.arrays[a1_edge.data.data])
        B, B_desc = (b1_edge.data.data, sdfg.arrays[b1_edge.data.data])
        C, C_desc = (c_edge.data.data, sdfg.arrays[c_edge.data.data])
        if (
            len(A_desc.shape) != 2
            or len(A_desc.shape) != 2
            or dace.symbolic.evaluate(A_desc.shape, sdfg.constants)
            != dace.symbolic.evaluate(B_desc.shape, sdfg.constants)
            or len(C_desc.shape) != 2
        ):
            return False

        if A_desc.dtype != dace.float64 and A_desc.dtype != dace.float32:
            return False
        if C_desc.dtype != dace.float64 and C_desc.dtype != dace.float32:
            return False

        c_strides = [
            c_edge.data.get_stride(sdfg, self.outer_map_entry.map, dim=i)
            for i in range(2)
        ] + [
            c_edge.data.get_stride(sdfg, self.inner_map_entry.map, dim=i)
            for i in range(1)
        ]
        if (
            c_edge.data.wcr is None
            or "+" not in c_edge.data.wcr
            or c_strides != [C_desc.shape[1], 0, 1]
        ):
            return False

        # Determine A and B
        a1_strides = [
            a1_edge.data.get_stride(sdfg, self.outer_map_entry.map, dim=i)
            for i in range(2)
        ] + [
            a1_edge.data.get_stride(sdfg, self.inner_map_entry.map, dim=i)
            for i in range(1)
        ]
        a2_strides = [
            a2_edge.data.get_stride(sdfg, self.outer_map_entry.map, dim=i)
            for i in range(2)
        ] + [
            a2_edge.data.get_stride(sdfg, self.inner_map_entry.map, dim=i)
            for i in range(1)
        ]
        if a1_strides == [A_desc.shape[1], 1, 0]:
            if a2_strides != [0, 1, A_desc.shape[1]]:
                return False
        elif a1_strides == [0, 1, A_desc.shape[1]]:
            if a2_strides != [A_desc.shape[1], 1, 0]:
                return False

            tmp = a1_edge
            a1_edge = a2_edge
            a2_edge = tmp
        else:
            return False

        b1_strides = [
            b1_edge.data.get_stride(sdfg, self.outer_map_entry.map, dim=i)
            for i in range(2)
        ] + [
            b1_edge.data.get_stride(sdfg, self.inner_map_entry.map, dim=i)
            for i in range(1)
        ]
        b2_strides = [
            b2_edge.data.get_stride(sdfg, self.outer_map_entry.map, dim=i)
            for i in range(2)
        ] + [
            b2_edge.data.get_stride(sdfg, self.inner_map_entry.map, dim=i)
            for i in range(1)
        ]
        if b1_strides == [B_desc.shape[1], 1, 0]:
            if b2_strides != [0, 1, B_desc.shape[1]]:
                return False
        elif b1_strides == [0, 1, B_desc.shape[1]]:
            if b2_strides != [B_desc.shape[1], 1, 0]:
                return False

            tmp = b1_edge
            b1_edge = b2_edge
            b2_edge = tmp
        else:
            return False

        (b1, e1, s1), (b2, e2, s2) = self.outer_map_entry.map.range
        b, e, s = self.inner_map_entry.map.range[0]
        if (
            b1 != 0
            or dace.symbolic.evaluate(e1, sdfg.constants)
            != dace.symbolic.evaluate(A_desc.shape[0], sdfg.constants) - 1
            or s1 != 1
        ):
            return False
        if (
            b2 != 0
            or dace.symbolic.evaluate(e2, sdfg.constants)
            != dace.symbolic.evaluate(A_desc.shape[1], sdfg.constants) - 1
            or s2 != 1
        ):
            return False
        if b != 0 or s != 1 or str(e) != self.outer_map_entry.map.params[0]:
            return False

        op = tasklet.code.as_string
        if tasklet.language == dace.Language.CPP:
            op = op.replace(";", "")

        _, op = op.split("=")
        op = op.strip()
        if (
            op
            != f"(({a2_edge.dst_conn} * {b1_edge.dst_conn}) + ({b2_edge.dst_conn} * {a1_edge.dst_conn}))"
            and op
            != f"(({a2_edge.dst_conn} * (1.500000e+00)) * {b1_edge.dst_conn}) + (({b2_edge.dst_conn} * (1.500000e+00)) * {a1_edge.dst_conn})"
            and op
            != f"(({b2_edge.dst_conn} * (1.500000e+00)) * {a1_edge.dst_conn}) + (({a2_edge.dst_conn} * (1.500000e+00)) * {b1_edge.dst_conn})"
            and op
            != f"(({b2_edge.dst_conn} * {a1_edge.dst_conn}) + ({a2_edge.dst_conn} * {b1_edge.dst_conn}))"
            and (
                scalar_in_conns
                and op
                != f"((({b2_edge.dst_conn} * {scalar_in_conns[0]}) * {a1_edge.dst_conn}) + (({a2_edge.dst_conn} * {scalar_in_conns[0]}) * {b1_edge.dst_conn}))"
            )
        ):
            return False

        return True

    def apply(self, state: SDFGState, sdfg: dace.SDFG):
        outer_map_entry = self.outer_map_entry
        inner_map_entry = self.inner_map_entry
        outer_map_exit = state.exit_node(self.outer_map_entry)
        inner_map_exit = state.exit_node(self.inner_map_entry)
        tasklets = list(state.all_nodes_between(self.inner_map_entry, inner_map_exit))
        tasklet = tasklets[0]

        inedges = state.in_edges(tasklet)
        array_in_conns = list()
        scalar_in_conns = list()
        for iedge in inedges:
            if sdfg.arrays[iedge.data.data].total_size == 1:
                scalar_in_conns.append(iedge.dst_conn)
                continue

            if iedge.dst_conn not in array_in_conns:
                array_in_conns.append(iedge.dst_conn)

        out_conns = list(tasklet.out_connectors.keys())

        a1_edge = list(state.in_edges_by_connector(tasklet, array_in_conns[0]))[0]
        a2_edge = list(state.in_edges_by_connector(tasklet, array_in_conns[1]))[0]
        b1_edge = list(state.in_edges_by_connector(tasklet, array_in_conns[2]))[0]
        b2_edge = list(state.in_edges_by_connector(tasklet, array_in_conns[3]))[0]

        if a1_edge.data.data != a2_edge.data.data:
            if a1_edge.data.data == b1_edge.data.data:
                tmp = a2_edge
                a2_edge = b1_edge
                b1_edge = tmp
            else:
                tmp = a2_edge
                a2_edge = b2_edge
                b2_edge = tmp

        # Determine A and B

        A, A_desc = (a1_edge.data.data, sdfg.arrays[a1_edge.data.data])
        a1_strides = [
            a1_edge.data.get_stride(sdfg, self.outer_map_entry.map, dim=i)
            for i in range(2)
        ] + [
            a1_edge.data.get_stride(sdfg, self.inner_map_entry.map, dim=i)
            for i in range(1)
        ]
        a2_strides = [
            a2_edge.data.get_stride(sdfg, self.outer_map_entry.map, dim=i)
            for i in range(2)
        ] + [
            a2_edge.data.get_stride(sdfg, self.inner_map_entry.map, dim=i)
            for i in range(1)
        ]
        if a1_strides == [0, 1, A_desc.shape[1]] and a2_strides == [
            A_desc.shape[1],
            1,
            0,
        ]:
            tmp = a1_edge
            a1_edge = a2_edge
            a2_edge = tmp

        B, B_desc = (b1_edge.data.data, sdfg.arrays[b1_edge.data.data])
        b1_strides = [
            b1_edge.data.get_stride(sdfg, self.outer_map_entry.map, dim=i)
            for i in range(2)
        ] + [
            b1_edge.data.get_stride(sdfg, self.inner_map_entry.map, dim=i)
            for i in range(1)
        ]
        b2_strides = [
            b2_edge.data.get_stride(sdfg, self.outer_map_entry.map, dim=i)
            for i in range(2)
        ] + [
            b2_edge.data.get_stride(sdfg, self.inner_map_entry.map, dim=i)
            for i in range(1)
        ]
        if b1_strides == [0, 1, B_desc.shape[1]] and b2_strides == [
            B_desc.shape[1],
            1,
            0,
        ]:
            tmp = b1_edge
            b1_edge = b2_edge
            b2_edge = tmp

        c_edge = list(state.out_edges_by_connector(tasklet, out_conns[0]))[0]
        C, C_desc = (c_edge.data.data, sdfg.arrays[c_edge.data.data])
        C_node = list(state.out_edges(outer_map_exit))[0].dst

        A_node = None
        B_node = None
        scalar_nodes = []
        for iedge in state.in_edges(self.outer_map_entry):
            if iedge.data.data == A:
                A_node = iedge.src
            elif iedge.data.data == B:
                B_node = iedge.src
            else:
                scalar_nodes.append(iedge.src)

        libnode = Syr2k("_Syr2k_", uplo="L", trans="N", alpha=1.0, beta=0.0)
        state.add_node(libnode)
        libnode.implementation = "MKL"

        state.add_edge(A_node, None, libnode, "_a", dace.Memlet.from_array(A, A_desc))
        state.add_edge(B_node, None, libnode, "_b", dace.Memlet.from_array(B, B_desc))
        state.add_edge(libnode, "_c", C_node, None, dace.Memlet.from_array(C, C_desc))

        state.remove_node(tasklet)

        state.remove_node(outer_map_entry)
        state.remove_node(inner_map_entry)
        state.remove_node(inner_map_exit)
        state.remove_node(outer_map_exit)

        for node in scalar_nodes:
            if state.out_degree(node) == 0 and state.in_degree(node) == 0:
                state.remove_node(node)
