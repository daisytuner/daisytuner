# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import re

from dace.sdfg import utils as sdutil
from dace.sdfg.state import SDFGState
from dace.properties import make_properties
from dace.transformation.transformation import SingleStateTransformation, PatternNode

from daisytuner.library.blas import Gemv


@make_properties
class GEMV(SingleStateTransformation):
    map_entry = PatternNode(dace.nodes.MapEntry)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry)]

    def can_be_applied(
        self, state: dace.SDFGState, expr_index: int, sdfg: dace.SDFG, permissive=False
    ):
        if len(self.map_entry.map.params) != 2:
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

        C, C_desc = (c_edge.data.data, sdfg.arrays[c_edge.data.data])
        if len(C_desc.shape) != 1:
            return False

        c_strides = [
            c_edge.data.get_stride(sdfg, self.map_entry.map, dim=i) for i in range(2)
        ]
        if c_edge.data.wcr is None or "+" not in c_edge.data.wcr or c_strides != [1, 0]:
            return False

        A, A_desc = (a_edge.data.data, sdfg.arrays[a_edge.data.data])
        B, B_desc = (b_edge.data.data, sdfg.arrays[b_edge.data.data])
        if len(A_desc.shape) == 2:
            if len(B_desc.shape) != 1:
                return False
        elif len(A_desc.shape) == 1:
            if len(B_desc.shape) != 2:
                return False

            tmp = (A, A_desc, a_edge)
            A, A_desc, a_edge = (B, B_desc, b_edge)
            B, B_desc, b_edge = tmp
        else:
            return False

        if A_desc.dtype != dace.float64 and A_desc.dtype != dace.float32:
            return False
        if B_desc.dtype != dace.float64 and B_desc.dtype != dace.float32:
            return False
        if C_desc.dtype != dace.float64 and C_desc.dtype != dace.float32:
            return False

        op = tasklet.code.as_string
        if tasklet.language == dace.Language.CPP:
            op = op.replace(";", "")

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
            or re.match(p1_, op) is not None
            or re.match(p2_, op) is not None
        ):
            return True

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

        if len(A_desc.shape) == 1:
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

        state.remove_node(map_exit)
        state.remove_node(self.map_entry)
        state.remove_node(tasklet)

        libnode = Gemv("_Gemv_", transA=False, alpha=1.0, beta=0.0)
        state.add_node(libnode)

        state.add_edge(A_node, None, libnode, "_A", dace.Memlet.from_array(A, A_desc))
        state.add_edge(B_node, None, libnode, "_x", dace.Memlet.from_array(B, B_desc))
        state.add_edge(libnode, "_y", C_node, None, dace.Memlet.from_array(C, C_desc))
