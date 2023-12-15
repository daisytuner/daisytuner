# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import copy

from dace.sdfg import SDFG
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.state import SDFGState
from dace.transformation import transformation
from dace.properties import make_properties


@make_properties
class MapWrapping(transformation.SingleStateTransformation):
    code_node = transformation.PatternNode(nodes.CodeNode)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.code_node)]

    def can_be_applied(
        self, state: dace.SDFGState, expr_index: int, sdfg: dace.SDFG, permissive=False
    ):
        if state.parent.parent_nsdfg_node is not None:
            return False

        if state.entry_node(self.code_node) is not None:
            return False

        return True

    def apply(self, state: SDFGState, sdfg: SDFG):
        node = self.code_node

        me, mx = state.add_map(
            node.label + "_map",
            {node.label + "__mapi": "0:1"},
            schedule=dace.dtypes.ScheduleType.Default,
        )

        # Store in/out edges in lists so that they don't get corrupted when they are removed from the graph.
        in_edges = list(state.in_edges(node))
        out_edges = list(state.out_edges(node))
        me.in_connectors = {
            ("IN_" + e.dst_conn): None for e in in_edges if e.dst_conn is not None
        }
        me.out_connectors = {
            ("OUT_" + e.dst_conn): None for e in in_edges if e.dst_conn is not None
        }
        mx.in_connectors = {
            ("IN_" + e.src_conn): None for e in out_edges if e.src_conn is not None
        }
        mx.out_connectors = {
            ("OUT_" + e.src_conn): None for e in out_edges if e.src_conn is not None
        }

        # Create memlets through map
        for e in in_edges:
            state.remove_edge(e)

            dst_conn = "IN_" + e.dst_conn if e.dst_conn is not None else None
            src_conn = "OUT_" + e.dst_conn if e.dst_conn is not None else None
            state.add_edge(e.src, e.src_conn, me, dst_conn, copy.deepcopy(e.data))
            state.add_edge(me, src_conn, e.dst, e.dst_conn, copy.deepcopy(e.data))
        for e in out_edges:
            state.remove_edge(e)

            dst_conn = "IN_" + e.src_conn if e.src_conn is not None else None
            src_conn = "OUT_" + e.src_conn if e.src_conn is not None else None
            state.add_edge(e.src, e.src_conn, mx, dst_conn, copy.deepcopy(e.data))
            state.add_edge(mx, src_conn, e.dst, e.dst_conn, copy.deepcopy(e.data))

        # Map without inputs
        if len(in_edges) == 0:
            state.add_nedge(me, node, dace.Memlet())
