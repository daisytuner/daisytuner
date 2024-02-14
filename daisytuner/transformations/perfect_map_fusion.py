# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace

from typing import Any

from dace.sdfg import SDFG, SDFGState, nodes as nds
from dace.sdfg import utils as sdutil
from dace.properties import make_properties
from dace.transformation import transformation
from dace.transformation.dataflow import OTFMapFusion, MapFusion


@make_properties
class PerfectMapFusion(transformation.SingleStateTransformation):

    first_map_exit = transformation.PatternNode(nds.ExitNode)
    array = transformation.PatternNode(nds.AccessNode)
    second_map_entry = transformation.PatternNode(nds.EntryNode)

    @classmethod
    def expressions(cls):
        return [
            sdutil.node_path_graph(cls.first_map_exit, cls.array, cls.second_map_entry)
        ]

    def can_be_applied(
        self,
        graph: dace.SDFGState,
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        # Chain: One-to-one producer-consumer relation
        if len(set(edge.dst for edge in graph.out_edges(self.array))) > 1:
            return False

        # Perfectly nested
        first_map_entry = graph.entry_node(self.first_map_exit)
        first_subgraph = graph.scope_subgraph(
            first_map_entry, include_entry=False, include_exit=False
        )
        if any(
            [isinstance(node, dace.nodes.NestedSDFG) for node in first_subgraph.nodes()]
        ):
            return False
        first_depths = set(
            [
                PerfectMapFusion.scope_depth(node, graph)
                for node in first_subgraph.nodes()
                if isinstance(node, (dace.nodes.Tasklet, dace.nodes.NestedSDFG))
            ]
        )
        if len(first_depths) > 1:
            return False

        second_subgraph = graph.scope_subgraph(
            self.second_map_entry, include_entry=False, include_exit=False
        )
        if any(
            [
                isinstance(node, dace.nodes.NestedSDFG)
                for node in second_subgraph.nodes()
            ]
        ):
            return False
        second_depths = set(
            [
                PerfectMapFusion.scope_depth(node, graph)
                for node in second_subgraph.nodes()
                if isinstance(node, (dace.nodes.Tasklet, dace.nodes.NestedSDFG))
            ]
        )
        if len(second_depths) > 1:
            return False

        if second_depths != first_depths:
            return False

        # Shadowing: Safe to remove array?
        # TODO: Implement as pass and compute shadowed set
        if not sdfg.arrays[self.array.data].transient:
            return False
        for state in sdfg.states():
            for dnode in state.data_nodes():
                if dnode == self.array:
                    continue

                if dnode.data == self.array.data:
                    return False

        xform = OTFMapFusion()
        xform.first_map_exit = self.first_map_exit
        xform.array = self.array
        xform.second_map_entry = self.second_map_entry
        if not xform.can_be_applied(graph, expr_index, sdfg):
            xform = MapFusion()
            xform.first_map_exit = self.first_map_exit
            xform.array = self.array
            xform.second_map_entry = self.second_map_entry
            if not xform.can_be_applied(graph, expr_index, sdfg):
                return False

        return True

    def apply(self, graph: SDFGState, sdfg: SDFG) -> None:
        xform = OTFMapFusion()
        xform.first_map_exit = self.first_map_exit
        xform.array = self.array
        xform.second_map_entry = self.second_map_entry
        if xform.can_be_applied(graph, self.expr_index, sdfg):
            xform.apply(graph, sdfg)
            return

        xform = MapFusion()
        xform.first_map_exit = self.first_map_exit
        xform.array = self.array
        xform.second_map_entry = self.second_map_entry
        if xform.can_be_applied(graph, self.expr_index, sdfg):
            xform.apply(graph, sdfg)
            return

    @staticmethod
    def scope_depth(node: dace.nodes.Node, state: dace.SDFGState) -> int:
        depth = 0
        cur = node
        while state.entry_node(cur) is not None:
            depth += 1
            cur = state.entry_node(cur)

        return depth
