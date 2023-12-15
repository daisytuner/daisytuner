# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace

from typing import Any, Dict, Optional
from dace.transformation import pass_pipeline as ppl
from dace.transformation.dataflow import MapFission


class MaximalMapFission(ppl.ScopePass):
    """
    Fissions map nests as maximal as possible to simplify memory accesses.
    """

    CATEGORY: str = "Normalization"

    def __init__(self) -> None:
        super().__init__()

    def modifies(self) -> ppl.Modifies:
        return (
            ppl.Modifies.Scopes
            | ppl.Modifies.Descriptors
            | ppl.Modifies.AccessNodes
            | ppl.Modifies.Memlets
            | ppl.Modifies.Tasklets
        )

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.Scopes

    def apply(
        self,
        scope: dace.nodes.EntryNode,
        state: dace.SDFGState,
        pipeline_results: Dict[str, Any],
    ) -> Optional[Any]:
        if not isinstance(scope, dace.nodes.MapEntry):
            return None

        sdfg = state.parent
        scope_exit = state.exit_node(scope)
        scope_graph: dace.sdfg.ScopeSubgraphView = state.scope_subgraph(
            scope, include_entry=False, include_exit=False
        )

        # Stop criterion
        perfectly_nested = set(
            (
                state.entry_node(n)
                for n in scope_graph.nodes()
                if isinstance(n, dace.nodes.Tasklet)
            )
        )
        perfectly_nested = len(perfectly_nested) == 1
        inedges = state.in_edges(scope_exit)
        inarrays = set([edge.data.data for edge in inedges if edge.data is not None])
        if perfectly_nested and len(inarrays) <= 1:
            return None

        xform = MapFission()
        xform.map_entry = scope
        xform.expr_index = 0
        if xform.can_be_applied(graph=state, sdfg=sdfg, expr_index=0):
            xform.apply(state, sdfg)
            sdfg.validate()
            return len(inarrays)

        return None

    @staticmethod
    def is_normalized(sdfg: dace.SDFG):
        for nsdfg in sdfg.all_sdfgs_recursive():
            for state in nsdfg.states():
                for node in state.nodes():
                    if not isinstance(node, dace.nodes.MapEntry):
                        continue

                    scope_exit = state.exit_node(node)
                    scope_graph: dace.sdfg.ScopeSubgraphView = state.scope_subgraph(
                        node, include_entry=False, include_exit=False
                    )

                    # Multiple write memlets
                    inedges = state.in_edges(scope_exit)
                    inarrays = set(
                        [edge.data.data for edge in inedges if edge.data is not None]
                    )
                    if len(inarrays) > 1:
                        return False

                    # Imperfectly nested tasklets
                    perfectly_nested = set(
                        (
                            state.entry_node(n)
                            for n in scope_graph.nodes()
                            if isinstance(n, dace.nodes.Tasklet)
                        )
                    )
                    perfectly_nested = len(perfectly_nested) <= 1
                    if not perfectly_nested:
                        return False

        return True
