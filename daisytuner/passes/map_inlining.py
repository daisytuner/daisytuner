# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace

from typing import Any, Dict

from dace.transformation import pass_pipeline as ppl
from dace.transformation.interstate import StateFusion

from daisytuner.transformations import InlineMap, StateFission


class MapInlining(ppl.Pass):
    """
    Distributes maps over the states of nested SDFGs inside the maps' scopes.
    """

    CATEGORY: str = "Normalization"

    recursive = True

    def __init__(self) -> None:
        super().__init__()

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.Everything

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Dict[str, Any]) -> int:
        for nsdfg in sdfg.all_sdfgs_recursive():
            for nstate in nsdfg.states():
                for node in nstate.nodes():
                    if not isinstance(node, dace.nodes.MapEntry):
                        continue

                    scope_exit = nstate.exit_node(node)
                    scope_graph: dace.sdfg.ScopeSubgraphView = nstate.scope_subgraph(
                        node
                    )

                    for node_ in scope_graph.nodes():
                        if not isinstance(node_, dace.nodes.NestedSDFG):
                            continue

                        nested_sdfg = node_.sdfg
                        if nested_sdfg.has_cycles():
                            continue

                        for s in nested_sdfg.states():
                            perfectly_nested = set(
                                (
                                    s.entry_node(n)
                                    for n in scope_graph.nodes()
                                    if isinstance(n, dace.nodes.Tasklet)
                                )
                            )
                            perfectly_nested = len(perfectly_nested) == 1
                            if not perfectly_nested:
                                nested_sdfg.apply_transformations_repeated(
                                    StateFission, validate=False
                                )
                                sdfg.validate()
                                break

                        xform = InlineMap()
                        xform.map_entry = node
                        xform.nested_sdfg = node_
                        xform.map_exit = scope_exit
                        if xform.can_be_applied(state=nstate, sdfg=nsdfg, expr_index=0):
                            xform.apply(nstate, nsdfg)
                            sdfg.validate()
                            return len(nsdfg.states())

        sdfg.apply_transformations_repeated(StateFusion, validate=False)
        sdfg.validate()
        return None
