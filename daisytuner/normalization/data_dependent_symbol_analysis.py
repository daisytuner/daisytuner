# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import ast
import copy
import dace

from typing import Dict, Any

from dace.transformation import pass_pipeline as ppl


class DataDependentSymbolAnalysis(ppl.Pass):
    """ """

    CATEGORY: str = "Analysis"

    def __init__(self, initial_symbols: Dict[str, dace.Memlet] = None) -> None:
        super().__init__()
        if initial_symbols is None:
            self._initial_symbols = {}
        else:
            self._initial_symbols = initial_symbols

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & (ppl.Modifies.InterstateEdges | ppl.Modifies.States)

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Dict[str, Any]) -> int:
        if sdfg.in_degree(sdfg.start_state) > 0:
            return None

        # Bookkeeping on active symbol to memlet mapping
        results = {}
        results[sdfg.start_state] = self._initial_symbols

        # Traverse control-flow graph
        start_state = sdfg.start_state
        for state in sdfg.topological_sort(start_state):
            if state == sdfg.start_state:
                state_results = self._initial_symbols
            else:
                # Collect active mappings
                active_symbols = {}
                predecessors = set(sdfg.predecessors(state))
                for pred in predecessors:
                    if pred not in results:
                        continue

                    for sym, memlet in results[pred].items():
                        if sym not in active_symbols:
                            active_symbols[sym] = {}
                        active_symbols[sym][pred] = memlet

                # Collect all updates from branches
                updated_symbols = {}
                for iedge in sdfg.in_edges(state):
                    if not iedge.data.assignments:
                        continue

                    for sym, assign in iedge.data.assignments.items():
                        # Check if expression is a memlet
                        expr = ast.parse(assign).body[0].value
                        if not isinstance(expr, ast.Subscript):
                            return None

                        array = expr.value.id
                        slice = expr.slice
                        if not array in sdfg.arrays:
                            return None

                        # Valid to parse
                        memlet = dace.Memlet(expr=assign)
                        if sym not in updated_symbols:
                            updated_symbols[sym] = {}
                        if iedge.src not in updated_symbols[sym]:
                            updated_symbols[sym][iedge.src] = []

                        updated_symbols[sym][iedge.src].append(memlet)

                # Update currently active symbols
                all_symbols = set(active_symbols.keys()).union(updated_symbols.keys())
                state_results = {}
                for sym in all_symbols:
                    mapping = {}
                    for pred in predecessors:
                        if sym in active_symbols:
                            if pred in active_symbols[sym]:
                                mapping[pred] = active_symbols[sym][pred]
                                continue

                        mapping[pred] = None

                    for pred in predecessors:
                        if sym in updated_symbols:
                            if pred in updated_symbols[sym]:
                                branch_mappings = updated_symbols[sym][pred]

                                if not all(
                                    x == branch_mappings[0]
                                    and x.data == branch_mappings[0].data
                                    for x in branch_mappings
                                ):
                                    # Case 1: Updates are not consistent. Hence, invalidate symbol

                                    mapping[pred] = None
                                elif len(branch_mappings) == len(
                                    sdfg.edges_between(pred, state)
                                ):
                                    # Case 2: Consistent update across all branches. Hence, overwrite previous values

                                    mapping[pred] = branch_mappings[0]
                                else:
                                    if branch_mappings[0] == mapping[pred]:
                                        # Case 3: Memlet is not changed
                                        mapping[pred] = branch_mappings[0]
                                    else:
                                        # Case 4: Memlet is updated in some branches, but contradicts previous values
                                        mapping[pred] = None

                    possible_values = list(mapping.values())
                    assert len(possible_values) == len(predecessors)

                    if any(val is None for val in possible_values):
                        continue
                    if not all(
                        x == possible_values[0] and x.data == possible_values[0].data
                        for x in possible_values
                    ):
                        continue

                    state_results[sym] = possible_values[0]

            # Store state results
            results[state] = state_results

            # Traverse nested SDFGs recursively
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    pipeline = DataDependentSymbolAnalysis(
                        initial_symbols=copy.deepcopy(state_results)
                    )
                    nested_results = pipeline.apply_pass(node.sdfg, pipeline_results)
                    results[node.sdfg] = nested_results

        results = {sdfg: results}
        if DataDependentSymbolAnalysis.__name__ not in pipeline_results:
            return results

        prev_report = pipeline_results[DataDependentSymbolAnalysis.__name__]
        if prev_report == results:
            return None
        else:
            return results
