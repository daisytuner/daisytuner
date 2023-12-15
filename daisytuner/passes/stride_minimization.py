# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import itertools
import math
import dace

from typing import Any, Dict, Optional
from dace.memlet import Memlet
from dace.sdfg import SDFGState, nodes as nd, ScopeSubgraphView
from dace.transformation import pass_pipeline as ppl


class StrideMinimization(ppl.ScopePass):
    """
    Permutes maps such that the strides are minimized.
    """

    CATEGORY: str = "Normalization"

    def __init__(self, assumptions: Dict = None) -> None:
        super().__init__()
        if assumptions is None:
            self.assumptions = {}
        else:
            self.assumptions = assumptions

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Scopes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.Scopes

    def apply(
        self,
        scope: nd.EntryNode,
        state: SDFGState,
        pipeline_results: Dict[str, Any],
    ) -> Optional[Any]:
        if not isinstance(scope, (nd.MapEntry, nd.ConsumeEntry)):
            # We only handle maps or consume scopes and descendants thereof.
            return None

        scopegraph: ScopeSubgraphView = state.scope_subgraph(scope)
        scope_exit = state.exit_node(scope)

        s_map = scope.map
        n_params = s_map.get_param_num()

        # Overapproximation
        parent_maps = [scope]
        while state.entry_node(parent_maps[-1]) is not None:
            parent_maps.append(state.entry_node(parent_maps[-1]))

        symbols = {**state.parent.constants, **self.assumptions}
        for map_entry in parent_maps[::-1]:
            for i, param in enumerate(map_entry.map.params):
                b, e, s = map_entry.map.range[i]
                try:
                    symbols[param] = dace.symbolic.evaluate(e, symbols=symbols)
                except:
                    continue

        cur_min = [math.inf] * n_params
        min_perm = None
        orig_perm = s_map.params

        # Find this scope's reads and writes.
        border_edges = set()
        scope_data = set()
        for i_mem in state.in_edges(scope):
            if i_mem.data.data is None:
                continue

            border_edges.add(i_mem)
            scope_data.add(i_mem.data.data)
        for o_mem in state.out_edges(scope_exit):
            if o_mem.data.data is None:
                continue

            border_edges.add(o_mem)
            scope_data.add(o_mem.data.data)

        # Brute-force enumerate all possible map permutations to find the one
        # that minimizes the strides of the scope from the inside out (last dim
        # first).
        for _perm in itertools.permutations(s_map.params):
            perm = list(_perm)
            # Apply the permutation that's currently being evaluated.
            scope.range.ranges = [
                r
                for list_param in perm
                for map_param, r in zip(s_map.params, scope.range.ranges)
                if list_param == map_param
            ]
            scope.map.params = perm

            strides_list = [{} for _ in range(n_params)]
            strides_sum = [0 for _ in range(n_params)]

            for e in scopegraph.edges():
                if not isinstance(e.data, Memlet):
                    continue
                if e not in border_edges:
                    for i in range(1, n_params + 1):
                        dim = -i
                        stride = e.data.get_stride(state.parent, scope, dim)
                        try:
                            stride = dace.symbolic.evaluate(stride, symbols=symbols)
                        except:
                            stride = math.inf
                        strides_list[dim][e.data.data] = stride

            for i in range(n_params):
                for dt in scope_data:
                    strides_sum[i] += strides_list[i][dt]

            # Check if the permutation is better than the current best one.
            is_less = False
            for i in range(1, n_params + 1):
                dim = -i
                if strides_sum[dim] < cur_min[dim]:
                    is_less = True
                elif strides_sum[dim] == cur_min[dim]:
                    continue
                break
            if is_less or min_perm is None:
                cur_min = strides_sum
                min_perm = perm

        # Apply the minimum-stride-permutation.
        scope.range.ranges = [
            r
            for list_param in min_perm
            for map_param, r in zip(s_map.params, scope.range.ranges)
            if list_param == map_param
        ]
        scope.map.params = min_perm
        if orig_perm != min_perm:
            return (min_perm, orig_perm)
        return None

    @staticmethod
    def is_normalized(sdfg: dace.SDFG, assumptions: Dict = None):
        if assumptions is None:
            assumptions = {}

        for nsdfg in sdfg.all_sdfgs_recursive():
            for state in nsdfg.states():
                for node in state.nodes():
                    if not isinstance(node, dace.nodes.MapEntry):
                        continue

                    scope_exit = state.exit_node(node)
                    scopegraph = state.scope_subgraph(node)

                    s_map = node.map
                    n_params = s_map.get_param_num()
                    orig_perm = s_map.params
                    cur_min = [math.inf * n_params]
                    min_perm = None

                    # Overapproximation
                    parent_maps = [node]
                    while state.entry_node(parent_maps[-1]) is not None:
                        parent_maps.append(state.entry_node(parent_maps[-1]))

                    symbols = {**state.parent.constants, **assumptions}
                    for map_entry in parent_maps[::-1]:
                        for i, param in enumerate(map_entry.map.params):
                            b, e, s = map_entry.map.range[i]
                            try:
                                symbols[param] = dace.symbolic.evaluate(
                                    e, symbols=symbols
                                )
                            except:
                                symbols[param] = math.inf

                    # Find this scope's reads and writes.
                    border_edges = set()
                    scope_data = set()
                    for i_mem in state.in_edges(node):
                        if i_mem.data.data is None:
                            continue

                        border_edges.add(i_mem)
                        scope_data.add(i_mem.data.data)
                    for o_mem in state.out_edges(scope_exit):
                        if o_mem.data.data is None:
                            continue

                        border_edges.add(o_mem)
                        scope_data.add(o_mem.data.data)

                    # Check permutations
                    for _perm in itertools.permutations(s_map.params):
                        perm = list(_perm)
                        node.range.ranges = [
                            r
                            for list_param in perm
                            for map_param, r in zip(s_map.params, node.range.ranges)
                            if list_param == map_param
                        ]
                        node.map.params = perm

                        strides_list = [{} for _ in range(n_params)]
                        strides_sum = [0 for _ in range(n_params)]

                        for e in scopegraph.edges():
                            if not isinstance(e.data, Memlet):
                                continue
                            if e not in border_edges:
                                for i in range(1, n_params + 1):
                                    dim = -i
                                    stride = e.data.get_stride(state.parent, node, dim)
                                    try:
                                        stride = dace.symbolic.evaluate(
                                            stride, symbols=symbols
                                        )
                                    except:
                                        stride = math.inf
                                    strides_list[dim][e.data.data] = stride

                        for i in range(n_params):
                            for dt in scope_data:
                                strides_sum[i] += strides_list[i][dt]

                        # Check if the permutation is better than the current best one.
                        is_less = False
                        for i in range(1, n_params + 1):
                            dim = -i
                            if strides_sum[dim] < cur_min[dim]:
                                is_less = True
                            elif strides_sum[dim] == cur_min[dim]:
                                continue
                            break
                        if is_less or min_perm is None:
                            cur_min = strides_sum
                            min_perm = perm

                    # Apply original perm
                    node.range.ranges = [
                        r
                        for list_param in orig_perm
                        for map_param, r in zip(s_map.params, node.range.ranges)
                        if list_param == map_param
                    ]
                    node.map.params = orig_perm

                    if orig_perm != min_perm:
                        return False

        return True
