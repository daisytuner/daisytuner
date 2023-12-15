# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace

from dace.sdfg import utils as sdutil
from dace.sdfg import nodes, SDFG
from dace.sdfg.state import SDFGState, StateSubgraphView
from dace.transformation import transformation
from dace.properties import make_properties
from typing import List, Tuple


@make_properties
class MapUntiling(transformation.SingleStateTransformation):
    """Reverses the tiling of a single map."""

    outer_map_entry = transformation.PatternNode(nodes.MapEntry)
    inner_map_entry = transformation.PatternNode(nodes.MapEntry)

    # Finds paramter pairs that represent a map tiling.
    def find_pairs(self, state: SDFGState) -> List[Tuple[str, str]]:
        outer_map = self.outer_map_entry.map
        inner_map = self.inner_map_entry.map

        # Create dict of usage counts for each symbol.
        outer_map_exit = state.exit_node(self.outer_map_entry)
        subgraph_nodes = set(
            state.all_nodes_between(self.outer_map_entry, outer_map_exit)
        )
        subgraph_nodes.add(self.outer_map_entry)
        subgraph_nodes.add(outer_map_exit)
        view: StateSubgraphView = StateSubgraphView(state, subgraph_nodes)

        usage_count = {}
        for node in view.nodes():
            for symbol in node.free_symbols:
                usage_count[str(symbol)] = usage_count.get(str(symbol), 0) + 1

        for edge in view.edges():
            for symbol in edge.data.free_symbols:
                usage_count[str(symbol)] = usage_count.get(str(symbol), 0) + 1

        # Outer map parameters must have a block size larger than one and appear exactly 3 times (incoming edge, outgoing edge, inner map).
        outer_params = {
            str(p): r
            for p, r in zip(outer_map.params, outer_map.range)
            if r[2] > 1 and usage_count.get(str(p), 0) == 3
        }

        # Inner map parameters must depend on outer map parameters, have step size of one,and the block size matches.
        result = []

        for p, r in zip(inner_map.params, inner_map.range):
            if str(r[0]) in outer_params.keys() and r[2] == 1:
                outer_range = outer_params[str(r[0])]

                # Convert expressions to symbolic form.
                expr = dace.symbolic.SymExpr(
                    f"Min({str(outer_range[1])}, {str(r[0])} + {str(outer_range[2])} - 1) - {str(r[1].expr)}"
                )

                # Check if the expressions are equivalent.
                if str(dace.symbolic.simplify_ext(expr)) == "0":
                    result.append((str(p), str(r[0])))

        return result

    # Computes new parameters with ranges.
    def update_map_ranges(self, pairs: List[Tuple[str, str]]):
        outer_map = self.outer_map_entry.map
        inner_map = self.inner_map_entry.map

        # Compute new outer parameters with ranges.
        outer_map_params = list(zip(outer_map.params, outer_map.range))
        outer_map_params_dict = {str(key): value for key, value in outer_map_params}

        new_outer_params = [
            (p, r) for p, r in outer_map_params if str(p) not in [t[1] for t in pairs]
        ]

        # Compute new inner parameters with ranges.
        new_inner_params = []
        for i, param in enumerate(inner_map.params):
            outer_param = dict(pairs).get(str(param))
            if not outer_param:
                new_inner_params.append((param, inner_map.range[i]))
                continue

            range = outer_map_params_dict.get(outer_param)
            if not range:
                new_inner_params.append((param, inner_map.range[i]))
                continue

            new_inner_params.append((param, (range[0], range[1], 1)))

        # Update inner and outer map parameters and ranges
        self.outer_map_entry.map.params = [p for p, r in new_outer_params]
        self.outer_map_entry.map.range = [r for p, r in new_outer_params]
        self.inner_map_entry.map.range = [r for p, r in new_inner_params]

    # Updates indices of the inner map's incoming edges.
    def update_incoming_edges(
        self, state: dace.SDFGState, pairs: List[Tuple[str, str]]
    ):
        for in_connector, out_connector in zip(
            self.outer_map_entry.in_connectors,
            self.outer_map_entry.out_connectors,
        ):
            # Assume there is exactly one incoming edge per in_connector.
            in_edge = list(
                state.in_edges_by_connector(self.outer_map_entry, in_connector)
            )[0]

            for out_edge in state.out_edges_by_connector(
                self.outer_map_entry, out_connector
            ):
                # FIXME: Handle all possible types.
                if not (
                    isinstance(out_edge.data.src_subset, dace.subsets.Range)
                    and isinstance(in_edge.data.src_subset, dace.subsets.Range)
                ):
                    continue

                for i, elem in enumerate(out_edge.data.src_subset):
                    # Parameter eliminated, copy incoming range.
                    if str(elem[0]) in [r for p, r in pairs]:
                        out_edge.data.src_subset[i] = in_edge.data.src_subset[i]

    # Updates indices of the inner map's outgoing edges.
    def update_outgoing_edges(
        self, state: dace.SDFGState, pairs: List[Tuple[str, str]]
    ):
        outer_map_exit = state.exit_node(self.outer_map_entry)

        for in_connector, out_connector in zip(
            outer_map_exit.in_connectors,
            outer_map_exit.out_connectors,
        ):
            # Assume there is exactly one outgoing edge per out_connector.
            out_edge = list(
                state.out_edges_by_connector(outer_map_exit, out_connector)
            )[0]

            for in_edge in state.in_edges_by_connector(outer_map_exit, in_connector):
                # FIXME: Handle all possible types.
                if not (
                    isinstance(in_edge.data.dst_subset, dace.subsets.Range)
                    and isinstance(out_edge.data.dst_subset, dace.subsets.Range)
                ):
                    continue

                for i, elem in enumerate(in_edge.data.dst_subset):
                    # Parameter eliminated, copy outgoing range.
                    if str(elem[0]) in [r for p, r in pairs]:
                        in_edge.data.dst_subset[i] = out_edge.data.dst_subset[i]

    # Removes empty outer map.
    def remove_empty_outer_map(
        self,
        state: dace.SDFGState,
    ):
        # Outer map has used parameters.
        if self.outer_map_entry.map.params:
            return

        outer_map_exit = state.exit_node(self.outer_map_entry)
        inner_map_exit = state.exit_node(self.inner_map_entry)

        # Perform type checks
        assert isinstance(
            outer_map_exit, dace.sdfg.nodes.MapExit
        ), "outer_map_exit must be of type MapExit"
        assert isinstance(
            inner_map_exit, dace.sdfg.nodes.MapExit
        ), "inner_map_exit must be of type MapExit"

        sdutil.merge_maps(
            state,
            self.outer_map_entry,
            outer_map_exit,
            self.inner_map_entry,
            inner_map_exit,
        )

    @classmethod
    def expressions(cls):
        return [
            sdutil.node_path_graph(
                cls.outer_map_entry,
                cls.inner_map_entry,
            )
        ]

    def can_be_applied(
        self, state: dace.SDFGState, expr_index: int, sdfg: dace.SDFG, permissive=False
    ):
        # Check if we can identify any tiling pairs.
        return len(self.find_pairs(state)) > 0

    def apply(self, state: SDFGState, sdfg: SDFG):
        # Find all parameter pairs to untile.
        pairs = self.find_pairs(state)

        # Update outer and inner map ranges.
        self.update_map_ranges(pairs)

        # Update indices of the inner map's incoming edges.
        self.update_incoming_edges(state, pairs)

        # Update indices of the inner map's outgoing edges.
        self.update_outgoing_edges(state, pairs)

        # Clean up empty outer map.
        self.remove_empty_outer_map(state)
