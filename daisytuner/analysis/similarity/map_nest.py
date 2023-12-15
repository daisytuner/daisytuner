# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
from __future__ import annotations

import dace

from typing import List, Generator

from dace.sdfg.state import StateSubgraphView
from dace.sdfg.analysis.cutout import SDFGCutout


class MapNest(StateSubgraphView):
    """
    A map nest is a basic unit of optimization. It describes a subgraph of a state
    defined by a top-level map entry.
    """

    def __init__(self, state: dace.SDFGState, root: dace.nodes.MapEntry) -> None:
        assert MapNest.is_valid(state.parent, state, root)

        # Collect all nodes in subgraph
        map_exit = state.exit_node(root)
        subgraph_nodes = set(state.all_nodes_between(root, map_exit))
        subgraph_nodes.add(root)
        subgraph_nodes.add(map_exit)

        for edge in state.in_edges(root):
            subgraph_nodes.add(edge.src)
        for edge in state.out_edges(map_exit):
            subgraph_nodes.add(edge.dst)

        super().__init__(state, list(subgraph_nodes))
        self._root = root
        self._state = state

    @property
    def sdfg(self) -> dace.SDFG:
        return self._state.parent

    @property
    def state(self) -> dace.SDFGState:
        return self._state

    @property
    def root(self) -> dace.nodes.MapEntry:
        return self._root

    def inputs(self) -> Generator[dace.nodes.AccessNode]:
        for dnode in self.data_nodes():
            if self.in_degree(dnode) == 0:
                yield dnode
            elif self.out_degree(dnode) == 0 and any(
                [
                    e.data is not None and e.data.wcr is not None
                    for e in self.in_edges(dnode)
                ]
            ):
                yield dnode

    def outputs(self) -> Generator[dace.nodes.AccessNode]:
        for dnode in self.data_nodes():
            if self.out_degree(dnode) == 0:
                yield dnode

    def is_data_dependent(self) -> bool:
        raise NotImplementedError

    def as_cutout(self) -> dace.SDFG:
        cutout = SDFGCutout.singlestate_cutout(
            self._graph,
            *self.nodes(),
            make_copy=True,
            make_side_effects_global=False,
            use_alibi_nodes=False,
            symbols_map=self._graph.parent.constants,
        )
        for dnode in cutout.start_state.data_nodes():
            if cutout.start_state.out_degree(dnode) == 0:
                cutout.arrays[dnode.data].transient = False

        return cutout

    @staticmethod
    def is_valid(
        sdfg: dace.SDFG, state: dace.SDFGState, root: dace.nodes.MapEntry
    ) -> bool:
        return isinstance(root, dace.nodes.MapEntry)
