# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import networkx as nx

from typing import Dict

from daisytuner.analysis.similarity.map_nest import MapNest


class BeginState:
    def __init__(self, state: dace.SDFGState) -> None:
        self._state = state

    @property
    def state(self) -> dace.SDFGState:
        return self._state


class EndState:
    def __init__(self, state: dace.SDFGState) -> None:
        self._state = state

    @property
    def state(self) -> dace.SDFGState:
        return self._state


class EmptyNode:
    pass


class GraphOfMaps(nx.DiGraph):
    """
    Acyclic scheduling graph
    """

    def __init__(self, sdfg: dace.SDFG) -> None:
        super().__init__()

        assert GraphOfMaps.is_valid(sdfg)
        self._sdfg = sdfg

        # 1. Init nodes:
        #   - A control node for each state
        #   - A node for each map nest
        self._map_nests = {}
        self._begin_nodes = {}
        self._end_nodes = {}
        map_nests_per_state = {state: {} for state in self._sdfg.states()}
        for state in self._sdfg.states():
            # State's control node
            begin_node = BeginState(state=state)
            self._begin_nodes[state] = begin_node
            self.add_node(begin_node)

            end_node = EndState(state=state)
            self._end_nodes[state] = end_node
            self.add_node(end_node)

            for node in state.nodes():
                if not isinstance(node, dace.nodes.MapEntry):
                    continue
                if not state.entry_node(node) is None:
                    continue

                # Create map nest
                map_nest = MapNest(state=state, root=node)
                self._map_nests[node] = map_nest
                self.add_node(node)

                map_nests_per_state[state][node] = map_nest

        # Init edges: Define scheduling dependencies
        visited = set()
        for state in sdfg.topological_sort(sdfg.start_state):
            # a. Data-dependencies: Connect maps w.r.t data dependencies
            for node in map_nests_per_state[state]:
                exit_node = state.exit_node(node)
                for oedge in state.out_edges(exit_node):
                    assert isinstance(oedge.dst, dace.nodes.AccessNode)
                    for oedge_ in state.out_edges(oedge.dst):
                        assert isinstance(oedge_.dst, dace.nodes.MapEntry)
                        if not self.has_edge(u=node, v=oedge_.dst):
                            self.add_edge(u_of_edge=node, v_of_edge=oedge_.dst)

            # b. Connect control nodes
            begin_node = self._begin_nodes[state]
            end_node = self._end_nodes[state]

            # Case: Empty state
            if not map_nests_per_state[state]:
                empty_node = EmptyNode()
                self.add_node(empty_node)
                self.add_edge(u_of_edge=begin_node, v_of_edge=empty_node)
                self.add_edge(u_of_edge=empty_node, v_of_edge=end_node)
            else:
                # Case: non-empty state
                for node in map_nests_per_state[state]:
                    if self.in_degree(node) == 0:
                        self.add_edge(u_of_edge=begin_node, v_of_edge=node)
                    if self.out_degree(node) == 0:
                        self.add_edge(u_of_edge=node, v_of_edge=end_node)

            for succ in self._sdfg.successors(state):
                if succ in visited:
                    # No cyclic scheduling dependencies
                    continue

                self.add_edge(u_of_edge=end_node, v_of_edge=self._begin_nodes[succ])

            visited.add(state)

    @property
    def sdfg(self) -> dace.SDFG:
        return self._sdfg

    @property
    def map_nests(self) -> Dict[dace.nodes.MapEntry, MapNest]:
        return self._map_nests

    @property
    def begin_nodes(self) -> Dict[dace.SDFGState, BeginState]:
        self._begin_nodes

    @property
    def end_nodes(self) -> Dict[dace.SDFGState, EndState]:
        self._end_nodes

    @staticmethod
    def is_valid(sdfg: dace.SDFG) -> bool:
        for state in sdfg.states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.AccessNode):
                    continue
                if state.entry_node(node) is not None:
                    continue

                if not isinstance(node, dace.nodes.MapEntry):
                    return False

        return True
