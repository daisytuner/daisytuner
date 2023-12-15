# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import networkx as nx

from typing import Generator, Tuple, Dict

from daisytuner.analysis.similarity.map_nest import MapNest


class BarrierNode:
    pass


class GraphOfMaps(nx.DiGraph):
    def __init__(self, sdfg: dace.SDFG) -> None:
        super().__init__()
        self._sdfg = sdfg

        # Collect map nests
        self._map_nests = {}
        map_nests_per_state = {state: {} for state in self._sdfg.states()}
        for state in self._sdfg.states():
            for node in state.nodes():
                if not isinstance(node, dace.nodes.MapEntry):
                    continue
                if not state.entry_node(node) is None:
                    continue

                # Create map nest
                map_nest = MapNest(state=state, root=node)
                map_nests_per_state[state][node] = map_nest
                self._map_nests[node] = map_nest
                self.add_node(node)

        # Connect map nests within each state based on data dependencies
        for state in map_nests_per_state:
            for node in map_nests_per_state[state]:
                exit_node = state.exit_node(node)
                for oedge in state.out_edges(exit_node):
                    assert isinstance(oedge.dst, dace.nodes.AccessNode)
                    for oedge_ in state.out_edges(oedge.dst):
                        assert isinstance(oedge_.dst, dace.nodes.MapEntry)

                        if not self.has_edge(u=node, v=oedge_.dst):
                            self.add_edge(u_of_edge=node, v_of_edge=oedge_.dst)

        # Collect sources and sinks of each state
        sources_per_state = {}
        sinks_per_state = {}
        for state in self._sdfg.states():
            sources_per_state[state] = set()
            sinks_per_state[state] = set()
            for node in map_nests_per_state[state]:
                if self.in_degree(node) == 0:
                    sources_per_state[state].add(node)
                if self.out_degree(node) == 0:
                    sinks_per_state[state].add(node)

        # Add dependencies between states
        synchronization_points = {}
        for barrier, current_state in GraphOfMaps.sdfg_walker(self._sdfg):
            if barrier is None:
                # No synchronization point, connect to previous states by data dependencies
                for pred in self._sdfg.predecessor_states(current_state):
                    for sink in sinks_per_state[pred]:
                        first_outputs = {
                            dnode.data for dnode in self._map_nests[sink].outputs()
                        }
                        for source in sources_per_state[current_state]:
                            if self.has_edge(u=sink, v=source):
                                continue

                            second_inputs = {
                                dnode.data for dnode in self._map_nests[source].inputs()
                            }
                            second_outputs = {
                                dnode.data
                                for dnode in self._map_nests[source].outputs()
                            }
                            if first_outputs.intersection(
                                (second_inputs | second_outputs)
                            ):
                                self.add_edge(u_of_edge=sink, v_of_edge=source)
            else:
                # Synchronization point, connect sinks_per_state -> barrier -> sources_per_state

                # Add barrier node
                self.add_node(barrier)
                synchronization_points[current_state] = barrier

                # Connect barrier to sources_per
                for source in sources_per_state[current_state]:
                    self.add_edge(u_of_edge=barrier, v_of_edge=source)

                # Connect sinks_per_state to barrier
                for pred in self._sdfg.predecessor_states(current_state):
                    for sink in sinks_per_state[pred]:
                        self.add_edge(u_of_edge=sink, v_of_edge=barrier)

    @property
    def sdfg(self) -> dace.SDFG:
        return self._sdfg

    @property
    def map_nests(self) -> Dict[dace.nodes.MapEntry, MapNest]:
        return self._map_nests

    @staticmethod
    def sdfg_walker(
        sdfg: dace.SDFG,
    ) -> Generator[Tuple[BarrierNode, dace.SDFGState], None, None]:
        for state in sdfg.topological_sort(sdfg.start_state):
            in_edges = sdfg.in_edges(state)
            if len(in_edges) == 0:
                yield (None, state)
            elif len(in_edges) > 1:
                yield (BarrierNode(), state)
            else:
                iedge = in_edges[0]
                needs_barrier = (
                    not iedge.data.is_unconditional() or iedge.data.assignments
                )
                if needs_barrier:
                    yield (BarrierNode(), state)
                else:
                    yield (None, state)
