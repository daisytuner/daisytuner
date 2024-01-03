# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import networkx as nx

from dace import sdfg, memlet
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.state import SDFGState
from dace.transformation import transformation


class ComponentFusion(transformation.MultiStateTransformation):
    first_state = transformation.PatternNode(sdfg.SDFGState)
    second_state = transformation.PatternNode(sdfg.SDFGState)

    @staticmethod
    def annotates_memlets():
        return False

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.first_state, cls.second_state)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        first_state: SDFGState = self.first_state
        second_state: SDFGState = self.second_state

        # Unique interstate edge
        if graph.out_degree(first_state) != 1:
            return False
        if graph.in_degree(second_state) != 1:
            return False

        out_edge = graph.out_edges(first_state)[0]
        if not out_edge.data.is_unconditional() or out_edge.data.assignments:
            return False

        return True

    def apply(self, _, sdfg):
        first_state: SDFGState = self.first_state
        second_state: SDFGState = self.second_state

        # Remove interstate edge(s)
        edges = sdfg.edges_between(first_state, second_state)
        for edge in edges:
            sdfg.remove_edge(edge)

        # Find dependencies
        # Overapproximation for now
        cc_first = list(nx.weakly_connected_components(first_state._nx))
        cc_second = list(nx.weakly_connected_components(second_state._nx))

        dependencies = []
        for cc in cc_first:
            arrays_first = set(
                [node.data for node in cc if isinstance(node, nodes.AccessNode)]
            )
            for cc_ in cc_second:
                arrays_second = set(
                    [node.data for node in cc_ if isinstance(node, nodes.AccessNode)]
                )
                if arrays_first.intersection(arrays_second):
                    source_sink_pairs = (
                        [node for node in cc if first_state.out_degree(node) == 0],
                        [node for node in cc_ if second_state.in_degree(node) == 0],
                    )
                    dependencies.append(source_sink_pairs)

        # Merge second state to first state
        for node in second_state.nodes():
            if isinstance(node, nodes.NestedSDFG):
                node.sdfg.parent = first_state

            if node not in first_state.nodes():
                first_state.add_node(node)

        for src, src_conn, dst, dst_conn, data in second_state.edges():
            first_state.add_edge(src, src_conn, dst, dst_conn, data)

        # Add happens-before memlets
        for sinks, sources in dependencies:
            for sink in sinks:
                for source in sources:
                    first_state.add_nedge(sink, source, memlet.Memlet())

        # Redirect edges and remove second state
        sdutil.change_edge_src(sdfg, second_state, first_state)
        sdfg.remove_node(second_state)
        if sdfg.start_state == second_state:
            sdfg.start_state = sdfg.node_id(first_state)
