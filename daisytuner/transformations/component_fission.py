# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import networkx as nx

from dace.sdfg import utils as sdutil
from dace.sdfg.state import StateSubgraphView
from dace.transformation import transformation, helpers
from dace.properties import make_properties


@make_properties
class ComponentFission(transformation.MultiStateTransformation):
    """ """

    state = transformation.PatternNode(dace.SDFGState)

    @staticmethod
    def annotates_memlets():
        return False

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.state)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        ccs = list(nx.weakly_connected_components(self.state._nx))
        return len(ccs) > 1

    def apply(self, _, sdfg):
        state = self.state

        cc = list(nx.weakly_connected_components(self.state._nx))[0]
        subgraph = StateSubgraphView(state, cc)
        helpers.state_fission(sdfg, subgraph=subgraph)
