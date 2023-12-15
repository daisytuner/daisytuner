# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import networkx as nx

from dace.sdfg import SDFG
from dace.sdfg import utils as sdutil
from dace.sdfg.state import SDFGState, StateSubgraphView
from dace.transformation import transformation, helpers
from dace.properties import make_properties, Property


@make_properties
class StateFission(transformation.SingleStateTransformation):
    access_node = transformation.PatternNode(dace.nodes.AccessNode)

    allow_transients = Property(dtype=bool, default=False)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.access_node)]

    def annotates_memlets(self) -> bool:
        return True

    def can_be_applied(
        self,
        state: dace.SDFGState,
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ):
        if not self.allow_transients and sdfg.arrays[self.access_node.data].transient:
            return False

        if (
            state.in_degree(self.access_node) == 0
            or state.out_degree(self.access_node) == 0
        ):
            return False

        if state.entry_node(self.access_node) is not None:
            return False

        return True

    def apply(self, state: SDFGState, sdfg: SDFG):
        access_node = self.access_node
        helpers.state_fission_after(sdfg, state, access_node)
