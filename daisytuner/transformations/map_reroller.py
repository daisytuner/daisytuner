# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace

from dace.sdfg import utils as sdutil
from dace.sdfg.state import StateSubgraphView
from dace.transformation import transformation
from dace.properties import make_properties


@make_properties
class MapReroller(transformation.SingleStateTransformation):
    map_entry = transformation.PatternNode(dace.nodes.MapEntry)
    tasklet = transformation.PatternNode(dace.nodes.Tasklet)
    map_exit = transformation.PatternNode(dace.nodes.MapExit)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry, cls.tasklet, cls.map_exit)]

    def can_be_applied(
        self, state: dace.SDFGState, expr_index: int, sdfg: dace.SDFG, permissive=False
    ):
        # Scope subgraph has exactly two tasklets
        subgraph_nodes = set(state.all_nodes_between(self.map_entry, self.map_exit))
        if len(subgraph_nodes) != 2:
            return False

        tasklets = [
            node for node in subgraph_nodes if isinstance(node, dace.nodes.Tasklet)
        ]
        if len(tasklets) != 2:
            return False

        tasklet2 = tasklets[1] if tasklets[0] == self.tasklet else tasklets[0]

        # Both tasklets execute the same code
        # if self.tasklet.code.as_string != tasklet2.code.as_string:
        #     return False
        if (
            self.tasklet.in_connectors != tasklet2.in_connectors
            or self.tasklet.out_connectors != tasklet2.out_connectors
        ):
            return False

        # Write memlets are of structure: 2i, 2i + 1 in last dim
        if (
            not state.in_degree(self.map_exit) == 2
            or state.out_degree(self.tasklet) != 1
        ):
            return False

        edge1 = state.out_edges(self.tasklet)[0]
        ranges1 = edge1.data.subset

        edge2 = state.out_edges(tasklet2)[0]
        ranges2 = edge2.data.subset

        if len(ranges1) != len(ranges2):
            return False

        last_dim = self.map_entry.map.params[-1]
        for i in range(len(ranges1)):
            range1 = ranges1[i]
            range2 = ranges2[i]
            if i < len(ranges1) - 1:
                if range1 != range2:
                    return False
            else:  # Last dim
                b1, e1, s1 = range1
                b2, e2, s2 = range2
                if b1 != b2 - 1:
                    return False
                if e1 != e2 - 1:
                    return False
                if s1 != s2:
                    return False

                b_exp = 2 * dace.symbolic.pystr_to_symbolic(last_dim)
                if b_exp != b1:
                    return False

        return True

    def apply(self, state: dace.SDFGState, sdfg: dace.SDFG):
        scope_subgraph = state.scope_subgraph(
            self.map_entry, include_entry=True, include_exit=True
        )

        # Update range
        b, e, s = self.map_entry.map.range[-1]
        self.map_entry.map.range[-1] = (b, 2 * e + 1, s)

        # Update iterator
        iter = self.map_entry.map.params[-1]
        scope_subgraph.replace(iter, f"({iter} / 2)")

        tasklet2 = None
        for node in state.all_nodes_between(self.map_entry, self.map_exit):
            if node != self.tasklet:
                tasklet2 = node
                break
        state.remove_node(tasklet2)
