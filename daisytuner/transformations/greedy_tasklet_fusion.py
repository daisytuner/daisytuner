# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace

from dace.transformation.dataflow import TaskletFusion


class GreedyTaskletFusion(TaskletFusion):

    OPS_BLACKLIST = ["pow", "exp", "sqrt", "sin", "cos", "tanh"]

    def can_be_applied(
        self,
        graph: dace.SDFGState,
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        if not super().can_be_applied(graph, expr_index, sdfg, permissive):
            return False

        if expr_index == 0 and graph.out_degree(self.data) > 1:
            return False

        if expr_index == 1 and graph.out_degree(self.t1) > 1:
            return False

        for op in GreedyTaskletFusion.OPS_BLACKLIST:
            if op in self.t1.code.as_string:
                return False
            if op in self.t2.code.as_string:
                return False

        return True
