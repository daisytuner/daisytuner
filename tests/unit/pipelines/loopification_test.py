# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace

from dace.transformation.optimizer import Optimizer
from dace.transformation.interstate.loop_unroll import LoopUnroll

from daisytuner.pipelines import Loopification


def test_two_dimensional_maps():
    M = dace.symbol("M")
    N = dace.symbol("N")

    @dace.program
    def sdfg_test_two_dimensional_maps(
        A: dace.float32[M, N], B: dace.float32[M, N], C: dace.float32[M, N]
    ):
        C = A + B

    sdfg = sdfg_test_two_dimensional_maps.to_sdfg()
    sdfg.specialize({"M": 32, "N": 16})

    results = {}
    pipeline = Loopification()
    pipeline.apply_pass(sdfg, results)

    detected_loops = list(Optimizer(sdfg=sdfg).get_pattern_matches(patterns=LoopUnroll))
    assert len(detected_loops) == 2
