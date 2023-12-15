# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace

from daisytuner.transformations import GreedyTaskletFusion


def test_chain():
    @dace.program
    def sdfg_chain(A: dace.float64[32], B: dace.float64[32]):
        for i in dace.map[0:32]:
            B[i] = A[i] * 2 + 1

    sdfg = sdfg_chain.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations_repeated(GreedyTaskletFusion)
    assert applied == 2


def test_v():
    @dace.program
    def sdfg_v(A: dace.float64[32], B: dace.float64[32], C: dace.float64[32]):
        for i in dace.map[0:32]:
            C[i] = 2 * A[i] + 2 * B[i]

    sdfg = sdfg_v.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations_repeated(GreedyTaskletFusion)
    assert applied == 3


def test_common_result():
    @dace.program
    def sdfg_common_result(
        A: dace.float64[32], B: dace.float64[32], C: dace.float64[32]
    ):
        for i in dace.map[0:32]:
            tmp = A[i] + 1
            B[i] = tmp
            C[i] = tmp

    sdfg = sdfg_common_result.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations_repeated(GreedyTaskletFusion)
    assert applied == 0
