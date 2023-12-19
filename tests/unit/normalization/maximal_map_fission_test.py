# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace

from dace.transformation import pass_pipeline as ppl
from daisytuner.normalization import MaximalMapFission


def test_independent_writes():
    @dace.program
    def independent_writes(
        A: dace.float64[32, 32], B: dace.float64[32, 32], C: dace.float64[32, 32]
    ):
        for i, j in dace.map[0:32, 0:32]:
            a = A[i, j]
            B[i, j] = 2 * a
            C[j, i] = a + 1

    sdfg = independent_writes.to_sdfg()
    sdfg.simplify()
    assert not MaximalMapFission.is_normalized(sdfg)

    report = {}
    pipeline = ppl.FixedPointPipeline([MaximalMapFission()])
    pipeline.apply_pass(sdfg, report)
    assert report
    assert MaximalMapFission.is_normalized(sdfg)


def test_dependent_writes_same_array():
    @dace.program
    def dependent_writes_same_array(A: dace.float64[64], B: dace.float64[64]):
        for i in dace.map[0:32:2]:
            a = A[i]
            b = A[i + 1]
            B[i] = a + 1
            B[i + 1] = b + 1

    sdfg = dependent_writes_same_array.to_sdfg()
    sdfg.simplify()
    assert MaximalMapFission.is_normalized(sdfg)
