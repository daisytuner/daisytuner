# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import numpy as np

from dace.sdfg.nodes import MapEntry
from dace.transformation import pass_pipeline as ppl

from daisytuner.passes import StrideMinimization

BSize = dace.symbol("BSize")
N, M, K = [dace.symbol(k) for k in "NMK"]


def test_wcr_matmul_stride_minimization():

    N = 32
    M = 32
    K = 128

    @dace.program
    def matmul_collapsed(
        A: dace.float64[N, K],
        B: dace.float64[K, M],
        C: dace.float64[N, M],
    ):
        for i, j, k in dace.map[0:N, 0:M, 0:K]:
            with dace.tasklet:
                a << A[i, k]
                b << B[k, j]
                c >> C(1, lambda a, b: a + b)[i, j]

                c = a * b

    sdfg = matmul_collapsed.to_sdfg()
    assert not StrideMinimization.is_normalized(sdfg)

    pipeline = ppl.FixedPointPipeline([StrideMinimization()])
    res = {}
    pipeline.apply_pass(sdfg, res)

    map_entry = None
    for n in sdfg.nodes()[0].nodes():
        if isinstance(n, MapEntry):
            map_entry = n
            break
    assert map_entry is not None
    assert res[StrideMinimization.__name__][map_entry][0] == ["i", "k", "j"]
    assert res[StrideMinimization.__name__][map_entry][1] == ["i", "j", "k"]
    assert StrideMinimization.is_normalized(sdfg)


def test_split_batched_matmul_stride_minimization():

    BSize = 8
    N = 32
    M = 32
    K = 128

    @dace.program
    def batched_matmul_split(
        A: dace.float64[BSize, N, K],
        B: dace.float64[BSize, K, M],
        C: dace.float64[BSize, N, M],
    ):
        for i, b in dace.map[0:N, 0:BSize]:
            for j, k in dace.map[0:M, 0:K]:
                with dace.tasklet:
                    a << A[b, i, k]
                    b << B[b, k, j]
                    c >> C(1, lambda a, b: a + b)[b, i, j]

                    c = a * b

    sdfg = batched_matmul_split.to_sdfg()
    assert not StrideMinimization.is_normalized(sdfg)

    pipeline = ppl.FixedPointPipeline([StrideMinimization()])
    res = {}
    pipeline.apply_pass(sdfg, res)

    map_entry_o = None
    map_entry_i = None
    for n in sdfg.nodes()[0].nodes():
        if isinstance(n, MapEntry):
            if "i" in n.map.params and "b" in n.map.params:
                map_entry_o = n
            elif "j" in n.map.params and "k" in n.map.params:
                map_entry_i = n
    assert map_entry_o is not None
    assert map_entry_i is not None
    assert res[StrideMinimization.__name__][map_entry_o][0] == ["b", "i"]
    assert res[StrideMinimization.__name__][map_entry_o][1] == ["i", "b"]
    assert res[StrideMinimization.__name__][map_entry_i][0] == ["k", "j"]
    assert res[StrideMinimization.__name__][map_entry_i][1] == ["j", "k"]
    assert StrideMinimization.is_normalized(sdfg)
