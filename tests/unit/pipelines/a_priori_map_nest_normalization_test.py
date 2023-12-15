# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace

from daisytuner.pipelines import APrioriMapNestNormalization


def test_independent_memory_accesses():
    @dace.program
    def independent_memory_accesses(
        A: dace.float64[32, 32], B: dace.float64[32, 32], C: dace.float64[32, 32]
    ):
        for i, j in dace.map[0:32, 0:32]:
            a = A[i, j]
            B[i, j] = 2 * a
            C[j, i] = a + 1

    sdfg = independent_memory_accesses.to_sdfg()
    sdfg.simplify()
    assert not APrioriMapNestNormalization.is_normalized(sdfg)

    results = {}
    pipeline = APrioriMapNestNormalization()
    pipeline.apply_pass(sdfg, results)


def test_dependent_memory_accesses():
    @dace.program
    def dependent_memory_accesses(A: dace.float64[64], B: dace.float64[64]):
        for i in dace.map[0:32:2]:
            a = A[i]
            b = A[i + 1]
            B[i] = a + 1
            B[i + 1] = b + 1

    sdfg = dependent_memory_accesses.to_sdfg()
    sdfg.simplify()
    assert APrioriMapNestNormalization.is_normalized(sdfg)

    results = {}
    pipeline = APrioriMapNestNormalization()
    pipeline.apply_pass(sdfg, results)


def test_stride_minimization():
    N, M, K = [dace.symbol(k) for k in "NMK"]
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
    sdfg.simplify()
    assert not APrioriMapNestNormalization.is_normalized(sdfg)

    results = {}
    pipeline = APrioriMapNestNormalization()
    pipeline.apply_pass(sdfg, results)
