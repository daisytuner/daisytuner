import copy
import dace
import numpy as np

from daisytuner.optimizer.tiramisu.tiramisu_tuner import TiramisuTuner


def test_vecadd():
    K = dace.symbol("K")

    @dace.program
    def sdfg_vecadd(A: dace.float64[K], B: dace.float64[K], C: dace.float64[K]):
        for k in dace.map[0:K]:
            with dace.tasklet:
                a << A[k]
                b << B[k]
                c >> C[k]

                c = a + b

    sdfg = sdfg_vecadd.to_sdfg()
    sdfg.specialize({"K": 1024})
    sdfg.simplify()

    A = np.random.random((1024,)).astype(np.float64)
    B = np.random.random((1024,)).astype(np.float64)
    C = np.zeros((1024,), dtype=np.float64)
    C_opt = np.zeros((1024,), dtype=np.float64)
    args = {"A": A, "B": B, "C": C}

    tuner = TiramisuTuner()
    assert tuner.can_be_tuned(sdfg)
    sdfg_opt, info = tuner.tune(sdfg, arguments=copy.deepcopy(args))
    sdfg_opt.validate()

    sdfg(A=A, B=B, C=C)
    sdfg_opt(A=A, B=B, C=C_opt)
    assert np.allclose(C, C_opt)


def test_mxv():
    N = dace.symbol("N")
    K = dace.symbol("K")

    @dace.program
    def sdfg_mxv(A: dace.float64[N, K], B: dace.float64[K], C: dace.float64[N]):
        for i, k in dace.map[0:N, 0:K]:
            with dace.tasklet:
                a << A[i, k]
                b << B[k]
                c >> C(1, lambda e, f: e + f)[i]

                c = a * b

    sdfg = sdfg_mxv.to_sdfg()
    sdfg.specialize({"N": 1024, "K": 128})
    sdfg.simplify()

    A = np.random.random((1024, 128)).astype(np.float64)
    B = np.random.random((128,)).astype(np.float64)
    C = np.zeros((1024,), dtype=np.float64)
    C_opt = np.zeros((1024,), dtype=np.float64)
    args = {"A": A, "B": B, "C": C}

    tuner = TiramisuTuner()
    assert tuner.can_be_tuned(sdfg)

    sdfg_opt, info = tuner.tune(sdfg, arguments=copy.deepcopy(args))
    sdfg_opt.validate()
    assert info["score"] < -10

    sdfg(A=A, B=B, C=C)
    sdfg_opt(A=A, B=B, C=C_opt)
    assert np.allclose(C, C_opt)


def test_matmul():
    @dace.program
    def matmul(
        A: dace.float64[1024, 1024],
        B: dace.float64[1024, 1024],
        C: dace.float64[1024, 1024],
    ):
        for i, j, k in dace.map[0:1024, 0:1024, 0:1024]:
            with dace.tasklet:
                a << A[i, k]
                b << B[k, j]
                c >> C(1, lambda a, b: a + b)[i, j]

                c = a * b

    sdfg = matmul.to_sdfg()
    sdfg.simplify()

    A = np.random.random((1024, 1024)).astype(np.float64)
    B = np.random.random((1024, 1024)).astype(np.float64)
    C = np.zeros((1024, 1024), dtype=np.float64)
    C_opt = np.zeros((1024, 1024), dtype=np.float64)
    args = {"A": A, "B": B, "C": C}

    tuner = TiramisuTuner()
    assert tuner.can_be_tuned(sdfg)

    sdfg_opt, info = tuner.tune(sdfg, arguments=copy.deepcopy(args))
    sdfg_opt.validate()
    assert info["score"] < -20

    sdfg(A=A, B=B, C=C)
    sdfg_opt(A=A, B=B, C=C_opt)
    assert np.allclose(C, C_opt)
