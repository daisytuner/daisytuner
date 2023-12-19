import copy
import dace
import numpy as np

from daisytuner.normalization import MapExpandedForm
from daisytuner.optimizer import PlutoTuner


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

    pipeline = MapExpandedForm()
    pipeline.apply_pass(sdfg, {})

    tuner = PlutoTuner()
    assert tuner.can_be_tuned(sdfg)
    sdfg_opt, _ = tuner.tune(sdfg, arguments=copy.deepcopy(args))
    sdfg_opt.validate()

    sdfg(A=A, B=B, C=C)
    sdfg_opt(A=A, B=B, C=C_opt)
    assert np.allclose(C, C_opt)


# def test_mxv():
#     N = dace.symbol("N")
#     K = dace.symbol("K")

#     @dace.program
#     def sdfg_mxv(A: dace.float64[N, K], B: dace.float64[K], C: dace.float64[N]):
#         for i, k in dace.map[0:N, 0:K]:
#             with dace.tasklet:
#                 a << A[i, k]
#                 b << B[k]
#                 c >> C(1, lambda e, f: e + f)[i]

#                 c = a * b

#     sdfg = sdfg_mxv.to_sdfg()
#     sdfg.specialize({"N": 4, "K": 2})
#     sdfg.simplify()

#     pipeline = MapExpandedForm()
#     pipeline.apply_pass(sdfg, {})

#     # Tune SDFG
#     from daisytuner.tuning import PlutoTuner

#     tuner = PlutoTuner()
#     assert tuner.can_be_tuned(sdfg)
#     sdfg_opt = tuner.tune(sdfg)
#     sdfg_opt.validate()

#     A = np.random.random((4, 2)).astype(np.float64)
#     B = np.random.random((2,)).astype(np.float64)
#     C = np.zeros((4,), dtype=np.float64)
#     C_opt = np.zeros((4,), dtype=np.float64)

#     sdfg(A=A, B=B, C=C)
#     sdfg_opt(A=A, B=B, C=C_opt)
#     assert np.allclose(C, C_opt)


# def test_mm():
#     N = dace.symbol("N")
#     M = dace.symbol("M")
#     K = dace.symbol("K")

#     @dace.program
#     def sdfg_mm(A: dace.float64[N, K], B: dace.float64[K, M], C: dace.float64[N, M]):
#         for i, j, k in dace.map[0:N, 0:M, 0:K]:
#             with dace.tasklet:
#                 a << A[i, j]
#                 b << B[k, j]
#                 c >> C(1, lambda e, f: e + f)[i, j]

#                 c = a * b

#     sdfg = sdfg_mm.to_sdfg()
#     sdfg.specialize({"M": 8, "N": 4, "K": 2})
#     sdfg.simplify()

#     pipeline = MapExpandedForm()
#     pipeline.apply_pass(sdfg, {})

#     # Tune SDFG
#     from daisytuner.tuning import PlutoTuner
#     tuner = PlutoTuner()
#     assert tuner.can_be_tuned(sdfg)
#     sdfg_opt = tuner.tune(sdfg)
#     sdfg_opt.validate()

#     A = np.random.random((4, 2)).astype(np.float64)
#     B = np.random.random((2, 8)).astype(np.float64)
#     C = np.zeros((4, 8), dtype=np.float64)
#     C_opt = np.zeros((4, 8), dtype=np.float64)

#     sdfg(A=A, B=B, C=C)
#     sdfg_opt(A=A, B=B, C=C_opt)
#     assert np.allclose(C, C_opt)
