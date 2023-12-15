# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import platform
import dace
import pytest
import numpy as np

from scipy.linalg.blas import ssyrk, dsyrk, csyrk, zsyrk

from daisytuner.library.blas import Syrk

N = dace.symbol("N")
K = dace.symbol("K")


@pytest.mark.skipif(
    platform.system() == "Windows", reason="Test cannot be run on Windows"
)
@pytest.mark.parametrize(
    "dtype, n, k, alpha, beta, lower, trans",
    [
        pytest.param(dace.float64, 32, 16, 1.0, 0.0, 0, 0),
        pytest.param(dace.float64, 32, 16, 1.0, 0.0, 0, 1),
        pytest.param(dace.float64, 32, 16, 1.0, 0.0, 0, 2),
        pytest.param(dace.float64, 32, 16, 1.0, 0.0, 1, 0),
        pytest.param(dace.float64, 32, 16, 1.0, 0.0, 1, 1),
        pytest.param(dace.float64, 32, 16, 1.0, 0.0, 1, 2),
        pytest.param(dace.float64, 32, 16, 2.0, 1.0, 0, 0),
        pytest.param(dace.float64, 32, 64, 2.0, 1.0, 0, 0),
        pytest.param(dace.float32, 32, 16, 1.0, 0.0, 0, 0),
        pytest.param(dace.float32, 32, 16, 1.0, 0.0, 0, 1),
        pytest.param(dace.float32, 32, 16, 1.0, 0.0, 0, 2),
        pytest.param(dace.float32, 32, 16, 1.0, 0.0, 1, 0),
        pytest.param(dace.float32, 32, 16, 1.0, 0.0, 1, 1),
        pytest.param(dace.float32, 32, 16, 1.0, 0.0, 1, 2),
        pytest.param(dace.float32, 32, 16, 2.0, 1.0, 0, 0),
        pytest.param(dace.float32, 32, 64, 2.0, 1.0, 0, 0),
        pytest.param(dace.complex64, 32, 16, 2.0, 1.0, 0, 0),
        pytest.param(dace.complex128, 32, 64, 2.0, 1.0, 0, 0),
    ],
)
def test_syrk(dtype, n, k, alpha, beta, lower, trans):
    sdfg = dace.SDFG("syrk")
    state = sdfg.add_state()

    C, C_arr = sdfg.add_array("C", [N, N], dtype)
    if trans == 0:
        A, A_arr = sdfg.add_array("A", [N, K], dtype)
    else:
        A, A_arr = sdfg.add_array("A", [K, N], dtype)

    rA = state.add_read("A")
    wC = state.add_write("C")

    libnode = Syrk(
        "_syrk_",
        trans=("N", "T", "C")[trans],
        uplo=("U", "L")[lower],
        alpha=alpha,
        beta=beta,
    )
    libnode.implementation = "MKL"
    state.add_node(libnode)

    state.add_edge(rA, None, libnode, "_a", dace.Memlet.from_array(A, A_arr))
    state.add_edge(libnode, "_c", wC, None, dace.Memlet.from_array(C, C_arr))
    if beta != 0:
        rC = state.add_read("C")
        state.add_edge(rC, None, libnode, "_cin", dace.Memlet.from_array(C, C_arr))

    sdfg.expand_library_nodes()
    sdfg.specialize({"N": n, "K": k})

    c = np.random.random((n, n)).astype(dtype.as_numpy_dtype())
    if trans == 0:
        a = np.random.random((n, k)).astype(dtype.as_numpy_dtype())
    else:
        a = np.random.random((k, n)).astype(dtype.as_numpy_dtype())

    if dtype == dace.float32:
        ref = ssyrk(alpha, a, beta=beta, c=c, lower=lower, trans=trans)
    elif dtype == dace.float64:
        ref = dsyrk(alpha, a, beta=beta, c=c, lower=lower, trans=trans)
    elif dtype == dace.complex64:
        ref = csyrk(alpha, a, beta=beta, c=c, lower=lower, trans=trans)
    elif dtype == dace.complex128:
        ref = zsyrk(alpha, a, beta=beta, c=c, lower=lower, trans=trans)

    sdfg(A=a, C=c)
    assert np.allclose(c, ref)
