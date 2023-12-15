# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace

from dace.libraries.blas import Gemm

from daisytuner.transformations.blas import GEMM

N, M, K = [dace.symbol(k) for k in "NMK"]


@dace.program
def matmul(A: dace.float64[N, K], B: dace.float64[K, M], C: dace.float64[N, M]):
    for i, j, k in dace.map[0:N, 0:M, 0:K]:
        with dace.tasklet:
            a << A[i, k]
            b << B[k, j]
            c >> C(1, lambda a, b: a + b)[i, j]

            c = a * b


def test_matmul():
    sdfg = matmul.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations(GEMM, validate=True)
    assert applied == 1
    assert len(sdfg.states()) == 1

    state = sdfg.start_state
    assert len(state.nodes()) == 4
    assert len(state.data_nodes()) == 3

    comps = [
        node for node in state.nodes() if not isinstance(node, dace.nodes.AccessNode)
    ]
    assert len(comps) == 1
    assert isinstance(comps[0], Gemm)
