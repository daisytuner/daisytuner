# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace

from dace.libraries.blas import Gemv

from daisytuner.transformations.blas import GEMV

N, M = [dace.symbol(k) for k in "NM"]


@dace.program
def mxv(A: dace.float64[N, M], B: dace.float64[M], C: dace.float64[N]):
    for i, j in dace.map[0:N, 0:M]:
        with dace.tasklet:
            a << A[i, j]
            b << B[j]
            c >> C(1, lambda a, b: a + b)[i]

            c = a * b


@dace.program
def mxv_flip(B: dace.float64[M], A: dace.float64[N, M], C: dace.float64[N]):
    for i, j in dace.map[0:N, 0:M]:
        with dace.tasklet:
            b << B[j]
            a << A[i, j]
            c >> C(1, lambda a, b: a + b)[i]

            c = a * b


def test_mxv():
    sdfg = mxv.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations(GEMV, validate=True)
    assert applied == 1
    assert len(sdfg.states()) == 1

    state = sdfg.start_state
    assert len(state.nodes()) == 4
    assert len(state.data_nodes()) == 3

    comps = [
        node for node in state.nodes() if not isinstance(node, dace.nodes.AccessNode)
    ]
    assert len(comps) == 1
    assert isinstance(comps[0], Gemv)


def test_mxv_flip():
    sdfg = mxv_flip.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations(GEMV, validate=True)
    assert applied == 1
    assert len(sdfg.states()) == 1

    state = sdfg.start_state
    assert len(state.nodes()) == 4
    assert len(state.data_nodes()) == 3

    comps = [
        node for node in state.nodes() if not isinstance(node, dace.nodes.AccessNode)
    ]
    assert len(comps) == 1
    assert isinstance(comps[0], Gemv)
