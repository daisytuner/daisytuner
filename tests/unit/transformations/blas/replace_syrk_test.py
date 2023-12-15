# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace

from daisytuner.library.blas import Syrk

from daisytuner.transformations.blas import SYRK

N, K = [dace.symbol(k) for k in "NK"]


@dace.program
def syrk(A: dace.float64[N, K], C: dace.float64[N, N]):
    for i, k in dace.map[0:N, 0:K]:
        for j in dace.map[0 : i + 1]:
            with dace.tasklet:
                a << A[i, k]
                b << A[j, k]
                c >> C(1, lambda a, b: a + b)[i, j]

                c = a * b


def test_syrk():
    sdfg = syrk.to_sdfg()
    sdfg.simplify()
    sdfg.specialize({"N": 32, "K": 10})

    applied = sdfg.apply_transformations(SYRK, validate=True)
    assert applied == 1
    assert len(sdfg.states()) == 1

    state = sdfg.start_state
    assert len(state.nodes()) == 3
    assert len(state.data_nodes()) == 2

    comps = [
        node for node in state.nodes() if not isinstance(node, dace.nodes.AccessNode)
    ]
    assert len(comps) == 1
    assert isinstance(comps[0], Syrk)
