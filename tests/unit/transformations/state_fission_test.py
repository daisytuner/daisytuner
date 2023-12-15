# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace

from daisytuner.transformations import StateFission


def test_fission_two_maps():
    @dace.program
    def sdfg_fission_two_maps(A: dace.float64[32, 32], B: dace.float64[32]):
        for i in dace.map[0:32]:
            with dace.tasklet:
                a << A[i]
                b >> A[i]
                b = a + 1

        for i in dace.map[0:32]:
            with dace.tasklet:
                a << A[i]
                b >> B[i]

                b = 2 * a

    sdfg = sdfg_fission_two_maps.to_sdfg()
    assert len(sdfg.states()) == 1

    applied = sdfg.apply_transformations_repeated(StateFission)
    assert applied == 1
    assert len(sdfg.states()) == 2


def test_fission_tree():
    @dace.program
    def sdfg_fission_tree(
        A: dace.float64[32, 32], B: dace.float64[32], C: dace.float64[32]
    ):
        for i in dace.map[0:32]:
            with dace.tasklet:
                a << A[i]
                b >> A[i]
                b = a + 1

        for i in dace.map[0:32]:
            with dace.tasklet:
                a << A[i]
                b >> B[i]

                b = 2 * a

        for i in dace.map[0:32]:
            with dace.tasklet:
                a << A[i]
                b >> C[i]

                b = 2 * a

    sdfg = sdfg_fission_tree.to_sdfg()
    assert len(sdfg.states()) == 1

    applied = sdfg.apply_transformations_repeated(StateFission)
    assert applied == 1
    assert len(sdfg.states()) == 2


def test_fission_v():
    @dace.program
    def sdfg_fission_v(
        A: dace.float64[32, 32], B: dace.float64[32], C: dace.float64[32]
    ):
        for i in dace.map[0:32]:
            with dace.tasklet:
                a << A[i]
                b >> A[i]
                b = a + 1

        for i in dace.map[0:32]:
            with dace.tasklet:
                a << B[i]
                b >> B[i]

                b = 2 * a

        for i in dace.map[0:32]:
            with dace.tasklet:
                a << A[i]
                b << B[i]
                c >> C[i]

                b = a + b

    sdfg = sdfg_fission_v.to_sdfg()
    assert len(sdfg.states()) == 1

    applied = sdfg.apply_transformations_repeated(StateFission)
    assert applied == 1
    assert len(sdfg.states()) == 2
