# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace

from daisytuner.transformations import TaskletSimplification


def test_outer_brackets():
    @dace.program
    def outer_brackets(A: dace.float64[32], B: dace.float64[32]):
        for i in dace.map[0:32]:
            with dace.tasklet(dace.Language.CPP):
                a << A[i]
                b >> B[i]
                """
                b = (a * 2);
                """

    sdfg = outer_brackets.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations_repeated(TaskletSimplification)
    assert applied == 1


def test_function_call():
    @dace.program
    def function_call(A: dace.float64[32], B: dace.float64[32]):
        for i in dace.map[0:32]:
            with dace.tasklet(dace.Language.CPP):
                a << A[i]
                b >> B[i]
                """
                b = sin(a * 2);
                """

    sdfg = function_call.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations_repeated(TaskletSimplification)
    assert applied == 0


def test_non_cpp():
    @dace.program
    def non_cpp(A: dace.float64[32], B: dace.float64[32]):
        for i in dace.map[0:32]:
            with dace.tasklet:
                a << A[i]
                b >> B[i]
                b = a * 2

    sdfg = non_cpp.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations_repeated(TaskletSimplification)
    assert applied == 0
