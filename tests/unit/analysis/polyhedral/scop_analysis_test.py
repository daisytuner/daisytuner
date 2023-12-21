import dace
import re

from dace.codegen import control_flow as cflow


def test_loop():
    N = dace.symbol("N")

    @dace.program
    def sdfg_loop(A: dace.float32[N], B: dace.float32[N]):
        for i in range(N):
            with dace.tasklet:
                a << A[i]
                b >> B[i]

                b = a

    sdfg = sdfg_loop.to_sdfg()

    # Scop analysis
    from daisytuner.analysis.polyhedral import ScopAnalysis

    analysis = ScopAnalysis(sdfg=sdfg)
    cft = cflow.structured_control_flow_tree(sdfg, lambda _: "")
    scop = analysis._traverse(cft, [], [], {})

    assert bool(
        re.fullmatch("\[N\] -> { (.*)\[i\] : 0 <= i < N }", scop.domain.to_str())
    )
    assert bool(re.fullmatch("\[N\] -> { (.*)\[i\] -> A\[i\] }", scop.read.to_str()))
    assert bool(re.fullmatch("\[N\] -> { (.*)\[i\] -> B\[i\] }", scop.write.to_str()))


def test_single_dimension():
    N = dace.symbol("N")

    @dace.program
    def sdfg_single_dimension(A: dace.float32[N], B: dace.float32[N]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> B[i]

                b = a

    sdfg = sdfg_single_dimension.to_sdfg()

    # Scop analysis
    from daisytuner.analysis.polyhedral import ScopAnalysis

    scop, _ = ScopAnalysis.create(sdfg)

    assert bool(
        re.fullmatch("\[N\] -> { (.*)\[i\] : 0 <= i < N }", scop.domain.to_str())
    )
    assert bool(re.fullmatch("\[N\] -> { (.*)\[i\] -> A\[i\] }", scop.read.to_str()))
    assert bool(re.fullmatch("\[N\] -> { (.*)\[i\] -> B\[i\] }", scop.write.to_str()))


# def test_multi_dimension():
#     N = dace.symbol("N")
#     M = dace.symbol("M")

#     @dace.program
#     def sdfg_multi_dimension(A: dace.float32[N, M], B: dace.float32[M, N]):
#         for i, j in dace.map[0:N, 0:M]:
#             with dace.tasklet:
#                 a << A[i, j]
#                 b >> B[j, i]

#                 b = a

#     sdfg = sdfg_multi_dimension.to_sdfg()

#     # Scop analysis
#     from daisytuner.analysis.polyhedral import ScopAnalysis
#     scop, _ = ScopAnalysis.create(sdfg)

#     assert bool(
#         re.fullmatch(
#             "\[N, M\] -> { (.*)\[i, j\] : 0 <= i < N and 0 <= j < M }",
#             scop.domain.to_str(),
#         )
#     )
#     assert bool(
#         re.fullmatch("\[N, M\] -> { (.*)\[i, j\] -> A\[i, j\] }", scop.read.to_str())
#     )
#     assert bool(
#         re.fullmatch("\[N, M\] -> { (.*)\[i, j\] -> B\[j, i\] }", scop.write.to_str())
#     )
