import dace

from daisytuner.analysis.similarity import MapNest
from daisytuner.tuning.tiramisu.tiramisu_tuner import TiramisuTuner


# def test_matmul():
#     @dace.program
#     def matmul(
#         A: dace.float64[1024, 1024],
#         B: dace.float64[1024, 1024],
#         C: dace.float64[1024, 1024],
#     ):
#         for i, j, k in dace.map[0:1024, 0:1024, 0:1024]:
#             with dace.tasklet:
#                 a << A[i, k]
#                 b << B[k, j]
#                 c >> C(1, lambda a, b: a + b)[i, j]

#                 c = a * b

#     sdfg = matmul.to_sdfg()
#     sdfg.simplify()

#     map_entry = None
#     for node in sdfg.start_state.nodes():
#         if not isinstance(node, dace.nodes.MapEntry):
#             continue

#         map_entry = node
#         break

#     loop_nest = MapNest.create(sdfg, sdfg.start_state, map_entry)

#     tuner = TiramisuTuner(beam_size=5, max_depth=3)
#     schedule, score = tuner.tune(loop_nest)
#     assert schedule
#     assert score < -20
