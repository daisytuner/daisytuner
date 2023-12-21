# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import copy
import dace
import math
import numpy as np

from daisytuner.analysis.performance_modeling import BoatHullModel
from daisytuner.profiling.measure import measure


# def test_copy_cpu():
#     """
#     Numbers are based on Lukas' Acer Notebook.
#     """

#     @dace.program
#     def sdfg_copy(A: dace.float64[1024, 1024], B: dace.float64[1024, 1024]):
#         for i, j in dace.map[0:1024, 0:1024]:
#             with dace.tasklet:
#                 a << A[i, j]
#                 b >> B[i, j]

#                 b = a

#     sdfg = sdfg_copy.to_sdfg()

#     # Reference runtime
#     # args = {
#     #     "A": np.random.rand(1024, 1024).astype(np.float64),
#     #     "B": np.random.rand(1024, 1024).astype(np.float64),
#     # }
#     # sdfg.instrument = dace.InstrumentationType.Timer
#     # runtime = measure(sdfg, args, measurements=1)[0]
#     # runtime = runtime * 1e-3
#     runtime = 0.001055

#     # Modeled runtime
#     model = BoatHullModel(
#         host_peakflops=71873,
#         device_peakflops=166450,
#         host_cores=12,
#         device_cores=3840,
#         host_memory_bandwidth=17455,
#         device_memory_bandwidth=336050,
#         interconnect_bandwidth=3320,
#     )
#     estimated_runtime = model.compute(sdfg)
#     # assert math.log10(runtime) == math.log10(estimated_runtime)


# def test_copy_gpu():
#     """
#     Numbers are based on Lukas' Acer Notebook.
#     """

#     @dace.program
#     def sdfg_copy(A: dace.float64[1024, 1024], B: dace.float64[1024, 1024]):
#         for i, j in dace.map[0:1024, 0:1024]:
#             with dace.tasklet:
#                 a << A[i, j]
#                 b >> B[i, j]

#                 b = a

#     sdfg = sdfg_copy.to_sdfg()
#     sdfg.apply_gpu_transformations()

#     # Reference runtime
#     # args = {
#     #     "A": np.random.rand(1024, 1024).astype(np.float64),
#     #     "B": np.random.rand(1024, 1024).astype(np.float64),
#     # }
#     # sdfg.instrument = dace.InstrumentationType.Timer
#     # runtime = measure(sdfg, args, measurements=1)[0]
#     # runtime = runtime * 1e-3
#     runtime = 0.003298

#     # Modeled runtime
#     model = BoatHullModel(
#         host_peakflops=71873,
#         device_peakflops=166450,
#         host_cores=12,
#         device_cores=3840,
#         host_memory_bandwidth=17455,
#         device_memory_bandwidth=336050,
#         interconnect_bandwidth=3320,
#     )
#     estimated_runtime = model.compute(sdfg)
#     # assert math.log10(runtime) == math.log10(estimated_runtime)


# def test_triad_cpu():
#     """
#     Numbers are based on Lukas' Acer Notebook.
#     """

#     @dace.program
#     def sdfg_triad(
#         A: dace.float64[32768],
#         B: dace.float64[32768],
#         C: dace.float64[32768],
#     ):
#         for i in dace.map[0:32768]:
#             with dace.tasklet:
#                 a >> A[i]
#                 b << B[i]
#                 c << C[i]

#                 a = b + c * 2.0

#     sdfg = sdfg_triad.to_sdfg()

#     # Reference runtime
#     # args = {
#     #     "A": np.random.rand(32768).astype(np.float64),
#     #     "B": np.random.rand(32768).astype(np.float64),
#     #     "C": np.random.rand(32768).astype(np.float64),
#     # }
#     # sdfg.instrument = dace.InstrumentationType.Timer
#     # runtime = measure(sdfg, args, measurements=1)[0]
#     # runtime = runtime * 1e-3
#     runtime = 0.000151

#     # Modeled runtime
#     model = BoatHullModel(
#         host_peakflops=71873,
#         device_peakflops=166450,
#         host_cores=12,
#         device_cores=3840,
#         host_memory_bandwidth=17455,
#         device_memory_bandwidth=336050,
#         interconnect_bandwidth=3320,
#     )
#     estimated_runtime = model.compute(sdfg)
#     # assert math.log10(runtime) == math.log10(estimated_runtime)


# def test_triad_gpu():
#     """
#     Numbers are based on Lukas' Acer Notebook.
#     """

#     @dace.program
#     def sdfg_triad(
#         A: dace.float64[32768],
#         B: dace.float64[32768],
#         C: dace.float64[32768],
#     ):
#         for i in dace.map[0:32768]:
#             with dace.tasklet:
#                 a >> A[i]
#                 b << B[i]
#                 c << C[i]

#                 a = b + c * 2.0

#     sdfg = sdfg_triad.to_sdfg()
#     sdfg.apply_gpu_transformations()

#     # Reference runtime
#     # args = {
#     #     "A": np.random.rand(32768).astype(np.float64),
#     #     "B": np.random.rand(32768).astype(np.float64),
#     #     "C": np.random.rand(32768).astype(np.float64),
#     # }
#     # sdfg.instrument = dace.InstrumentationType.Timer
#     # runtime = measure(sdfg, args, measurements=1)[0]
#     # runtime = runtime * 1e-3
#     runtime = 0.000347

#     # Modeled runtime
#     model = BoatHullModel(
#         host_peakflops=71873,
#         device_peakflops=166450,
#         host_cores=12,
#         device_cores=3840,
#         host_memory_bandwidth=17455,
#         device_memory_bandwidth=336050,
#         interconnect_bandwidth=3320,
#     )
#     estimated_runtime = model.compute(sdfg)
#     # assert math.log10(runtime) == math.log10(estimated_runtime)
