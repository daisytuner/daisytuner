# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import copy
import dace
import math
import numpy as np

from daisytuner.analysis.performance_modeling import DeepBoatHullModel
from daisytuner.analysis.similarity import MapNestModel


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
#     cpu_model = MapNestModel.create(device=dace.DeviceType.CPU)
#     gpu_model = MapNestModel.create(device=dace.DeviceType.GPU)
#     model = DeepBoatHullModel(
#         cpu_model=cpu_model,
#         gpu_model=gpu_model,
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
#     # sdfg = copy.deepcopy(sdfg)
#     # args = {
#     #     "A": np.random.rand(1024, 1024).astype(np.float64),
#     #     "B": np.random.rand(1024, 1024).astype(np.float64),
#     # }
#     # sdfg.instrument = dace.InstrumentationType.Timer
#     # runtime = measure(sdfg, args, measurements=1)[0]
#     # runtime = runtime * 1e-3
#     runtime = 0.003298

#     # Modeled runtime
#     cpu_model = MapNestModel.create(device=dace.DeviceType.CPU)
#     gpu_model = MapNestModel.create(device=dace.DeviceType.GPU)
#     model = DeepBoatHullModel(
#         cpu_model=cpu_model,
#         gpu_model=gpu_model,
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
#     cpu_model = MapNestModel.create(device=dace.DeviceType.CPU)
#     gpu_model = MapNestModel.create(device=dace.DeviceType.GPU)
#     model = DeepBoatHullModel(
#         cpu_model=cpu_model,
#         gpu_model=gpu_model,
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
#     cpu_model = MapNestModel.create(device=dace.DeviceType.CPU)
#     gpu_model = MapNestModel.create(device=dace.DeviceType.GPU)
#     model = DeepBoatHullModel(
#         cpu_model=cpu_model,
#         gpu_model=gpu_model,
#         interconnect_bandwidth=3320,
#     )
#     estimated_runtime = model.compute(sdfg)
#     # assert math.log10(runtime) == math.log10(estimated_runtime)
