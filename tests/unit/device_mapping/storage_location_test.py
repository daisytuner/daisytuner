# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import numpy as np

from daisytuner.analysis.similarity.benchmarking import CPUBenchmark, GPUBenchmark

from daisytuner.device_mapping import Environment, Action
from daisytuner.device_mapping.state import StorageLocation


def test_storage_location_both():
    @dace.program
    def sdfg_storage_location_both(
        A: dace.float64[256, 256], B: dace.float64[256, 256], C: dace.float64[256, 256]
    ):
        for i, j in dace.map[0:256, 0:256]:
            with dace.tasklet:
                a << A[i, j]
                b >> B[i, j]

                b = 2 * a

        for k, l in dace.map[0:256, 0:256]:
            with dace.tasklet:
                a << A[k, l]
                c >> C[k, l]

                c = 2 * a

        for m, n in dace.map[0:256, 0:256]:
            with dace.tasklet:
                b << B[m, n]
                a >> A[m, n]

                a = 2 * b

    sdfg = sdfg_storage_location_both.to_sdfg()
    sdfg.simplify()

    maps = {}
    for state in sdfg.states():
        for node in state.nodes():
            if not isinstance(node, dace.nodes.MapEntry):
                continue

            maps[node.map.params[0]] = node

    host_benchmark = CPUBenchmark.from_cache("garbenheim")
    device_benchmark = GPUBenchmark.from_cache("garbenheim")
    env = Environment(
        sdfg=sdfg, cpu_benchmark=host_benchmark, gpu_benchmark=device_benchmark
    )
    current_state = env.state

    i = 0
    terminated = current_state.terminated
    while not terminated:
        if i == 0:
            action = Action.COPY_HOST_TO_DEVICE, "A"
            current_state, reward, terminated, truncated, info = env.step(action=action)
            _, active_gom = current_state.active()

            assert active_gom.array_table["A"] == StorageLocation.BOTH
            assert active_gom.array_table["B"] == StorageLocation.HOST
            assert active_gom.array_table["C"] == StorageLocation.HOST
        elif i == 1:
            action = Action.COPY_HOST_TO_DEVICE, "B"
            current_state, reward, terminated, truncated, info = env.step(action=action)
            _, active_gom = current_state.active()

            assert active_gom.array_table["A"] == StorageLocation.BOTH
            assert active_gom.array_table["B"] == StorageLocation.BOTH
            assert active_gom.array_table["C"] == StorageLocation.HOST
        elif i == 2:
            action = Action.SCHEDULE_MAP_NEST_HOST, maps["k"]
            current_state, reward, terminated, truncated, info = env.step(action=action)
            _, active_gom = current_state.active()

            assert active_gom.array_table["A"] == StorageLocation.BOTH
            assert active_gom.array_table["B"] == StorageLocation.BOTH
            assert active_gom.array_table["C"] == StorageLocation.HOST
        elif i == 3:
            action = Action.SCHEDULE_MAP_NEST_DEVICE, maps["i"]
            current_state, reward, terminated, truncated, info = env.step(action=action)
            _, active_gom = current_state.active()

            assert active_gom.array_table["A"] == StorageLocation.BOTH
            assert active_gom.array_table["B"] == StorageLocation.DEVICE
            assert active_gom.array_table["C"] == StorageLocation.HOST
        elif i == 4:
            action = Action.NEXT_STATE, None
            current_state, reward, terminated, truncated, info = env.step(action=action)
            _, active_gom = current_state.active()

            assert active_gom.array_table["A"] == StorageLocation.BOTH
            assert active_gom.array_table["B"] == StorageLocation.DEVICE
            assert active_gom.array_table["C"] == StorageLocation.HOST
        elif i == 5:
            action = Action.SCHEDULE_MAP_NEST_DEVICE, maps["m"]
            current_state, reward, terminated, truncated, info = env.step(action=action)
            _, active_gom = current_state.active()

            assert active_gom.array_table["A"] == StorageLocation.DEVICE
            assert active_gom.array_table["B"] == StorageLocation.DEVICE
            assert active_gom.array_table["C"] == StorageLocation.HOST
        elif i == 6:
            action = Action.COPY_DEVICE_TO_HOST, "B"
            current_state, reward, terminated, truncated, info = env.step(action=action)
            _, active_gom = current_state.active()

            assert active_gom.array_table["A"] == StorageLocation.DEVICE
            assert active_gom.array_table["B"] == StorageLocation.BOTH
            assert active_gom.array_table["C"] == StorageLocation.HOST
        elif i == 7:
            action = Action.COPY_DEVICE_TO_HOST, "A"
            current_state, reward, terminated, truncated, info = env.step(action=action)
            _, active_gom = current_state.active()

            assert active_gom.array_table["A"] == StorageLocation.BOTH
            assert active_gom.array_table["B"] == StorageLocation.BOTH
            assert active_gom.array_table["C"] == StorageLocation.HOST
        elif i == 8:
            action = Action.NEXT_STATE, None
            current_state, reward, terminated, truncated, info = env.step(action=action)
        else:
            assert False

        i += 1

    sdfg_opt = info["schedule"]

    A = np.random.random((256, 256)).astype(np.float64)
    A_opt = A.copy()
    B = np.random.random((256, 256)).astype(np.float64)
    B_opt = B.copy()
    C = np.zeros((256, 256), dtype=np.float64)
    C_opt = np.zeros((256, 256), dtype=np.float64)

    sdfg(A=A, B=B, C=C)
    sdfg_opt(A=A_opt, B=B_opt, C=C_opt)
    assert np.allclose(A, A_opt)
    assert np.allclose(B, B_opt)
    assert np.allclose(C, C_opt)
