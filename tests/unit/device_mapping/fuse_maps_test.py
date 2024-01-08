# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import numpy as np

from daisytuner.analysis.similarity.benchmarking import CPUBenchmark, GPUBenchmark

from daisytuner.device_mapping.state.storage_location import StorageLocation
from daisytuner.device_mapping import Environment, Action


def test_otf_map_fusion():
    @dace.program
    def sdfg_otf_map_fusion(A: dace.float64[512, 256], B: dace.float64[512, 256]):
        tmp = dace.define_local((512, 256), dtype=dace.float64)
        for i, j in dace.map[0:512, 0:256]:
            with dace.tasklet:
                a << A[i, j]
                b >> tmp[i, j]

                b = a + 1.0

        for k, l in dace.map[0:512, 0:256]:
            with dace.tasklet:
                a << tmp[k, l]
                b >> B[k, l]

                b = a + 1.0

    sdfg = sdfg_otf_map_fusion.to_sdfg()
    sdfg.simplify()

    host_benchmark = CPUBenchmark.from_cache("garbenheim")
    device_benchmark = GPUBenchmark.from_cache("garbenheim")
    env = Environment(
        sdfg=sdfg, cpu_benchmark=host_benchmark, gpu_benchmark=device_benchmark
    )
    current_state = env.state

    # Graph of maps
    gom = current_state.active()[1]
    first_map_entry = None
    second_map_entry = None
    for map_entry in gom.map_nests:
        if "i" in map_entry.map.params:
            first_map_entry = map_entry
        elif "k" in map_entry.map.params:
            second_map_entry = map_entry

    current_state = env.state
    _, active_gom = current_state.active()

    i = 0
    terminated = current_state.terminated
    while not terminated:
        if i == 0:
            map_entries = list(active_gom.map_nests.keys())
            assert len(map_entries) == 2
            assert len(active_gom.map_nest_schedules) == 2

            current_state, reward, terminated, truncated, info = env.step(
                action=(Action.FUSE_MAPS, (first_map_entry, second_map_entry))
            )

            _, active_gom = current_state.active()
        elif i == 1:
            map_entries = list(active_gom.map_nests.keys())
            assert len(map_entries) == 1
            assert len(active_gom.map_nest_schedules) == 1

            map_entry = map_entries[0]
            assert active_gom.map_nests[map_entry] in active_gom.map_nest_schedules

            current_state, reward, terminated, truncated, info = env.step(
                action=(Action.SCHEDULE_MAP_NEST_HOST, map_entry)
            )
        elif i == 2:
            action = Action.NEXT_STATE, None
            current_state, reward, terminated, truncated, info = env.step(action=action)
        else:
            assert False

        i += 1

    sdfg_opt = info["schedule"]

    A = np.random.random((512, 256)).astype(np.float64)
    A_opt = A.copy()
    B = np.random.random((512, 256)).astype(np.float64)
    B_opt = B.copy()

    sdfg(A=A, B=B)
    sdfg_opt(A=A_opt, B=B_opt)
    assert np.allclose(A, A_opt)
    assert np.allclose(B, B_opt)


def test_subgraph_fusion():
    @dace.program
    def sdfg_subgraph_fusion(A: dace.float64[2, 256], B: dace.float64[2, 256]):
        for i in dace.map[0:256]:
            with dace.tasklet:
                a << A[0, i]
                b >> B[0, i]

                b = a + 1.0

        for i in dace.map[0:256]:
            with dace.tasklet:
                a << A[1, i]
                b >> B[1, i]

                b = a * 2.0

    sdfg = sdfg_subgraph_fusion.to_sdfg()
    sdfg.simplify()

    host_benchmark = CPUBenchmark.from_cache("garbenheim")
    device_benchmark = GPUBenchmark.from_cache("garbenheim")
    env = Environment(
        sdfg=sdfg, cpu_benchmark=host_benchmark, gpu_benchmark=device_benchmark
    )
    current_state = env.state

    # Graph of maps
    state, gom = current_state.active()
    left_map_entry = None
    right_map_entry = None
    for map_entry in gom.map_nests:
        tasklet = [
            node
            for node in state.scope_subgraph(map_entry)
            if isinstance(node, dace.nodes.Tasklet)
        ][0]
        if "+" in tasklet.code.as_string:
            left_map_entry = map_entry
        elif "*" in tasklet.code.as_string:
            right_map_entry = map_entry

    current_state = env.state
    _, active_gom = current_state.active()

    i = 0
    terminated = current_state.terminated
    while not terminated:
        if i == 0:
            map_entries = list(active_gom.map_nests.keys())
            assert len(map_entries) == 2
            assert len(active_gom.map_nest_schedules) == 2

            current_state, reward, terminated, truncated, info = env.step(
                action=(Action.FUSE_MAPS, (left_map_entry, right_map_entry))
            )

            _, active_gom = current_state.active()
        elif i == 1:
            map_entries = list(active_gom.map_nests.keys())
            assert len(map_entries) == 1
            assert len(active_gom.map_nest_schedules) == 1

            map_entry = map_entries[0]
            assert active_gom.map_nests[map_entry] in active_gom.map_nest_schedules

            current_state, reward, terminated, truncated, info = env.step(
                action=(Action.SCHEDULE_MAP_NEST_HOST, map_entry)
            )
        elif i == 2:
            action = Action.NEXT_STATE, None
            current_state, reward, terminated, truncated, info = env.step(action=action)
        else:
            assert False

        i += 1

    sdfg_opt = info["schedule"]

    A = np.random.random((512, 256)).astype(np.float64)
    A_opt = A.copy()
    B = np.random.random((512, 256)).astype(np.float64)
    B_opt = B.copy()

    sdfg(A=A, B=B)
    sdfg_opt(A=A_opt, B=B_opt)
    assert np.allclose(A, A_opt)
    assert np.allclose(B, B_opt)
