# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import numpy as np

from daisytuner.analysis.similarity.benchmarking import CPUBenchmark, GPUBenchmark

from daisytuner.device_mapping import Environment
from daisytuner.device_mapping.agents import GreedyAgent


def test_single_state():
    @dace.program
    def sdfg_single_state(
        A: dace.float64[512, 256], B: dace.float64[256], C: dace.float64[512]
    ):
        for i, j in dace.map[0:512, 0:256]:
            with dace.tasklet:
                a << A[i, j]
                b << B[j]
                c >> C(1, lambda e, f: e + f)[i]

                c = a * b

    sdfg = sdfg_single_state.to_sdfg()
    sdfg.simplify()

    host_benchmark = CPUBenchmark.from_cache("garbenheim")
    device_benchmark = GPUBenchmark.from_cache("garbenheim")
    env = Environment(
        sdfg=sdfg, cpu_benchmark=host_benchmark, gpu_benchmark=device_benchmark
    )
    current_state = env.state

    # Graph of states
    assert len(current_state.nodes()) == 2
    assert len(current_state.edges()) == 1
    assert current_state.active()[0] == sdfg.start_state

    # Graph of maps
    gom = current_state.active()[1]
    assert len(gom.nodes()) == 1
    assert len(gom.edges()) == 0
    assert gom.array_table is not None
    assert all([dest == dace.DeviceType.CPU for dest in gom.array_table.values()])

    agent = GreedyAgent()
    terminated = current_state.terminated
    while not terminated:
        action = agent.action(current_state)
        current_state, reward, terminated, truncated, info = env.step(action=action)
        if terminated:
            assert reward == 1.0
        else:
            assert reward == 0.0

    sdfg_opt = info["schedule"]

    A = np.random.random((512, 256)).astype(np.float64)
    B = np.random.random((256,)).astype(np.float64)
    C = np.zeros((512,), dtype=np.float64)
    C_opt = np.zeros((512,), dtype=np.float64)

    sdfg(A=A, B=B, C=C)
    sdfg_opt(A=A, B=B, C=C_opt)
    assert np.allclose(C, C_opt)


def test_multiple_map_nests():
    @dace.program
    def sdfg_multiple_map_nests(
        A: dace.float64[512, 256], B: dace.float64[256], C: dace.float64[512]
    ):
        for i, j in dace.map[0:512, 0:256]:
            with dace.tasklet:
                a << A[i, j]
                b << B[j]
                c >> C(1, lambda e, f: e + f)[i]

                c = a * b

        for i in dace.map[0:512]:
            with dace.tasklet:
                c1 << C[i]
                c2 >> C[i]

                c2 = c1 + 1.0

    sdfg = sdfg_multiple_map_nests.to_sdfg()
    sdfg.simplify()

    host_benchmark = CPUBenchmark.from_cache("garbenheim")
    device_benchmark = GPUBenchmark.from_cache("garbenheim")
    env = Environment(
        sdfg=sdfg, cpu_benchmark=host_benchmark, gpu_benchmark=device_benchmark
    )
    current_state = env.state

    # Graph of states
    assert len(current_state.nodes()) == 2
    assert len(current_state.edges()) == 1
    assert current_state.active()[0] == sdfg.start_state

    # Graph of maps
    gom = current_state.active()[1]
    assert len(gom.nodes()) == 2
    assert len(gom.edges()) == 1
    assert gom.array_table is not None
    assert all([dest == dace.DeviceType.CPU for dest in gom.array_table.values()])

    agent = GreedyAgent()
    terminated = current_state.terminated
    while not terminated:
        action = agent.action(current_state)
        current_state, reward, terminated, truncated, info = env.step(action=action)
        if terminated:
            assert reward == 1.0
        else:
            assert reward == 0.0

    sdfg_opt = info["schedule"]

    A = np.random.random((512, 256)).astype(np.float64)
    B = np.random.random((256,)).astype(np.float64)
    C = np.zeros((512,), dtype=np.float64)
    C_opt = np.zeros((512,), dtype=np.float64)

    sdfg(A=A, B=B, C=C)
    sdfg_opt(A=A, B=B, C=C_opt)
    assert np.allclose(C, C_opt)


def test_two_states():
    @dace.program
    def sdfg_two_states(
        A: dace.float64[512, 256], B: dace.float64[256], C: dace.float64[512]
    ):
        for i in dace.map[0:512]:
            with dace.tasklet:
                c >> C[i]
                c = 0

        for i, j in dace.map[0:512, 0:256]:
            with dace.tasklet:
                a << A[i, j]
                b << B[j]
                c >> C(1, lambda e, f: e + f)[i]

                c = a * b

    sdfg = sdfg_two_states.to_sdfg()
    sdfg.simplify()

    host_benchmark = CPUBenchmark.from_cache("garbenheim")
    device_benchmark = GPUBenchmark.from_cache("garbenheim")
    env = Environment(
        sdfg=sdfg, cpu_benchmark=host_benchmark, gpu_benchmark=device_benchmark
    )
    current_state = env.state

    # Graph of states
    assert len(current_state.nodes()) == 3
    assert len(current_state.edges()) == 2
    assert current_state.active()[0] == sdfg.start_state

    # Graph of maps
    gom = current_state.active()[1]
    assert len(gom.nodes()) == 1
    assert len(gom.edges()) == 0
    assert gom.array_table is not None
    assert all([dest == dace.DeviceType.CPU for dest in gom.array_table.values()])

    agent = GreedyAgent()
    terminated = current_state.terminated
    while not terminated:
        action = agent.action(current_state)
        current_state, reward, terminated, truncated, info = env.step(action=action)
        if terminated:
            assert reward == 1.0
        else:
            assert reward == 0.0

    sdfg_opt = info["schedule"]

    A = np.random.random((512, 256)).astype(np.float64)
    B = np.random.random((256,)).astype(np.float64)
    C = np.zeros((512,), dtype=np.float64)
    C_opt = np.zeros((512,), dtype=np.float64)

    sdfg(A=A, B=B, C=C)
    sdfg_opt(A=A, B=B, C=C_opt)
    assert np.allclose(C, C_opt)


def test_loop():
    @dace.program
    def sdfg_loop(
        A: dace.float64[512, 256], B: dace.float64[256], C: dace.float64[512]
    ):
        for i in dace.map[0:512]:
            with dace.tasklet:
                c >> C[i]
                c = 0

        for k in range(256):
            for i, j in dace.map[0:512, 0:k]:
                with dace.tasklet:
                    a << A[i, j]
                    b << B[j]
                    c >> C(1, lambda e, f: e + f)[i]

                    c = a * b

    sdfg = sdfg_loop.to_sdfg()
    sdfg.simplify()

    host_benchmark = CPUBenchmark.from_cache("garbenheim")
    device_benchmark = GPUBenchmark.from_cache("garbenheim")
    env = Environment(
        sdfg=sdfg, cpu_benchmark=host_benchmark, gpu_benchmark=device_benchmark
    )
    current_state = env.state

    # Graph of states
    assert len(current_state.nodes()) == 5
    assert len(current_state.edges()) == 4
    assert not current_state.has_cycles()
    assert current_state.active()[0] == sdfg.start_state

    # Graph of maps
    gom = current_state.active()[1]
    assert len(gom.nodes()) == 1
    assert len(gom.edges()) == 0
    assert gom.array_table is not None
    assert all([dest == dace.DeviceType.CPU for dest in gom.array_table.values()])

    agent = GreedyAgent()
    terminated = current_state.terminated
    while not terminated:
        action = agent.action(current_state)
        current_state, reward, terminated, truncated, info = env.step(action=action)
        if terminated:
            assert reward == 1.0
        else:
            assert reward == 0.0

    sdfg_opt = info["schedule"]

    A = np.random.random((512, 256)).astype(np.float64)
    B = np.random.random((256,)).astype(np.float64)
    C = np.zeros((512,), dtype=np.float64)
    C_opt = np.zeros((512,), dtype=np.float64)

    sdfg(A=A, B=B, C=C)
    sdfg_opt(A=A, B=B, C=C_opt)
    assert np.allclose(C, C_opt)


def test_loop_with_branches():
    @dace.program
    def sdfg_loop_with_branches(
        A: dace.float64[512, 256], B: dace.float64[256], C: dace.float64[512]
    ):
        for i in dace.map[0:512]:
            with dace.tasklet:
                c >> C[i]
                c = 0

        for k in range(256):
            if k < 128:
                for i, j in dace.map[0:512, 0:k]:
                    with dace.tasklet:
                        a << A[i, j]
                        b << B[j]
                        c >> C(1, lambda e, f: e + f)[i]

                        c = a * b
            else:
                for i in dace.map[0:512]:
                    with dace.tasklet:
                        c1 << C[i]
                        c2 >> C[i]
                        c2 = c1 + 1

    sdfg = sdfg_loop_with_branches.to_sdfg()
    sdfg.simplify()

    host_benchmark = CPUBenchmark.from_cache("garbenheim")
    device_benchmark = GPUBenchmark.from_cache("garbenheim")
    env = Environment(
        sdfg=sdfg, cpu_benchmark=host_benchmark, gpu_benchmark=device_benchmark
    )
    current_state = env.state

    # Graph of states
    assert len(current_state.nodes()) == 8
    assert len(current_state.edges()) == 8
    assert not current_state.has_cycles()
    assert current_state.active()[0] == sdfg.start_state

    # Graph of maps
    gom = current_state.active()[1]
    assert len(gom.nodes()) == 1
    assert len(gom.edges()) == 0
    assert gom.array_table is not None
    assert all([dest == dace.DeviceType.CPU for dest in gom.array_table.values()])

    agent = GreedyAgent()
    terminated = current_state.terminated
    while not terminated:
        action = agent.action(current_state)
        current_state, reward, terminated, truncated, info = env.step(action=action)
        if terminated:
            assert reward == 1.0
        else:
            assert reward == 0.0

    sdfg_opt = info["schedule"]

    A = np.random.random((512, 256)).astype(np.float64)
    B = np.random.random((256,)).astype(np.float64)
    C = np.zeros((512,), dtype=np.float64)
    C_opt = np.zeros((512,), dtype=np.float64)

    sdfg(A=A, B=B, C=C)
    sdfg_opt(A=A, B=B, C=C_opt)
    assert np.allclose(C, C_opt)
