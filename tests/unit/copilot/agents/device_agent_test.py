# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import numpy as np

from daisytuner.copilot import Environment, Action
from daisytuner.copilot.agents import DeviceAgent
from daisytuner.copilot.rewards import Validity


def test_one_map():
    @dace.program
    def sdfg_one_map(
        A: dace.float64[1024, 128], B: dace.float64[128], C: dace.float64[1024]
    ):
        for i, k in dace.map[0:1024, 0:128]:
            with dace.tasklet:
                a << A[i, k]
                b << B[k]
                c >> C(1, lambda e, f: e + f)[i]

                c = a * b

    sdfg = sdfg_one_map.to_sdfg()
    sdfg.simplify()

    env = Environment(sdfg=sdfg, reward_function=Validity())
    agent = DeviceAgent()

    terminated = False
    current_state = env._current_state
    while not terminated:
        action = agent.action(current_state)
        current_state, reward, terminated, truncated, info = env.step(action=action)

        assert not truncated
        if terminated:
            assert reward == 1.0
        else:
            assert reward == 0.0

    sdfg_opt = current_state.generate_schedule()

    A = np.random.random((1024, 128)).astype(np.float64)
    B = np.random.random((128,)).astype(np.float64)
    C = np.zeros((1024,), dtype=np.float64)
    C_opt = np.zeros((1024,), dtype=np.float64)

    sdfg(A=A, B=B, C=C)
    sdfg_opt(A=A, B=B, C=C_opt)
    assert np.allclose(C, C_opt)


def test_two_maps():
    @dace.program
    def sdfg_two_maps(
        A: dace.float64[1024, 128], B: dace.float64[128], C: dace.float64[1024, 128]
    ):
        for i, k in dace.map[0:1024, 0:128]:
            with dace.tasklet:
                a << A[i, k]
                c >> C[i, k]

                c = a

        for i, k in dace.map[0:1024, 0:128]:
            with dace.tasklet:
                b << B[k]
                c1 << C[i, k]
                c2 >> C[i, k]

                c2 = c1 + b

    sdfg = sdfg_two_maps.to_sdfg()
    sdfg.simplify()

    env = Environment(sdfg=sdfg, reward_function=Validity())
    agent = DeviceAgent()

    terminated = False
    current_state = env._current_state
    while not terminated:
        action = agent.action(current_state)
        current_state, reward, terminated, truncated, info = env.step(action=action)

        assert not truncated
        if terminated:
            assert reward == 1.0
        else:
            assert reward == 0.0

    sdfg_opt = current_state.generate_schedule()

    A = np.random.random((1024, 128)).astype(np.float64)
    B = np.random.random((128,)).astype(np.float64)
    C = np.zeros((1024, 128), dtype=np.float64)
    C_opt = np.zeros((1024, 128), dtype=np.float64)

    sdfg(A=A, B=B, C=C)
    sdfg_opt(A=A, B=B, C=C_opt)
    assert np.allclose(C, C_opt)


def test_two_states():
    @dace.program
    def sdfg_two_states(
        A: dace.float64[1024, 128], B: dace.float64[128], C: dace.float64[1024, 128]
    ):
        for i, k in dace.map[0:1024, 0:128]:
            with dace.tasklet:
                b << B[k]
                c >> C[i, k]

                c = b

        for i, k in dace.map[0:1024, 0:128]:
            with dace.tasklet:
                a << A[i, k]
                b << B[k]
                c >> C(1, lambda e, f: e + f)[i, 0]

                c = a * b

    sdfg = sdfg_two_states.to_sdfg()
    sdfg.simplify()

    env = Environment(sdfg=sdfg, reward_function=Validity())
    agent = DeviceAgent()

    terminated = False
    current_state = env._current_state
    while not terminated:
        action = agent.action(current_state)
        current_state, reward, terminated, truncated, info = env.step(action=action)

        assert not truncated
        if terminated:
            assert reward == 1.0
        else:
            assert reward == 0.0

    sdfg_opt = current_state.generate_schedule()

    A = np.random.random((1024, 128)).astype(np.float64)
    B = np.random.random((128,)).astype(np.float64)
    C = np.zeros((1024, 128), dtype=np.float64)
    C_opt = np.zeros((1024, 128), dtype=np.float64)

    sdfg(A=A, B=B, C=C)
    sdfg_opt(A=A, B=B, C=C_opt)
    assert np.allclose(C, C_opt)
