# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import numpy as np

from daisytuner.analysis.similarity.benchmarking import CPUBenchmark, GPUBenchmark

from daisytuner.device_mapping import Environment, Action
from daisytuner.device_mapping.agents import FusionAgent


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
    agent = FusionAgent()
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
    A_opt = A.copy()
    B = np.random.random((512, 256)).astype(np.float64)
    B_opt = B.copy()

    sdfg(A=A, B=B)
    sdfg_opt(A=A_opt, B=B_opt)
    assert np.allclose(A, A_opt)
    assert np.allclose(B, B_opt)
