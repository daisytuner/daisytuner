# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import copy
import pytest
import numpy as np

from collections import Counter

from daisytuner.analysis.similarity.benchmarking import CPUBenchmark, GPUBenchmark

from daisytuner.device_mapping import Environment
from daisytuner.device_mapping.agents import GreedyAgent

from daisytuner.normalization import APrioriMapNestNormalization
from daisytuner.transformations import MapWrapping
from daisytuner.profiling.helpers import random_arguments

from pathlib import Path

EXPECTED_NORMALIZATION = {
    "2mm_a": Counter((3, 3, 2, 2)),
    "2mm_b": Counter((3, 3, 2)),
    "3mm_a": Counter((3, 3, 3, 2, 2, 2)),
    "3mm_b": Counter((3, 3, 3, 2, 2)),
    "atax_a": Counter((2, 2, 1)),
    "atax_b": Counter((2, 2)),
    "bicg_a": Counter((2, 2, 1)),
    "bicg_b": Counter((2, 2)),
    "correlation_a": Counter({1: 7, 2: 3}),
    "covariance_a": Counter({1: 7, 2: 3}),
    "doitgen_a": Counter((2, 1, 1)),
    "fdtd-2d_a": Counter((1, 2, 2, 2)),
    "fdtd-2d_b": Counter((1, 2, 2, 2)),
    "gemm_a": Counter((2, 3)),
    "gemm_b": Counter((2, 3)),
    "gemver_a": Counter((2, 2, 2)),
    "gemver_b": Counter((2, 2, 2)),
    "gesummv_a": Counter((1, 1, 2, 2, 1)),
    "gesummv_b": Counter((2, 2, 1)),
    "heat-3d_a": Counter((3, 3)),
    "heat-3d_b": Counter((3, 3)),
    "jacobi-2d_a": Counter((2, 2)),
    "jacobi-2d_b": Counter((2, 2)),
    "mvt_a": Counter((2, 2)),
    "mvt_b": Counter((2, 2)),
    "syr2k_a": Counter((1, 1, 1, 2)),
    "syr2k_b": Counter((1, 1, 1, 2)),
    "syrk_a": Counter((1, 1, 1, 2)),
    "syrk_b": Counter((1, 1, 1, 2)),
}


@pytest.mark.parametrize(
    "benchmark",
    [
        pytest.param("2mm_a"),
        pytest.param("2mm_b"),
        pytest.param("3mm_a"),
        pytest.param("3mm_b"),
        pytest.param("atax_a"),
        pytest.param("atax_b"),
        pytest.param("bicg_a"),
        pytest.param("bicg_b"),
        pytest.param("correlation_a"),
        pytest.param("covariance_a"),
        pytest.param("fdtd-2d_a"),
        pytest.param("fdtd-2d_b"),
        pytest.param("gemm_a"),
        pytest.param("gemm_b"),
        pytest.param("gemver_a"),
        pytest.param("gemver_b"),
        pytest.param("gesummv_a"),
        pytest.param("gesummv_b"),
        pytest.param("heat-3d_a"),
        pytest.param("heat-3d_b"),
        pytest.param("jacobi-2d_a"),
        pytest.param("jacobi-2d_b"),
        pytest.param("mvt_a"),
        pytest.param("mvt_b"),
        pytest.param("syr2k_a"),
        pytest.param("syr2k_b"),
        pytest.param("syrk_a"),
        pytest.param("syrk_b"),
    ],
)
def test_polybench(benchmark):
    sdfg = dace.SDFG.from_file(
        Path(__file__).parent / "polybench" / f"{benchmark}.sdfg"
    )
    sdfg_original = copy.deepcopy(sdfg)

    # 1. Normalization
    results = {}
    pipeline = APrioriMapNestNormalization()
    pipeline.apply_pass(sdfg, results)

    # 1.a Normal-form
    assert APrioriMapNestNormalization.is_normalized(sdfg)

    # 1.b Number of maps
    maps = []
    for nsdfg in sdfg.all_sdfgs_recursive():
        for state in nsdfg.states():
            for node in state.nodes():
                if not isinstance(node, dace.nodes.MapEntry):
                    continue

                maps.append(len(node.map.params))
    assert Counter(maps) == EXPECTED_NORMALIZATION[benchmark]

    # 1.c Compile
    sdfg.compile()

    # 1.d Numerical evaluation
    args_original = random_arguments(sdfg_original)
    args_normalized = copy.deepcopy(args_original)
    args_tuned = copy.deepcopy(args_original)

    sdfg_original(**args_original)
    sdfg(**args_normalized)

    for array, values in args_original.items():
        assert np.allclose(values, args_normalized[array], equal_nan=True)

    # 2. Device Mapping
    sdfg.apply_transformations_repeated(MapWrapping)

    host_benchmark = CPUBenchmark.from_cache("garbenheim")
    device_benchmark = GPUBenchmark.from_cache("garbenheim")
    env = Environment(
        sdfg=sdfg, cpu_benchmark=host_benchmark, gpu_benchmark=device_benchmark
    )
    agent = GreedyAgent()
    current_state = env.state
    terminated = current_state.terminated
    while not terminated:
        action = agent.action(current_state)
        current_state, reward, terminated, truncated, info = env.step(action=action)
        if terminated:
            assert reward == 1.0
        else:
            assert reward == 0.0

    sdfg_opt = info["schedule"]
    sdfg_opt(**args_tuned)
    for array, values in args_original.items():
        assert np.allclose(values, args_tuned[array], equal_nan=True)
