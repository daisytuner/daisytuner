# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import copy
import pytest
import numpy as np

from collections import Counter

from daisytuner.device_mapping import Environment
from daisytuner.device_mapping.agents import DeviceAgent
from daisytuner.pipelines import APrioriMapNestNormalization
from daisytuner.profiling.measure import random_arguments
from daisytuner.transformations import MapWrapping

from pathlib import Path

EXPECTED_NORMALIZATION = {
    "2mm": Counter((3, 3, 2, 2, 2, 2)),
    "3mm": Counter((3, 3, 3, 2, 2, 2)),
    "atax": Counter((2, 2, 1, 1)),
    "bicg": Counter((2, 2, 1, 1)),
    "conv2d": Counter({7: 1, 4: 1}),
    "correlation": Counter({1: 8, 2: 7}),
    "covariance": Counter((1, 1, 1, 1, 1, 2, 2, 2, 2)),
    "fdtd-2d": Counter((1, 2, 2, 2)),
    "gemm": Counter((2, 2, 2, 3)),
    "gemver": Counter((2, 2, 2, 2, 1, 1, 1, 1, 1)),
    "gesummv": Counter((2, 2, 2, 2, 1, 1, 1)),
    "heat-3d": Counter((3, 3)),
    "jacobi-2d": Counter((2, 2)),
    "mvt": Counter((2, 2, 1, 1, 1, 1)),
    "nbody": Counter({2: 42, 1: 18}),
    "resnet": Counter({2: 36, 1: 18}),
    "syr2k": Counter((1, 1, 1, 2, 2, 2)),
    "syrk": Counter((1, 1, 1, 2, 2)),
    "vadv": Counter({2: 40}),
}


@pytest.mark.parametrize(
    "benchmark",
    [
        pytest.param("2mm"),
        pytest.param("3mm"),
        pytest.param("atax"),
        pytest.param("bicg"),
        # pytest.param("conv2d"),
        pytest.param("correlation"),
        pytest.param("covariance"),
        pytest.param("fdtd-2d"),
        pytest.param("gemm"),
        pytest.param("gemver"),
        pytest.param("gesummv"),
        pytest.param("heat-3d"),
        pytest.param("jacobi-2d"),
        pytest.param("mvt"),
        pytest.param("nbody"),
        # pytest.param("resnet"),
        # pytest.param("spmv"),
        pytest.param("syr2k"),
        pytest.param("syrk"),
        pytest.param("vadv"),
    ],
)
def test_npbench(benchmark):
    sdfg = dace.SDFG.from_file(Path(__file__).parent / "npbench" / f"{benchmark}.sdfg")
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

    # 2. Tuning for CPU-GPU
    sdfg.apply_transformations_repeated(MapWrapping)

    env = Environment(sdfg=sdfg)
    agent = DeviceAgent()

    # 2.a Run optimization
    terminated = False
    current_state = env._current_state
    while not terminated:
        action = agent.action(current_state)
        current_state, reward, terminated, truncated, info = env.step(action=action)

    # 2.b Numerical evaluation
    sdfg_tuned = info["scheduled_sdfg"]
    sdfg_tuned(**args_tuned)

    for array, values in args_original.items():
        assert np.allclose(values, args_normalized[array], equal_nan=True)
