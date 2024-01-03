# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import sys
import json
import dace
import copy
import pytest
import numpy as np
import importlib

from collections import Counter

from daisytuner.analysis.similarity.benchmarking import CPUBenchmark, GPUBenchmark

from daisytuner.device_mapping import Environment
from daisytuner.device_mapping.agents import ClassificationAgent

from daisytuner.normalization import APrioriMapNestNormalization
from daisytuner.transformations import MapWrapping
from daisytuner.profiling.helpers import random_arguments

from pathlib import Path

EXPECTED_NORMALIZATION = {
    "bert_base_uncased": Counter({3: 204, 2: 170, 4: 24, 1: 4}),
    "cloudsc": Counter({1: 1}),
    "efficientnet_b0": Counter({1: 1}),
    "fvt": Counter({1: 1}),
}


@pytest.mark.parametrize(
    "benchmark",
    [
        pytest.param("bert_base_uncased"),
        pytest.param("cloudsc"),
        pytest.param("efficientnet_b0"),
        pytest.param("fvt"),
    ],
)
def test_applications(application):
    sdfg = dace.SDFG.from_file(
        Path(__file__).parent / "applications" / application / f"{application}.sdfg"
    )
    sdfg_original = copy.deepcopy(sdfg)

    # 1. Normalization
    results = {}
    pipeline = APrioriMapNestNormalization()
    pipeline.apply_pass(sdfg, results)

    maps = []
    for nsdfg in sdfg.all_sdfgs_recursive():
        for state in nsdfg.states():
            for node in state.nodes():
                if not isinstance(node, dace.nodes.MapEntry):
                    continue

                maps.append(len(node.map.params))
    assert Counter(maps) == EXPECTED_NORMALIZATION[application]

    # Evaluation
    with open(
        Path(__file__).parent
        / "applications"
        / "applications_info"
        / f"{application}.json",
        "r",
    ) as handle:
        application_info = json.load(handle)

    init_module = load_module(
        Path(__file__).parent / "applications" / application / f"{application}.py",
        application,
    )
    init_function = getattr(
        init_module, application_info["application"]["init"]["func_name"]
    )
    args_original = dict(
        zip(application_info["application"]["init"]["output_args"], init_function())
    )
    args_original = {**random_arguments(sdfg_original), **args_original}
    args_original = copy.deepcopy(args_original)
    args_normalized = copy.deepcopy(args_original)
    args_opt = copy.deepcopy(args_original)

    sdfg_original(**args_original)
    sdfg(**args_normalized)

    for array, values in args_original.items():
        assert np.allclose(values, args_normalized[array], equal_nan=True)

    # 2. Device Mapping
    # sdfg.apply_transformations_repeated(MapWrapping)

    # host_benchmark = CPUBenchmark.from_cache("garbenheim")
    # device_benchmark = GPUBenchmark.from_cache("garbenheim")
    # env = Environment(
    #     sdfg=sdfg, cpu_benchmark=host_benchmark, gpu_benchmark=device_benchmark
    # )
    # agent = ClassificationAgent()
    # current_state = env.state
    # terminated = current_state.terminated
    # while not terminated:
    #     action = agent.action(current_state)
    #     current_state, reward, terminated, truncated, info = env.step(action=action)

    # sdfg_opt = info["schedule"]
    # sdfg_opt(**args_opt)
    # for array, values in args_original.items():
    #     assert np.allclose(values, args_opt[array], equal_nan=True)


def load_module(source, module_name=None):
    spec = importlib.util.spec_from_file_location(module_name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module
