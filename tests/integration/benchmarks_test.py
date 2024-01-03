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
from daisytuner.profiling.helpers import random_arguments

from pathlib import Path

EXPECTED_NORMALIZATION = {
    "adi": Counter({1: 12}),
    "adist": Counter({1: 1}),
    "atax": Counter({1: 2, 2: 2}),
    "azimhist": Counter({1: 7}),
    "azimnaiv": Counter({1: 3, 2: 1}),
    "bicg": Counter({1: 2, 2: 2}),
    "cavtflow": Counter({1: 9, 2: 5}),
    "chanflow": Counter({1: 16, 2: 9}),
    "cholesky": Counter(),
    "cholesky2": Counter(),
    "coninteg": Counter({1: 1, 2: 6, 3: 1}),
    "conv2d": Counter({7: 1, 4: 1}),
    "correlat": Counter({1: 8, 2: 7}),
    "covarian": Counter({1: 5, 2: 4}),
    "crc16": Counter(),
    "deriche": Counter({1: 12, 2: 2}),
    "doitgen": Counter({2: 2, 3: 1}),
    "durbin": Counter({1: 5}),
    "fdtd_2d": Counter({1: 1, 2: 3}),
    "floydwar": Counter({2: 1}),
    "gemm": Counter({2: 3, 3: 1}),
    "gemver": Counter({1: 5, 2: 4}),
    "gesummv": Counter({1: 3, 2: 4}),
    "gramschm": Counter({1: 4, 2: 2}),
    "hdiff": Counter({3: 1}),
    "heat3d": Counter({3: 2}),
    "jacobi1d": Counter({1: 2}),
    "jacobi2d": Counter({2: 2}),
    "k2mm": Counter({3: 2, 2: 4}),
    "k3mm": Counter({3: 3, 2: 3}),
    "lenet": Counter({2: 14, 4: 4, 5: 6, 3: 3}),
    "lu": Counter({1: 2}),
    "ludcmp": Counter({1: 5, 2: 1}),
    "mandel1": Counter({2: 7}),
    "mandel2": Counter({2: 7, 1: 5}),
    "mlp": Counter({2: 10, 3: 3, 1: 2}),
    "mvt": Counter({2: 2, 1: 4}),
    "nbody": Counter({2: 42, 1: 18}),
    "npgofast": Counter({2: 42, 1: 18}),
    "nussinov": Counter({1: 1, 2: 1}),
    "resnet": Counter({2: 36, 1: 18}),
    "seidel2d": Counter({1: 1}),
    "softmax": Counter({4: 4, 3: 2}),
    "spmv": Counter(),
    "sselfeng": Counter({1: 1}),
    "sthamfft": Counter({2: 7, 3: 2, 1: 1}),
    "symm": Counter({2: 1, 1: 3}),
    "syr2k": Counter({1: 3, 2: 3}),
    "syrk": Counter({1: 3, 2: 2}),
    "trisolv": Counter({1: 1}),
    "trmm": Counter({1: 1, 2: 1}),
    "vadv": Counter({2: 40}),
}


@pytest.mark.parametrize(
    "benchmark",
    [
        # pytest.param("adist"),
        pytest.param("atax"),
        pytest.param("azimhist"),
        pytest.param("azimnaiv"),
        pytest.param("bicg"),
        pytest.param("cavtflow"),
        pytest.param("chanflow"),
        pytest.param("cholesky"),
        pytest.param("cholesky2"),
        pytest.param("conv2d"),
        pytest.param("correlat"),
        pytest.param("covarian"),
        pytest.param("crc16"),
        pytest.param("deriche"),
        pytest.param("doitgen"),
        pytest.param("durbin"),
        pytest.param("fdtd_2d"),
        # pytest.param("floydwar"),
        pytest.param("gemm"),
        pytest.param("gemver"),
        pytest.param("gesummv"),
        pytest.param("gramschm"),
        pytest.param("hdiff"),
        pytest.param("heat3d"),
        pytest.param("jacobi1d"),
        pytest.param("jacobi2d"),
        pytest.param("k2mm"),
        pytest.param("k3mm"),
        pytest.param("lenet"),
        pytest.param("lu"),
        pytest.param("ludcmp"),
        pytest.param("mandel1"),
        pytest.param("mandel2"),
        pytest.param("mlp"),
        pytest.param("mvt"),
        pytest.param("nbody"),
        pytest.param("npgofast"),
        pytest.param("nussinov"),
        pytest.param("resnet"),
        pytest.param("seidel2d"),
        # pytest.param("softmax"),
        pytest.param("spmv"),
        pytest.param("sselfeng"),
        pytest.param("sthamfft"),
        pytest.param("symm"),
        pytest.param("syr2k"),
        pytest.param("syrk"),
        pytest.param("trisolv"),
        pytest.param("trmm"),
        pytest.param("vadv"),
    ],
)
def test_benchmarks(benchmark):
    sdfg = dace.SDFG.from_file(
        Path(__file__).parent / "benchmarks" / benchmark / f"{benchmark}.sdfg"
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
    assert Counter(maps) == EXPECTED_NORMALIZATION[benchmark]

    # Evaluation
    with open(
        Path(__file__).parent / "benchmarks" / "bench_info" / f"{benchmark}.json", "r"
    ) as handle:
        bench_info = json.load(handle)

    init_module = load_module(
        Path(__file__).parent / "benchmarks" / benchmark / f"{benchmark}.py", benchmark
    )
    init_function = getattr(init_module, bench_info["benchmark"]["init"]["func_name"])

    init_args = {
        p: sdfg.constants[p] for p in bench_info["benchmark"]["init"]["input_args"]
    }
    args_original = dict(
        zip(bench_info["benchmark"]["init"]["output_args"], init_function(**init_args))
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
    host_benchmark = CPUBenchmark.from_cache("garbenheim")
    device_benchmark = GPUBenchmark.from_cache("garbenheim")

    sdfg_opt = Environment.preprocess(sdfg)
    env = Environment(
        sdfg=sdfg_opt, cpu_benchmark=host_benchmark, gpu_benchmark=device_benchmark
    )
    agent = ClassificationAgent()
    current_state = env.state
    terminated = current_state.terminated
    while not terminated:
        action = agent.action(current_state)
        current_state, reward, terminated, truncated, info = env.step(action=action)

    sdfg_opt = info["schedule"]
    sdfg_opt(**args_opt)
    for array, values in args_original.items():
        assert np.allclose(values, args_opt[array], equal_nan=True)


def load_module(source, module_name=None):
    spec = importlib.util.spec_from_file_location(module_name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module
