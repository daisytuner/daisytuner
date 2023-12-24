# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import json
import numpy as np

from pathlib import Path

from daisytuner.analysis.similarity import MapNest, MapNestModel
from daisytuner.analysis.similarity.benchmarking import CPUBenchmark
from daisytuner.profiling.metrics import MetricsFactory


def test_copy():
    @dace.program
    def sdfg_copy(A: dace.float64[1024, 1024], B: dace.float64[1024, 1024]):
        for i, j in dace.map[0:1024, 0:1024]:
            with dace.tasklet:
                a << A[i, j]
                b >> B[i, j]

                b = a

    sdfg = sdfg_copy.to_sdfg()
    sdfg.name = "sdfg_copy"
    sdfg.build_folder = str(
        Path(__file__).parent.parent.parent.parent / "test_data" / sdfg.name
    )
    map_entry = None
    for node in sdfg.start_state.nodes():
        if isinstance(node, dace.nodes.MapEntry):
            map_entry = node
            break
    map_nest = MapNest(sdfg.start_state, root=map_entry)

    # Default model
    model = MapNestModel.create(dace.DeviceType.CPU)
    archs_path = (
        Path(__file__).parent.parent.parent.parent / "test_data" / "architectures"
    )

    # haswellEP
    haswellEP_runtime = MetricsFactory.create(
        "Runtime", sdfg=sdfg, hostname="garbenheim", codename="haswellEP"
    )
    target_runtime = haswellEP_runtime.compute()

    with open(archs_path / "garbenheim.json", "r") as handle:
        haswellEP_cpu_info = json.load(handle)
        haswellEP_benchmark = CPUBenchmark(cpu_info=haswellEP_cpu_info)
    pred, *_ = model.predict(map_nest=map_nest, benchmark=haswellEP_benchmark)
    assert np.allclose(pred, target_runtime, atol=0.0, rtol=0.2)


def test_triad():
    @dace.program
    def sdfg_triad(
        A: dace.float64[32768],
        B: dace.float64[32768],
        C: dace.float64[32768],
    ):
        for i in dace.map[0:32768]:
            with dace.tasklet:
                a >> A[i]
                b << B[i]
                c << C[i]

                a = b + c * 2.0

    sdfg = sdfg_triad.to_sdfg()
    sdfg.name = "sdfg_triad"
    sdfg.build_folder = str(
        Path(__file__).parent.parent.parent.parent / "test_data" / sdfg.name
    )
    map_entry = None
    for node in sdfg.start_state.nodes():
        if isinstance(node, dace.nodes.MapEntry):
            map_entry = node
            break
    map_nest = MapNest(sdfg.start_state, root=map_entry)

    # Default model
    model = MapNestModel.create(dace.DeviceType.CPU)
    archs_path = (
        Path(__file__).parent.parent.parent.parent / "test_data" / "architectures"
    )

    # haswellEP
    haswellEP_runtime = MetricsFactory.create(
        "Runtime", sdfg=sdfg, hostname="garbenheim", codename="haswellEP"
    )
    target_runtime = haswellEP_runtime.compute()

    with open(archs_path / "garbenheim.json", "r") as handle:
        haswellEP_cpu_info = json.load(handle)
        haswellEP_benchmark = CPUBenchmark(cpu_info=haswellEP_cpu_info)
    pred, *_ = model.predict(map_nest=map_nest, benchmark=haswellEP_benchmark)
    assert np.allclose(pred, target_runtime, atol=0.0, rtol=0.2)


def test_indirect_copy():
    @dace.program
    def sdfg_indirect_copy(
        A: dace.float64[65536], B: dace.float64[65536], A_ind: dace.int32[65536]
    ):
        for i in dace.map[0:65536]:
            with dace.tasklet:
                a << A[A_ind[i]]
                b >> B[i]

                b = a

    sdfg_linear = sdfg_indirect_copy.to_sdfg()
    sdfg_linear.name = "sdfg_indirect_copy_linear"
    sdfg_linear.build_folder = str(
        Path(__file__).parent.parent.parent.parent / "test_data" / sdfg_linear.name
    )
    map_entry = None
    for node in sdfg_linear.start_state.nodes():
        if isinstance(node, dace.nodes.MapEntry):
            map_entry = node
            break
    map_nest = MapNest(sdfg_linear.start_state, root=map_entry)

    # Default model
    model = MapNestModel.create(dace.DeviceType.CPU)
    archs_path = (
        Path(__file__).parent.parent.parent.parent / "test_data" / "architectures"
    )

    # haswellEP
    haswellEP_runtime = MetricsFactory.create(
        "Runtime", sdfg=sdfg_linear, hostname="garbenheim", codename="haswellEP"
    )
    target_runtime = haswellEP_runtime.compute()

    with open(archs_path / "garbenheim.json", "r") as handle:
        haswellEP_cpu_info = json.load(handle)
        haswellEP_benchmark = CPUBenchmark(cpu_info=haswellEP_cpu_info)
    pred, *_ = model.predict(map_nest=map_nest, benchmark=haswellEP_benchmark)
    assert np.allclose(pred, target_runtime, atol=0.0, rtol=0.3)

    sdfg_random = sdfg_indirect_copy.to_sdfg()
    sdfg_random.name = "sdfg_indirect_copy_random"
    sdfg_random.build_folder = str(
        Path(__file__).parent.parent.parent.parent / "test_data" / sdfg_random.name
    )
    map_entry = None
    for node in sdfg_random.start_state.nodes():
        if isinstance(node, dace.nodes.MapEntry):
            map_entry = node
            break
    map_nest = MapNest(sdfg_random.start_state, root=map_entry)

    # haswellEP
    haswellEP_runtime = MetricsFactory.create(
        "Runtime", sdfg=sdfg_random, hostname="garbenheim", codename="haswellEP"
    )
    target_runtime = haswellEP_runtime.compute()
    pred, *_ = model.predict(map_nest=map_nest, benchmark=haswellEP_benchmark)
    assert np.allclose(pred, target_runtime, atol=0.0, rtol=0.2)
