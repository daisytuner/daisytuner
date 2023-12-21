# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import numpy as np

from pathlib import Path
from daisytuner.profiling.metrics.metrics_factory import MetricsFactory


def test_create_metric():
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
        Path(__file__).parent.parent.parent / "test_data" / sdfg.name
    )

    metric = MetricsFactory.create(
        "FLOP", sdfg, hostname="lukas-Nitro", codename="zen3"
    )
    assert metric.has_values()
    assert np.allclose(metric.compute(), (32768 * 2 + 10) * 1e-6, atol=1e-6, rtol=0.0)


def test_unknown_architecture():
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
        Path(__file__).parent.parent.parent / "test_data" / sdfg.name
    )

    try:
        metric = MetricsFactory.create(
            "FLOP", sdfg, hostname="lukas-Nitro", codename="zenA"
        )
        assert False
    except ModuleNotFoundError:
        assert True


def test_unknown_metric():
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
        Path(__file__).parent.parent.parent / "test_data" / sdfg.name
    )

    try:
        metric = MetricsFactory.create(
            "FLOP2", sdfg, hostname="lukas-Nitro", codename="zen"
        )
        assert False
    except ModuleNotFoundError:
        assert True
