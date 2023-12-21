# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import numpy as np

from pathlib import Path
from daisytuner.profiling.metrics.zen2 import FLOP


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

    metric = FLOP(sdfg, hostname="ault17.cscs.ch")
    # args = {
    #     "A": np.random.rand(1024, 1024).astype(np.float64),
    #     "B": np.random.rand(1024, 1024).astype(np.float64),
    # }
    # metric.measure(arguments=args, keep_existing=False)
    assert metric.has_values()
    value = metric.compute()
    assert np.allclose(value, 0.0, atol=2e-5, rtol=0.0)


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

    metric = FLOP(sdfg, hostname="ault17.cscs.ch")
    # args = {
    #     "A": np.random.rand(32768).astype(np.float64),
    #     "B": np.random.rand(32768).astype(np.float64),
    #     "C": np.random.rand(32768).astype(np.float64),
    # }
    # metric.measure(arguments=args, keep_existing=False)
    assert metric.has_values()
    value = metric.compute()
    assert np.allclose(value, (32768 * 2) * 1e-6, atol=2e-5, rtol=0.0)
