# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import numpy as np

from pathlib import Path
from daisytuner.profiling.metrics.haswellEP import LoadStoreRatio


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

    metric = LoadStoreRatio(sdfg, hostname="garbenheim")
    # args = {
    #     "A": np.random.rand(1024, 1024).astype(np.float64),
    #     "B": np.random.rand(1024, 1024).astype(np.float64),
    # }
    # metric.measure(arguments=args, keep_existing=False)
    assert metric.has_values()
    value = metric.compute()
    # L3 is a victim cache, data is not written back to DRAM
    # Store = 0, Load = num cache lines of A
    assert np.allclose(value, 1024 * 1024 * 8 / 64, atol=0.0, rtol=2.0)


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

    metric = LoadStoreRatio(sdfg, hostname="garbenheim")
    # args = {
    #     "A": np.random.rand(32768).astype(np.float64),
    #     "B": np.random.rand(32768).astype(np.float64),
    #     "C": np.random.rand(32768).astype(np.float64),
    # }
    # metric.measure(arguments=args, keep_existing=False)
    assert metric.has_values()
    value = metric.compute()
    # L3 is a victim cache, data is not written back to DRAM
    # Store = 0, Load = num cache lines of A
    assert np.allclose(value, 3 * 32768 * 8 / 64, atol=0.0, rtol=2.0)
