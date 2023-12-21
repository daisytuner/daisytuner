# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import numpy as np

from pathlib import Path
from daisytuner.profiling.metrics.skylakeX import L2Volume


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

    sdfg_random = sdfg_indirect_copy.to_sdfg()
    sdfg_random.name = "sdfg_indirect_copy_random"
    sdfg_random.build_folder = str(
        Path(__file__).parent.parent.parent.parent / "test_data" / sdfg_random.name
    )

    # linear_accesses = np.array([i for i in range(65536)], dtype=np.int32)
    # args_linear = {
    #     "A": np.random.rand(
    #         65536,
    #     ).astype(np.float64),
    #     "B": np.random.rand(
    #         65536,
    #     ).astype(np.float64),
    #     "A_ind": linear_accesses,
    # }
    # random_accesses = np.copy(linear_accesses)
    # np.random.shuffle(random_accesses)
    # args_random = {
    #     "A": np.random.rand(
    #         65536,
    #     ).astype(np.float64),
    #     "B": np.random.rand(
    #         65536,
    #     ).astype(np.float64),
    #     "A_ind": random_accesses,
    # }

    metric_linear = L2Volume(sdfg_linear, hostname="ault01.cscs.ch")
    # metric_linear.measure(arguments=args_linear, keep_existing=False)
    assert metric_linear.has_values()
    value_linear = metric_linear.compute()
    assert value_linear < 2.0

    metric_random = L2Volume(sdfg_random, hostname="ault01.cscs.ch")
    # metric_random.measure(arguments=args_random, keep_existing=False)
    assert metric_random.has_values()
    value_random = metric_random.compute()
    assert value_random > 5.5
