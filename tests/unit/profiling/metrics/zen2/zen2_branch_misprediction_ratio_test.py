# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import numpy as np

from pathlib import Path
from daisytuner.profiling.metrics.zen2 import BranchMispredictionRatio


def test_branching():
    def branching_static(A: dace.float64[16384]):
        for i in range(16384):
            if i % 2 == 0:
                A[i] = i
            else:
                A[i] = i + 1

    @dace.program
    def sdfg_branching_static(A: dace.float64[16384]):
        branching_static(A)

    def branching_dynamic(A: dace.float64[16384], A_ind: dace.int32[16384]):
        for i in range(16384):
            if A_ind[i] == 0:
                A[i] = i
            else:
                A[i] = i + 1

    @dace.program
    def sdfg_branching_dynamic(A: dace.float64[16384], A_ind: dace.int32[16384]):
        branching_dynamic(A, A_ind)

    sdfg_static = sdfg_branching_static.to_sdfg(simplify=False)
    sdfg_static.name = "sdfg_branching_static"
    sdfg_static.build_folder = str(
        Path(__file__).parent.parent.parent.parent / "test_data" / sdfg_static.name
    )

    sdfg_dynamic = sdfg_branching_dynamic.to_sdfg(simplify=False)
    sdfg_dynamic.name = "sdfg_branching_dynamic"
    sdfg_dynamic.build_folder = str(
        Path(__file__).parent.parent.parent.parent / "test_data" / sdfg_dynamic.name
    )

    # args_static = {
    #     "A": np.random.rand(
    #         16384,
    #     ).astype(np.float64),
    # }
    # indirect_vals = np.array([i % 2 for i in range(16384)], dtype=np.int32)
    # np.random.shuffle(indirect_vals)
    # args_dynamic = {
    #     "A": np.random.rand(
    #         16384,
    #     ).astype(np.float64),
    #     "A_ind": indirect_vals,
    # }

    metric_static = BranchMispredictionRatio(sdfg_static, hostname="ault17.cscs.ch")
    # metric_static.measure(arguments=args_static, keep_existing=False)
    assert metric_static.has_values()
    value_static = metric_static.compute()
    assert value_static < 0.01

    metric_dynamic = BranchMispredictionRatio(sdfg_dynamic, hostname="ault17.cscs.ch")
    # metric_dynamic.measure(arguments=args_dynamic, keep_existing=False)
    assert metric_dynamic.has_values()
    value_dynamic = metric_dynamic.compute()
    assert value_dynamic > 3.8
