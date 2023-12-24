# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import math
import torch
import platform

from typing import Dict

from daisytuner.profiling.metrics.metrics_factory import MetricsFactory
from daisytuner.profiling.likwid_helpers import cpu_codename


class CPUProfiling:
    """
    The profiling encoding provides features to analyze codes with
    data-dependent / non-statically analyzable properties.
    """

    def __init__(
        self,
        sdfg: dace.SDFG,
        hostname: str = platform.node(),
        codename: str = cpu_codename(),
    ):
        self._sdfg = sdfg
        self._hostname = hostname
        self._codename = codename
        self._encoding = None

    def encode(self) -> torch.tensor:
        if self._encoding is not None:
            return self._encoding

        # 1. Total workload
        retired_instructions = MetricsFactory.create(
            "Instructions", self._sdfg, self._hostname, self._codename
        )
        # 2. Degree of control-flow
        branch_rate = MetricsFactory.create(
            "BranchRate", self._sdfg, self._hostname, self._codename
        )
        # 3. Degree of data-dependent execution
        branch_misprediction_ratio = MetricsFactory.create(
            "BranchMispredictionRatio", self._sdfg, self._hostname, self._codename
        )
        # 4. L2 Cache behavior
        l2_volume = MetricsFactory.create(
            "L2Volume", self._sdfg, self._hostname, self._codename
        )
        # 5. DRAM<->Cache behavior
        load_rate = MetricsFactory.create(
            "LoadRate", self._sdfg, self._hostname, self._codename
        )
        # 6. DRAM<->Cache behavior
        store_rate = MetricsFactory.create(
            "StoreRate", self._sdfg, self._hostname, self._codename
        )

        # Define features as statistics over threads: min, max, sum, mean, std
        data = torch.zeros((6, 5), dtype=torch.float)

        values = retired_instructions.compute_per_thread()
        data[0, 0] = values.sum()
        data[0, 1] = values.mean()
        data[0, 2] = values.std()
        data[0, 3] = values.min()
        data[0, 4] = values.max()

        values = branch_rate.compute_per_thread()
        data[1, 0] = values.sum()
        data[1, 1] = values.mean()
        data[1, 2] = values.std()
        data[1, 3] = values.min()
        data[1, 4] = values.max()

        values = branch_misprediction_ratio.compute_per_thread()
        data[2, 0] = values.sum()
        data[2, 1] = values.mean()
        data[2, 2] = values.std()
        data[2, 3] = values.min()
        data[2, 4] = values.max()

        values = l2_volume.compute_per_thread()
        data[3, 0] = values.sum()
        data[3, 1] = values.mean()
        data[3, 2] = values.std()
        data[3, 3] = values.min()
        data[3, 4] = values.max()

        values = load_rate.compute_per_thread()
        data[4, 0] = values.sum()
        data[4, 1] = values.mean()
        data[4, 2] = values.std()
        data[4, 3] = values.min()
        data[4, 4] = values.max()

        values = store_rate.compute_per_thread()
        data[5, 0] = values.sum()
        data[5, 1] = values.mean()
        data[5, 2] = values.std()
        data[5, 3] = values.min()
        data[5, 4] = values.max()

        data = data.flatten()
        data = torch.log2(torch.log2(data + 1.0) + 1.0)
        self._encoding = data[None, :]
        return self._encoding

    def targets(self) -> torch.tensor:
        # 1. Runtime
        runtime = MetricsFactory.create(
            "Runtime", self._sdfg, self._hostname, self._codename
        )

        # Target vector
        data = torch.zeros(1, dtype=torch.float)
        data[0] = math.log2(runtime.compute())

        data = data[None, :]
        return data

    @staticmethod
    def dimensions() -> int:
        return 30


TARGETS_CPU = [
    "Runtime",
]
