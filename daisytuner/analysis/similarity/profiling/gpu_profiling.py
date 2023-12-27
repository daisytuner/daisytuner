# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import math
import torch
import platform

from typing import Dict

from daisytuner.profiling.metrics.metrics_factory import MetricsFactory
from daisytuner.profiling.likwid_helpers import gpu_codename


class GPUProfiling:
    def __init__(
        self,
        sdfg: dace.SDFG,
        hostname: str = platform.node(),
        codename: str = gpu_codename(),
    ):
        self._sdfg = sdfg
        self._hostname = hostname
        self._codename = codename
        self._encoding = None

    def encode(self) -> torch.tensor:
        if self._encoding is not None:
            return self._encoding

        retired_instructions = MetricsFactory.create(
            "Instructions", self._sdfg, self._hostname, self._codename
        )
        active_warps = MetricsFactory.create(
            "ActiveWarps", self._sdfg, self._hostname, self._codename
        )
        threads_launched = MetricsFactory.create(
            "ThreadsLaunched", self._sdfg, self._hostname, self._codename
        )
        branch_efficiency = MetricsFactory.create(
            "BranchEfficiency", self._sdfg, self._hostname, self._codename
        )
        dram_read_volume = MetricsFactory.create(
            "DramReadVolume", self._sdfg, self._hostname, self._codename
        )
        dram_write_volume = MetricsFactory.create(
            "DramWriteVolume", self._sdfg, self._hostname, self._codename
        )

        # Define features as statistics over threads: min, max, sum, mean, std
        data = torch.zeros(6, dtype=torch.float)
        data[0] = retired_instructions.compute()
        data[1] = active_warps.compute()
        data[2] = threads_launched.compute()
        data[3] = branch_efficiency.compute()
        data[4] = dram_read_volume.compute()
        data[5] = dram_write_volume.compute()

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
        return 6


TARGETS_GPU = [
    "Runtime",
]
