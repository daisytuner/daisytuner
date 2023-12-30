# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
from __future__ import annotations

import torch
import json
import numpy as np

from typing import Dict
from pathlib import Path

from daisytuner.profiling import Benchmarking


class GPUBenchmark:

    NUM_FEATURES = 5

    def __init__(self, gpu_info: Dict = None) -> None:
        self._data = None
        self._gpu_info = gpu_info

    def encode(self) -> torch.tensor:
        if self._data is not None:
            return self._data

        if not self._gpu_info:
            # Gather benchmarking data
            bench = Benchmarking()
            self._gpu_info = bench.analyze()["gpu"]

        vec = np.zeros((GPUBenchmark.NUM_FEATURES,), dtype=np.float32)
        vec[0] = self._gpu_info["compute_capability"]
        vec[1] = self._gpu_info["l2_cache"]
        vec[2] = self._gpu_info["memory"]
        vec[3] = self._gpu_info["clock_rate"]
        vec[4] = self._gpu_info["mem_clock_rate"]

        self._data = torch.tensor(vec, dtype=torch.float)[None, :]
        self._data = torch.log2(self._data + 1.0)

        return self._data

    @staticmethod
    def dimensions():
        return GPUBenchmark.NUM_FEATURES

    @staticmethod
    def from_cache(hostname: str) -> GPUBenchmark:
        with open(Path.home() / ".daisy" / f"{hostname}.json", "r") as handle:
            data = json.load(handle)
            return GPUBenchmark(gpu_info=data["gpu"])
