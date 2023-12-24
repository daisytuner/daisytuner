# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import torch
import numpy as np

from typing import Dict

from daisytuner.profiling import Benchmarking


class CPUBenchmark:

    NUM_FEATURES = 9

    def __init__(self, cpu_info: Dict = None) -> None:
        self._data = None
        self._cpu_info = cpu_info

    def encode(self) -> torch.tensor:
        if self._data is not None:
            return self._data

        if not self._cpu_info:
            # Gather benchmarking data
            bench = Benchmarking()
            self._cpu_info = bench.analyze()

        vec = np.zeros((CPUBenchmark.NUM_FEATURES,), dtype=np.float32)
        vec[0] = self._cpu_info["num_sockets"]
        vec[1] = self._cpu_info["cores_per_socket"]
        vec[2] = self._cpu_info["threads_per_core"]
        vec[3] = self._cpu_info["peakflops"]
        vec[4] = self._cpu_info["peakflops_avx"]
        vec[5] = self._cpu_info["stream_load"]
        vec[6] = self._cpu_info["stream_store"]
        vec[7] = self._cpu_info["stream_copy"]
        vec[8] = self._cpu_info["stream_triad"]

        self._data = torch.tensor(vec, dtype=torch.float)[None, :]
        self._data = torch.log2(self._data + 1.0)
        return self._data

    @staticmethod
    def dimensions():
        return CPUBenchmark.NUM_FEATURES
