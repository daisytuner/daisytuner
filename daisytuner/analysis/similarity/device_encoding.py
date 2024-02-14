# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import torch
import numpy as np

from typing import Dict

from daisytuner.profiling import Benchmarking


class CPUEncoding:

    NUM_FEATURES = 11

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

        vec = np.zeros((CPUEncoding.NUM_FEATURES,), dtype=np.float32)
        vec[0] = self._cpu_info["num_sockets"]
        vec[1] = self._cpu_info["cores_per_socket"]
        vec[2] = self._cpu_info["threads_per_core"]
        vec[3] = self._cpu_info["l2_cache"]
        vec[4] = self._cpu_info["l3_cache"]
        vec[5] = self._cpu_info["peakflops"]
        vec[6] = self._cpu_info["peakflops_avx"]
        vec[7] = self._cpu_info["stream_load"]
        vec[8] = self._cpu_info["stream_store"]
        vec[9] = self._cpu_info["stream_copy"]
        vec[10] = self._cpu_info["stream_triad"]

        self._data = torch.tensor(vec, dtype=torch.float)[None, :]
        return self._data

    @staticmethod
    def dimensions():
        return CPUEncoding.NUM_FEATURES


class GPUEncoding:

    NUM_FEATURES = 7

    def __init__(self, gpu_info: Dict) -> None:
        self._data = None
        self._gpu_info = gpu_info

    def encode(self) -> torch.tensor:
        if self._data is not None:
            return self._data

        vec = np.zeros((GPUEncoding.NUM_FEATURES,), dtype=np.float32)
        vec[0] = self._gpu_info["devices"]
        vec[1] = self._gpu_info["compute_capability"]
        vec[2] = self._gpu_info["l2_cache"]
        vec[3] = self._gpu_info["memory"]
        vec[4] = self._gpu_info["SIMD_width"]
        vec[5] = self._gpu_info["clock_rate"]
        vec[6] = self._gpu_info["mem_clock_rate"]
        self._data = torch.tensor(vec, dtype=torch.float)[None, :]
        return self._data

    @staticmethod
    def dimensions():
        return GPUEncoding.NUM_FEATURES
