# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import platform
import torch
import numpy as np

from pathlib import Path

from abc import ABC, abstractmethod

from daisytuner.analysis.similarity.profiling_features.targets import TARGET_GROUPS
from daisytuner.profiling.likwid_helpers import cpu_codename, gpu_codename


class ProfilingEncoding(ABC):
    def __init__(
        self,
        sdfg: dace.SDFG,
        device: dace.DeviceType,
        hostname: str,
        codename: str,
        cache_path: Path = None,
    ) -> None:
        self._sdfg = sdfg
        self._device = device
        self._hostname = hostname
        self._codename = codename
        self._cache_path = cache_path

        self._encoding = None

    def encode(self) -> torch.tensor:
        if self._encoding is not None:
            return self._encoding

        # Gather instrumentation data
        from daisytuner.profiling.profiling import Profiling

        instrumentation = Profiling(
            sdfg=self._sdfg,
            device=self._device,
            groups=TARGET_GROUPS[self._codename],
            cache_path=self._cache_path,
            hostname=self._hostname,
            codename=self._codename,
        )
        data = instrumentation.analyze()

        # Compute statistics over threads; median over repetitions
        data = data.groupby("REPETITION").agg(["min", "max", "sum", "mean"])
        data = data.median()
        data = self._vectorize(data)

        self._encoding = torch.tensor(data, dtype=torch.float)[None, :]
        return self._encoding

    @abstractmethod
    def _vectorize(self, data) -> np.ndarray:
        pass

    @staticmethod
    def _normalize(counters, name) -> np.ndarray:
        stats = np.zeros(4)
        for i, stat in enumerate(["min", "max", "sum", "mean"]):
            stats[i] = counters[name][stat]

        return stats

    @staticmethod
    def create(
        sdfg: dace.SDFG,
        device: dace.DeviceType = dace.DeviceType.CPU,
        cache_path: Path = None,
        hostname: str = None,
        codename: str = None,
    ):
        if hostname is not None:
            assert (
                codename is not None
            ), "Architecture codename must be provded when hostname is different to current host, e.g., broadwellEP"
        else:
            hostname = platform.node()
            codename = (
                cpu_codename() if device == dace.DeviceType.CPU else gpu_codename()
            )

        if codename == "broadwellEP":
            from daisytuner.analysis.similarity.profiling_features.broadwellEP_encoding import (
                BroadwellEPEncoding,
            )

            return BroadwellEPEncoding(
                sdfg=sdfg,
                device=device,
                cache_path=cache_path,
                hostname=hostname,
                codename=codename,
            )
        elif codename == "haswellEP":
            from daisytuner.analysis.similarity.profiling_features.haswellEP_encoding import (
                HaswellEPEncoding,
            )

            return HaswellEPEncoding(
                sdfg=sdfg,
                device=device,
                cache_path=cache_path,
                hostname=hostname,
                codename=codename,
            )
        elif codename == "skylakeX":
            from daisytuner.analysis.similarity.profiling_features.skylakeX_encoding import (
                SkylakeXEncoding,
            )

            return SkylakeXEncoding(
                sdfg=sdfg,
                device=device,
                cache_path=cache_path,
                hostname=hostname,
                codename=codename,
            )
        elif codename == "zen":
            from daisytuner.analysis.similarity.profiling_features.zen_encoding import (
                ZenEncoding,
            )

            return ZenEncoding(
                sdfg=sdfg,
                device=device,
                cache_path=cache_path,
                hostname=hostname,
                codename=codename,
            )
        elif codename == "zen2":
            from daisytuner.analysis.similarity.profiling_features.zen2_encoding import (
                Zen2Encoding,
            )

            return Zen2Encoding(
                sdfg=sdfg,
                device=device,
                cache_path=cache_path,
                hostname=hostname,
                codename=codename,
            )
        elif codename == "zen3":
            from daisytuner.analysis.similarity.profiling_features.zen3_encoding import (
                Zen3Encoding,
            )

            return Zen3Encoding(
                sdfg=sdfg,
                device=device,
                cache_path=cache_path,
                hostname=hostname,
                codename=codename,
            )
        elif codename == "nvidia_cc_ge_7":
            from daisytuner.analysis.similarity.profiling_features.nvidia_cc_ge_7_encoding import (
                NVIDIACCGE7Encoding,
            )

            return NVIDIACCGE7Encoding(
                sdfg=sdfg,
                device=device,
                cache_path=cache_path,
                hostname=hostname,
                codename=codename,
            )
        else:
            raise ValueError(f"Profiling features on {codename} are not supported")
