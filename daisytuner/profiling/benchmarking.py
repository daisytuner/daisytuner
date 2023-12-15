# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import os
import re
import json
import platform
import subprocess

from tqdm import tqdm
from pathlib import Path

from daisytuner.profiling.likwid_helpers import cpu_codename, cpu_topology


class Benchmarking:
    def __init__(self) -> None:
        assert platform.system() in [
            "Linux"
        ], f"Profiling is not supported {platform.system()}"

        self._hostname = platform.node()
        if "DAISY_GLOBAL_CACHE" in os.environ:
            self._cache_path = (
                Path(os.environ["DAISY_GLOBAL_CACHE"]) / f"{self._hostname}.json"
            )
        else:
            self._cache_path = Path.home() / ".daisy" / f"{self._hostname}.json"

    def analyze(self):
        if self._cache_path.is_file():
            with open(self._cache_path, "r") as handle:
                metrics = json.load(handle)
                return metrics

        arch = cpu_codename()
        topo = cpu_topology()

        metrics = {}
        metrics["arch"] = arch
        metrics["num_sockets"] = topo["numSockets"]
        metrics["cores_per_socket"] = topo["numCoresPerSocket"]
        metrics["threads_per_core"] = topo["numThreadsPerCore"]
        metrics["l2_cache"] = int(topo["cacheLevels"][2]["size"] / 1000)
        metrics["l3_cache"] = int(topo["cacheLevels"][3]["size"] / 1000)

        num_cpus = (
            metrics["threads_per_core"]
            * metrics["cores_per_socket"]
            * metrics["num_sockets"]
        )

        print("Executing STREAM benchmarks")
        metrics.update(Benchmarking._stream_benchmark(num_cpus))

        print("Executing peakflops benchmarks")
        metrics.update(Benchmarking._peakflops_benchmark(num_cpus))

        with open(self._cache_path, "w") as handle:
            json.dump(metrics, handle)

        return metrics

    @classmethod
    def _stream_benchmark(cls, num_cores: int):
        stream = {}
        for test in tqdm(["load", "store", "copy", "triad"]):
            process = subprocess.Popen(
                ["likwid-bench", f"-t{test}", f"-WN:2GB:{num_cores}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            stdout, stderr = process.communicate()
            res = re.findall(r"MByte/s:\t\t\d+\.\d+", stdout)
            if not res:
                raise ValueError(stderr)
            stream[f"stream_{test}"] = float(re.findall(r"\d+\.\d+", res[0])[0])

        return stream

    @classmethod
    def _peakflops_benchmark(cls, num_cores: int):
        peakflops = {}
        for name, test in tqdm(
            [("peakflops", "peakflops"), ("peakflops_avx", "peakflops_avx_fma")]
        ):
            process = subprocess.Popen(
                ["likwid-bench", f"-t{test}", f"-WN:360kB:{num_cores}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            stdout, stderr = process.communicate()
            res = re.findall(r"MFlops/s:\t\t\d+\.\d+", stdout)
            if not res:
                raise ValueError(stderr)
            peakflops[name] = float(re.findall(r"\d+\.\d+", res[0])[0])

        return peakflops
