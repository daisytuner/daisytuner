# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import pandas as pd

from typing import List

from daisytuner.profiling.metrics.metrics import Metrics


class Zen3Metrics(Metrics):
    def __init__(self, groups: List[str]) -> None:
        super().__init__(arch="zen3", groups=groups)

    def compute(self, counters: pd.DataFrame) -> pd.DataFrame:
        cs = counters.groupby("REPETITION").sum()
        cs = cs.median()
        runtime = counters.groupby("REPETITION").max()
        runtime = runtime.median()["TIME"]
        cycles = counters.groupby("REPETITION").max()
        cycles = cycles.median()["CPU_CLOCKS_UNHALTED"]
        cs_first = counters[counters["REPETITION"] == 0.0]
        cs_first = cs_first.sum()
        runtime_first = counters[counters["REPETITION"] == 0.0].max()
        runtime_first = runtime_first["TIME"]

        metrics = {}
        metrics["runtime"] = runtime
        metrics["runtime_0"] = runtime_first
        metrics["instructions_per_cycle"] = cs["RETIRED_INSTRUCTIONS"] / cycles

        # BRANCH
        if "BRANCH" in self._groups:
            metrics["branch_rate"] = (
                cs["RETIRED_BRANCH_INSTR"] / cs["RETIRED_INSTRUCTIONS"]
            )
            metrics["branch_misprediction_rate"] = (
                cs["RETIRED_MISP_BRANCH_INSTR"] / cs["RETIRED_INSTRUCTIONS"]
            )
            metrics["branch_misprediction_ratio"] = (
                cs["RETIRED_MISP_BRANCH_INSTR"] / cs["RETIRED_BRANCH_INSTR"]
            )
            metrics["instructions_per_branch"] = (
                cs["RETIRED_INSTRUCTIONS"] / cs["RETIRED_BRANCH_INSTR"]
            )

        if "DATA" in self._groups:
            metrics["load_to_store_ratio"] = (
                cs["LS_DISPATCH_LOADS"] + cs["LS_DISPATCH_LOAD_OP_STORES"]
            ) / (cs["LS_DISPATCH_STORES"] + cs["LS_DISPATCH_LOAD_OP_STORES"])

        if "CACHE" in self._groups:
            metrics["l2_request_rate"] = (
                cs["DATA_CACHE_ACCESSES"] / cs["RETIRED_INSTRUCTIONS"]
            )
            metrics["l2_miss_rate"] = (
                cs["DATA_CACHE_REFILLS_ALL"] / cs["RETIRED_INSTRUCTIONS"]
            )
            metrics["l2_miss_ratio"] = (
                cs["DATA_CACHE_REFILLS_ALL"] / cs["DATA_CACHE_ACCESSES"]
            )

        if "L2" in self._groups:
            metrics["l2_load_bandwidth"] = (
                1.0e-06 * cs["REQUESTS_TO_L2_GRP1_ALL_NO_PF"] * 64.0 / runtime
            )
            metrics["l2_load_data_volume"] = (
                1.0e-06 * cs["REQUESTS_TO_L2_GRP1_ALL_NO_PF"] * 64.0
            )
            metrics["l2_data_volume"] = (
                1.0e-06 * cs["REQUESTS_TO_L2_GRP1_ALL_NO_PF"] * 64.0
            )
            metrics["l2_bandwidth"] = metrics["l2_data_volume"] / runtime
            metrics["l2_data_volume_0"] = (
                1.0e-06 * cs_first["REQUESTS_TO_L2_GRP1_ALL_NO_PF"] * 64.0
            )
            metrics["l2_bandwidth_0"] = metrics["l2_data_volume_0"] / runtime_first

        if "L3" in self._groups:
            metrics["l3_load_data_volume"] = (
                1.0e-06
                * (
                    cs["L2_PF_HIT_IN_L3"]
                    + cs["L2_PF_MISS_IN_L3"]
                    + cs["L2_CACHE_MISS_AFTER_L1_MISS"]
                )
                * 64
            )
            metrics["l3_load_bandwidth"] = metrics["l3_load_data_volume"] / runtime
            metrics["l3_data_volume"] = (
                1.0e-06
                * (
                    cs["L2_PF_HIT_IN_L3"]
                    + cs["L2_PF_MISS_IN_L3"]
                    + cs["L2_CACHE_MISS_AFTER_L1_MISS"]
                )
                * 64
            )
            metrics["l3_bandwidth"] = metrics["l3_data_volume"] / runtime
            metrics["l3_data_volume_0"] = (
                1.0e-06
                * (
                    cs_first["L2_PF_HIT_IN_L3"]
                    + cs_first["L2_PF_MISS_IN_L3"]
                    + cs_first["L2_CACHE_MISS_AFTER_L1_MISS"]
                )
                * 64
            )
            metrics["l3_bandwidth_0"] = metrics["l3_data_volume_0"] / runtime_first

        if "L3CACHE" in self._groups:
            metrics["l3_request_rate"] = cs["L3_CACHE_REQ"] / cs["RETIRED_INSTRUCTIONS"]
            metrics["l3_miss_rate"] = (
                cs["L3_CACHE_REQ_MISS"] / cs["RETIRED_INSTRUCTIONS"]
            )
            metrics["l3_miss_ratio"] = cs["L3_CACHE_REQ_MISS"] / cs["L3_CACHE_REQ"]

        if "MEM1" in self._groups and "MEM2" in self._groups:
            metrics["memory_data_volume"] = (
                1.0e-06
                * (
                    cs["DRAM_CHANNEL_0"]
                    + cs["DRAM_CHANNEL_1"]
                    + cs["DRAM_CHANNEL_2"]
                    + cs["DRAM_CHANNEL_3"]
                    + cs["DRAM_CHANNEL_4"]
                    + cs["DRAM_CHANNEL_5"]
                    + cs["DRAM_CHANNEL_6"]
                    + cs["DRAM_CHANNEL_7"]
                )
                * 64.0
            )
            metrics["memory_bandwidth"] = metrics["memory_data_volume"] / runtime
            metrics["memory_data_volume_0"] = (
                1.0e-06
                * (
                    cs_first["DRAM_CHANNEL_0"]
                    + cs_first["DRAM_CHANNEL_1"]
                    + cs_first["DRAM_CHANNEL_2"]
                    + cs_first["DRAM_CHANNEL_3"]
                    + cs_first["DRAM_CHANNEL_4"]
                    + cs_first["DRAM_CHANNEL_5"]
                    + cs_first["DRAM_CHANNEL_6"]
                    + cs_first["DRAM_CHANNEL_7"]
                )
                * 64.0
            )
            metrics["memory_bandwidth_0"] = (
                metrics["memory_data_volume_0"] / runtime_first
            )

        if "FLOPS_SP" in self._groups:
            metrics["mflop_sp"] = 1.0e-06 * cs["RETIRED_SSE_AVX_FLOPS_ALL"]
            metrics["mflops_sp"] = metrics["mflop_sp"] / runtime
            metrics["mflops_sp_0"] = metrics["mflop_sp"] / runtime_first

        if "FLOPS_DP" in self._groups:
            metrics["mflop_dp"] = 1.0e-06 * cs["RETIRED_SSE_AVX_FLOPS_ALL"]
            metrics["mflops_dp"] = metrics["mflop_dp"] / runtime
            metrics["mflops_dp_0"] = metrics["mflop_dp"] / runtime

        if (
            "MEM1" in self._groups
            and "MEM2" in self._groups
            and "FLOPS_SP" in self._groups
            and "FLOPS_DP" in self._groups
        ):
            metrics["operational_intensity"] = (
                metrics["mflop_sp"] + metrics["mflop_dp"]
            ) / metrics["memory_data_volume_0"]

        return pd.DataFrame.from_dict(data=metrics, orient="index").T
