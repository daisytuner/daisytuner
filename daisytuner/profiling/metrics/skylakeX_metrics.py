# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import pandas as pd

from typing import List

from daisytuner.profiling.metrics.metrics import Metrics


class SkylakeXMetrics(Metrics):
    def __init__(self, groups: List[str]) -> None:
        super().__init__(arch="skylakeX", groups=groups)

    def compute(self, counters: pd.DataFrame) -> pd.DataFrame:
        # Sum over all threads
        cs = counters.groupby("REPETITION").sum()

        # Max/Median over repetitions
        cs = cs.median()
        runtime = counters.groupby("REPETITION").max()
        runtime = runtime.median()["TIME"]
        cycles = counters.groupby("REPETITION").max()
        cycles = cycles.median()["CPU_CLK_UNHALTED_CORE"]

        # First repetition stats
        cs_first = counters[counters["REPETITION"] == 0.0]
        cs_first = cs_first.sum()
        runtime_first = counters[counters["REPETITION"] == 0.0].max()
        runtime_first = runtime_first["TIME"]

        metrics = {}
        metrics["runtime"] = runtime
        metrics["runtime_0"] = runtime_first
        metrics["instructions_per_cycle"] = cs["INSTR_RETIRED_ANY"] / cycles

        # BRANCH
        if "BRANCH" in self._groups:
            metrics["branch_rate"] = (
                cs["BR_INST_RETIRED_ALL_BRANCHES"] / cs["INSTR_RETIRED_ANY"]
            )
            metrics["branch_misprediction_rate"] = (
                cs["BR_MISP_RETIRED_ALL_BRANCHES"] / cs["INSTR_RETIRED_ANY"]
            )
            metrics["branch_misprediction_ratio"] = (
                cs["BR_MISP_RETIRED_ALL_BRANCHES"] / cs["BR_INST_RETIRED_ALL_BRANCHES"]
            )
            metrics["instructions_per_branch"] = (
                cs["INSTR_RETIRED_ANY"] / cs["BR_INST_RETIRED_ALL_BRANCHES"]
            )

        # DATA
        if "DATA" in self._groups:
            metrics["load_to_store_ratio"] = (
                cs["MEM_INST_RETIRED_ALL_LOADS"] / cs["MEM_INST_RETIRED_ALL_STORES"]
            )

        # FLOPS_DP
        if "FLOPS_DP" in self._groups:
            metrics["mflop_dp"] = 1.0e-06 * (
                cs["FP_ARITH_INST_RETIRED_128B_PACKED_DOUBLE"] * 2.0
                + cs["FP_ARITH_INST_RETIRED_SCALAR_DOUBLE"]
                + cs["FP_ARITH_INST_RETIRED_256B_PACKED_DOUBLE"] * 4.0
                + cs["FP_ARITH_INST_RETIRED_512B_PACKED_DOUBLE"] * 8.0
            )
            metrics["mflops_dp"] = metrics["mflop_dp"] / runtime
            metrics["mflops_dp_0"] = metrics["mflop_dp"] / runtime_first
            vec = (
                cs["FP_ARITH_INST_RETIRED_128B_PACKED_DOUBLE"]
                + cs["FP_ARITH_INST_RETIRED_256B_PACKED_DOUBLE"]
                + cs["FP_ARITH_INST_RETIRED_512B_PACKED_DOUBLE"]
            )
            if metrics["mflop_dp"] > 0.0:
                metrics["vectorization_ratio_dp"] = vec / (
                    vec + cs["FP_ARITH_INST_RETIRED_SCALAR_DOUBLE"]
                )
            else:
                metrics["vectorization_ratio_dp"] = 1.0

        # FLOPS_SP
        if "FLOPS_SP" in self._groups:
            metrics["mflop_sp"] = 1.0e-06 * (
                cs["FP_ARITH_INST_RETIRED_128B_PACKED_SINGLE"] * 4.0
                + cs["FP_ARITH_INST_RETIRED_SCALAR_SINGLE"]
                + cs["FP_ARITH_INST_RETIRED_256B_PACKED_SINGLE"] * 8.0
                + cs["FP_ARITH_INST_RETIRED_512B_PACKED_SINGLE"] * 16.0
            )
            metrics["mflops_sp"] = metrics["mflop_sp"] / runtime
            metrics["mflops_sp_0"] = metrics["mflop_sp"] / runtime_first
            vec = (
                cs["FP_ARITH_INST_RETIRED_128B_PACKED_SINGLE"]
                + cs["FP_ARITH_INST_RETIRED_256B_PACKED_SINGLE"]
                + cs["FP_ARITH_INST_RETIRED_512B_PACKED_SINGLE"]
            )
            if metrics["mflop_sp"] > 0.0:
                metrics["vectoriation_ratio_sp"] = vec / (
                    vec + cs["FP_ARITH_INST_RETIRED_SCALAR_SINGLE"]
                )
            else:
                metrics["vectoriation_ratio_sp"] = 1.0

        # L2
        if "L2" in self._groups:
            metrics["l2_load_volume"] = 1.0e-06 * cs["L1D_REPLACEMENT"] * 64.0
            metrics["l2_load_bandwidth"] = (
                1.0e-06 * cs["L1D_REPLACEMENT"] * 64.0 / runtime
            )
            metrics["l2_evict_volume"] = 1.0e-06 * cs["L1D_M_EVICT"] * 64.0
            metrics["l2_evict_bandwidth"] = 1.0e-06 * cs["L1D_M_EVICT"] * 64.0 / runtime
            metrics["l2_volume"] = (
                1.0e-06
                * (
                    cs["L1D_M_EVICT"]
                    + cs["L1D_REPLACEMENT"]
                    + cs["ICACHE_64B_IFTAG_MISS"]
                )
                * 64.0
            )
            metrics["l2_bandwidth"] = metrics["l2_volume"] / runtime
            metrics["l2_volume_0"] = (
                1.0e-06
                * (
                    cs_first["L1D_M_EVICT"]
                    + cs_first["L1D_REPLACEMENT"]
                    + cs_first["ICACHE_64B_IFTAG_MISS"]
                )
                * 64.0
            )
            metrics["l2_bandwidth_0"] = metrics["l2_volume_0"] / runtime_first

        # L2CACHE
        if "L2CACHE" in self._groups:
            metrics["l2_request_rate"] = (
                cs["L2_TRANS_ALL_REQUESTS"] / cs["INSTR_RETIRED_ANY"]
            )
            metrics["l2_miss_rate"] = cs["L2_RQSTS_MISS"] / cs["INSTR_RETIRED_ANY"]
            metrics["l2_miss_ratio"] = cs["L2_RQSTS_MISS"] / cs["L2_TRANS_ALL_REQUESTS"]

        # L3
        if "L3" in self._groups:
            metrics["l3_load_bandwidth"] = (
                1.0e-06 * cs["L2_LINES_IN_ALL"] * 64.0 / runtime
            )
            metrics["l3_load_data_volume"] = 1.0e-06 * cs["L2_LINES_IN_ALL"] * 64.0
            metrics["dropped_cls_bandwidth"] = (
                1.0e-06 * cs["IDI_MISC_WB_DOWNGRADE"] * 64.0 / runtime
            )
            metrics["dropped_cls_data_volume"] = (
                1.0e-06 * cs["IDI_MISC_WB_DOWNGRADE"] * 64.0
            )
            metrics["L3_mem_evict_bandwidth"] = (
                1.0e-06 * cs["L2_TRANS_L2_WB"] * 64.0 / runtime
            )
            metrics["l3_mem_evict_data_volume"] = 1.0e-06 * cs["L2_TRANS_L2_WB"] * 64.0
            metrics["l3_data_volume"] = (
                1.0e-06 * (cs["L2_LINES_IN_ALL"] + cs["L2_TRANS_L2_WB"]) * 64
            )
            metrics["l3_bandwidth"] = metrics["l3_data_volume"] / runtime
            metrics["l3_data_volume_0"] = (
                1.0e-06
                * (cs_first["L2_LINES_IN_ALL"] + cs_first["L2_TRANS_L2_WB"])
                * 64
            )
            metrics["l3_bandwidth_0"] = metrics["l3_data_volume_0"] / runtime_first

        # L3CACHE
        if "L3CACHE" in self._groups:
            metrics["l3_request_rate"] = (
                cs["MEM_LOAD_RETIRED_L3_HIT"] + cs["MEM_LOAD_RETIRED_L3_MISS"]
            ) / cs["UOPS_RETIRED_ALL"]
            metrics["l3_miss_rate"] = (
                cs["MEM_LOAD_RETIRED_L3_MISS"] / cs["UOPS_RETIRED_ALL"]
            )
            metrics["l3_miss_ratio"] = cs["MEM_LOAD_RETIRED_L3_MISS"] / (
                cs["MEM_LOAD_RETIRED_L3_HIT"] + cs["MEM_LOAD_RETIRED_L3_MISS"]
            )

        # MEM
        if "MEM" in self._groups:
            metrics["memory_read_bandwidth"] = (
                1.0e-06 * (cs["CAS_COUNT_RD"]) * 64.0 / runtime
            )
            metrics["memory_read_data_volume"] = 1.0e-06 * (cs["CAS_COUNT_RD"]) * 64.0
            metrics["memory_write_bandwidth"] = (
                1.0e-06 * (cs["CAS_COUNT_WR"]) * 64.0 / runtime
            )
            metrics["memory_write_data_volume"] = 1.0e-06 * (cs["CAS_COUNT_WR"]) * 64.0
            metrics["memory_data_volume"] = (
                1.0e-06 * (cs["CAS_COUNT_RD"] + cs["CAS_COUNT_WR"]) * 64.0
            )
            metrics["memory_bandwidth"] = metrics["memory_data_volume"] / runtime
            metrics["memory_data_volume_0"] = (
                1.0e-06 * (cs_first["CAS_COUNT_RD"] + cs_first["CAS_COUNT_WR"]) * 64.0
            )
            metrics["memory_bandwidth_0"] = (
                metrics["memory_data_volume_0"] / runtime_first
            )

        # Operations Intensity
        if (
            "MEM" in self._groups
            and "FLOPS_SP" in self._groups
            and "FLOPS_DP" in self._groups
        ):
            metrics["operational_intensity"] = (
                metrics["mflop_sp"] + metrics["mflop_dp"]
            ) / metrics["memory_data_volume_0"]

        return pd.DataFrame.from_dict(data=metrics, orient="index").T
