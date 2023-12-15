# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import pandas as pd

from typing import List

from daisytuner.profiling.metrics.metrics import Metrics


class NVIDIACCGE7Metrics(Metrics):
    def __init__(self, groups: List[str]) -> None:
        super().__init__(arch="nvidia_cc_ge_7", groups=groups)

    def compute(self, counters: pd.DataFrame) -> pd.DataFrame:
        cs = counters.groupby("REPETITION").sum()
        cs = cs.median()
        runtime = counters.groupby("REPETITION").max()
        runtime = runtime.median()["TIME"]
        cs_first = counters[counters["REPETITION"] == 0.0]
        cs_first = cs_first.sum()
        runtime_first = counters[counters["REPETITION"] == 0.0].max()
        runtime_first = runtime_first["TIME"]

        metrics = {}
        metrics["runtime"] = runtime
        metrics["runtime_0"] = runtime_first

        metrics["ipc"] = (
            cs.loc["SMSP_INST_EXECUTED_AVG"] / cs.loc["SMSP_CYCLES_ACTIVE_AVG"]
        )

        metrics["achieved_occupancy"] = (
            cs.loc["SMSP_WARPS_ACTIVE_AVG"] / cs.loc["SMSP_CYCLES_ACTIVE_AVG"]
        )
        metrics["warp_execution_efficiency"] = (
            cs.loc["SMSP_THREADS_LAUNCHED_AVG"] / cs.loc["SMSP_WARPS_ACTIVE_AVG"]
        )
        metrics["eligible_warps_per_cycle"] = (
            cs.loc["SMSP_WARPS_ELIGIBLE_AVG"] / cs.loc["SMSP_CYCLES_ACTIVE_AVG"]
        )

        metrics["branch_efficiency"] = cs.loc[
            "SMSP_SASS_AVERAGE_BRANCH_TARGETS_THREADS_UNIFORM_PCT"
        ]

        metrics["flop_dp_efficiency"] = (
            cs.loc["SMSP_SASS_THREAD_INST_EXECUTED_OP_DADD_PRED_ON_AVG"]
            + cs.loc["SMSP_SASS_THREAD_INST_EXECUTED_OP_DFMA_PRED_ON_AVG"]
            + cs.loc["SMSP_SASS_THREAD_INST_EXECUTED_OP_DMUL_PRED_ON_AVG"]
        ) / cs.loc["SMSP_CYCLES_ACTIVE_AVG"]
        metrics["flop_hp_efficiency"] = (
            cs.loc["SMSP_SASS_THREAD_INST_EXECUTED_OP_HADD_PRED_ON_AVG"]
            + cs.loc["SMSP_SASS_THREAD_INST_EXECUTED_OP_HFMA_PRED_ON_AVG"]
            + cs.loc["SMSP_SASS_THREAD_INST_EXECUTED_OP_HMUL_PRED_ON_AVG"]
        ) / cs.loc["SMSP_CYCLES_ACTIVE_AVG"]
        metrics["flop_sp_efficiency"] = (
            cs.loc["SMSP_SASS_THREAD_INST_EXECUTED_OP_FADD_PRED_ON_AVG"]
            + cs.loc["SMSP_SASS_THREAD_INST_EXECUTED_OP_FFMA_PRED_ON_AVG"]
            + cs.loc["SMSP_SASS_THREAD_INST_EXECUTED_OP_FMUL_PRED_ON_AVG"]
        ) / cs.loc["SMSP_CYCLES_ACTIVE_AVG"]

        metrics["dram_throughput"] = 1e-6 * cs.loc["DRAM_BYTES_SUM"] / runtime
        metrics["dram_read_throughput"] = 1e-6 * cs.loc["DRAM_BYTES_READ_SUM"] / runtime
        metrics["dram_write_throughput"] = (
            1e-6 * cs.loc["DRAM_BYTES_WRITE_SUM"] / runtime
        )

        metrics["global_hit_rate"] = cs.loc["L1TEX_T_SECTOR_HIT_RATE_PCT"] / 100
        metrics["gld_throughput"] = (
            1e-6 * cs.loc["L1TEX_T_BYTES_PIPE_LSU_MEM_GLOBAL_OP_LD_SUM"] / runtime
        )
        metrics["gst_throughput"] = (
            1e-6 * cs.loc["L1TEX_T_BYTES_PIPE_LSU_MEM_GLOBAL_OP_ST_SUM"] / runtime
        )

        metrics["l2_tex_hit_rate"] = cs.loc["LTS_T_SECTOR_HIT_RATE_MAX_RATE"]
        metrics["l2_tex_read_hit_rate"] = cs.loc[
            "LTS_T_SECTOR_OP_READ_HIT_RATE_MAX_RATE"
        ]
        metrics["l2_tex_write_hit_rate"] = cs.loc[
            "LTS_T_SECTOR_OP_WRITE_HIT_RATE_MAX_RATE"
        ]
        metrics["l2_tex_read_throughput"] = (
            1e-6 * cs.loc["LTS_T_SECTORS_SRCUNIT_TEX_OP_READ_SUM"] / runtime
        )
        metrics["l2_tex_write_throughput"] = (
            1e-6 * cs.loc["LTS_T_SECTORS_SRCUNIT_TEX_OP_WRITE_SUM"] / runtime
        )

        metrics["l2_throughput"] = 1e-6 * cs.loc["LTS_T_SECTORS_SUM"] / runtime
        metrics["l2_write_throughput"] = (
            1e-6 * cs.loc["LTS_T_SECTORS_OP_WRITE_SUM"] / runtime
        )
        metrics["l2_atomic_throughput"] = (
            1e-6 * cs.loc["LTS_T_SECTORS_OP_ATOM_SUM"] / runtime
        )
        metrics["l2_red_throughput"] = (
            1e-6 * cs.loc["LTS_T_SECTORS_OP_RED_SUM"] / runtime
        )

        metrics["local_hit_rate"] = cs.loc["LTS_T_SECTOR_OP_WRITE_HIT_RATE_MAX_RATE"]
        metrics["local_load_transactions"] = cs.loc[
            "L1TEX_T_SECTORS_PIPE_LSU_MEM_LOCAL_OP_LD_SUM"
        ] / (cs.loc["SMSP_INST_EXECUTED_AVG"] + 1)
        metrics["local_store_transactions"] = cs.loc[
            "L1TEX_T_SECTORS_PIPE_LSU_MEM_LOCAL_OP_ST_SUM"
        ] / (cs.loc["SMSP_INST_EXECUTED_AVG"] + 1)

        metrics["shared_efficiency"] = cs.loc[
            "SMSP_SASS_AVERAGE_DATA_BYTES_PER_WAVEFRONT_MEM_SHARED_PCT"
        ]
        metrics["shared_load_throughput"] = (
            1e-6 * cs.loc["L1TEX_DATA_PIPE_LSU_WAVEFRONTS_MEM_SHARED_SUM"] / runtime
        )
        metrics["shared_store_throughput"] = (
            1e-6 * cs.loc["LTS_T_SECTOR_OP_WRITE_HIT_RATE_MAX_RATE"] / runtime
        )

        metrics["sysmem_read_throughput"] = (
            1e-6 * cs.loc["LTS_T_SECTORS_APERTURE_SYSMEM_OP_READ_SUM"] / runtime
        )
        metrics["sysmem_write_throughput"] = (
            1e-6 * cs.loc["LTS_T_SECTORS_APERTURE_SYSMEM_OP_WRITE_SUM"] / runtime
        )

        return pd.DataFrame.from_dict(data=metrics, orient="index").T
