# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import numpy as np
import pandas as pd

from daisytuner.analysis.similarity.profiling_encoding import ProfilingEncoding


class NVIDIACCGE7Encoding(ProfilingEncoding):
    def _vectorize(self, data: pd.DataFrame) -> np.ndarray:
        # min, max, sum, mean, std
        num_statistics = 4
        num_counters = 17

        encoding = np.zeros((num_statistics * num_counters,))

        # Total instructions
        encoding[
            0 * num_statistics : 1 * num_statistics
        ] = ProfilingEncoding._normalize(data, "SMSP_INST_ISSUED_SUM")

        # FLOPS
        encoding[
            1 * num_statistics : 2 * num_statistics
        ] = ProfilingEncoding._normalize(
            data, "SMSP_SASS_THREAD_INST_EXECUTED_OP_DADD_PRED_ON_AVG"
        )
        encoding[
            2 * num_statistics : 3 * num_statistics
        ] = ProfilingEncoding._normalize(
            data, "SMSP_SASS_THREAD_INST_EXECUTED_OP_DFMA_PRED_ON_AVG"
        )
        encoding[
            3 * num_statistics : 4 * num_statistics
        ] = ProfilingEncoding._normalize(
            data, "SMSP_SASS_THREAD_INST_EXECUTED_OP_DMUL_PRED_ON_AVG"
        )
        encoding[
            4 * num_statistics : 5 * num_statistics
        ] = ProfilingEncoding._normalize(
            data, "SMSP_SASS_THREAD_INST_EXECUTED_OP_HADD_PRED_ON_AVG"
        )
        encoding[
            5 * num_statistics : 6 * num_statistics
        ] = ProfilingEncoding._normalize(
            data, "SMSP_SASS_THREAD_INST_EXECUTED_OP_HFMA_PRED_ON_AVG"
        )
        encoding[
            6 * num_statistics : 7 * num_statistics
        ] = ProfilingEncoding._normalize(
            data, "SMSP_SASS_THREAD_INST_EXECUTED_OP_HMUL_PRED_ON_AVG"
        )
        encoding[
            7 * num_statistics : 8 * num_statistics
        ] = ProfilingEncoding._normalize(
            data, "SMSP_SASS_THREAD_INST_EXECUTED_OP_FADD_PRED_ON_AVG"
        )
        encoding[
            8 * num_statistics : 9 * num_statistics
        ] = ProfilingEncoding._normalize(
            data, "SMSP_SASS_THREAD_INST_EXECUTED_OP_FFMA_PRED_ON_AVG"
        )
        encoding[
            9 * num_statistics : 10 * num_statistics
        ] = ProfilingEncoding._normalize(
            data, "SMSP_SASS_THREAD_INST_EXECUTED_OP_FMUL_PRED_ON_AVG"
        )

        # BRANCH
        encoding[
            10 * num_statistics : 11 * num_statistics
        ] = ProfilingEncoding._normalize(
            data, "SMSP_SASS_AVERAGE_BRANCH_TARGETS_THREADS_UNIFORM_RATIO"
        )

        # DRAM
        encoding[
            11 * num_statistics : 12 * num_statistics
        ] = 1e-6 * ProfilingEncoding._normalize(data, "DRAM_BYTES_SUM")
        encoding[
            12 * num_statistics : 13 * num_statistics
        ] = 1e-6 * ProfilingEncoding._normalize(data, "DRAM_BYTES_READ_SUM")
        encoding[
            13 * num_statistics : 14 * num_statistics
        ] = 1e-6 * ProfilingEncoding._normalize(data, "DRAM_BYTES_WRITE_SUM")

        # G
        encoding[
            14 * num_statistics : 15 * num_statistics
        ] = 1e-6 * ProfilingEncoding._normalize(
            data, "L1TEX_T_BYTES_PIPE_LSU_MEM_GLOBAL_OP_LD_SUM"
        )
        encoding[
            15 * num_statistics : 16 * num_statistics
        ] = 1e-6 * ProfilingEncoding._normalize(
            data, "L1TEX_T_BYTES_PIPE_LSU_MEM_GLOBAL_OP_ST_SUM"
        )

        # L2
        encoding[
            16 * num_statistics : 17 * num_statistics
        ] = 1e-6 * ProfilingEncoding._normalize(data, "LTS_T_SECTORS_SUM")

        return encoding
