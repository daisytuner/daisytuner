# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import numpy as np
import pandas as pd

from daisytuner.analysis.similarity.profiling_encoding import ProfilingEncoding


class SkylakeEncoding(ProfilingEncoding):
    def _vectorize(self, data: pd.DataFrame) -> np.ndarray:
        # min, max, sum, mean, std
        num_statistics = 4
        num_counters = 16

        encoding = np.zeros((num_statistics * num_counters,))

        # INSTRUCTIONS
        encoding[
            0 * num_statistics : 1 * num_statistics
        ] = ProfilingEncoding._normalize(data, "INSTR_RETIRED_ANY")

        # FLOPS_SP
        encoding[
            1 * num_statistics : 2 * num_statistics
        ] = ProfilingEncoding._normalize(data, "FP_ARITH_INST_RETIRED_SCALAR_SINGLE")
        encoding[
            2 * num_statistics : 3 * num_statistics
        ] = ProfilingEncoding._normalize(
            data,
            "FP_ARITH_INST_RETIRED_128B_PACKED_SINGLE",
        )
        encoding[
            3 * num_statistics : 4 * num_statistics
        ] = ProfilingEncoding._normalize(
            data,
            "FP_ARITH_INST_RETIRED_256B_PACKED_SINGLE",
        )
        encoding[4 * num_statistics : 5 * num_statistics] = 0.0

        # FLOPS_DP
        encoding[
            5 * num_statistics : 6 * num_statistics
        ] = ProfilingEncoding._normalize(data, "FP_ARITH_INST_RETIRED_SCALAR_DOUBLE")
        encoding[
            6 * num_statistics : 7 * num_statistics
        ] = ProfilingEncoding._normalize(
            data,
            "FP_ARITH_INST_RETIRED_128B_PACKED_DOUBLE",
        )
        encoding[
            7 * num_statistics : 8 * num_statistics
        ] = ProfilingEncoding._normalize(
            data,
            "FP_ARITH_INST_RETIRED_256B_PACKED_DOUBLE",
        )
        encoding[8 * num_statistics : 9 * num_statistics] = 0.0

        # BRANCH
        encoding[
            9 * num_statistics : 10 * num_statistics
        ] = ProfilingEncoding._normalize(data, "BR_INST_RETIRED_ALL_BRANCHES")
        encoding[
            10 * num_statistics : 11 * num_statistics
        ] = ProfilingEncoding._normalize(data, "BR_MISP_RETIRED_ALL_BRANCHES")

        # MEM Volume
        encoding[
            11 * num_statistics : 12 * num_statistics
        ] = ProfilingEncoding._normalize(
            data, "DRAM_READS"
        ) + ProfilingEncoding._normalize(
            data, "DRAM_WRITES"
        )

        # L3 Volume
        encoding[
            12 * num_statistics : 13 * num_statistics
        ] = ProfilingEncoding._normalize(
            data, "L2_LINES_IN_ALL"
        ) + ProfilingEncoding._normalize(
            data, "L2_TRANS_L2_WB"
        )

        # L2 Volume
        encoding[
            13 * num_statistics : 14 * num_statistics
        ] = ProfilingEncoding._normalize(
            data, "L1D_REPLACEMENT"
        ) + ProfilingEncoding._normalize(
            data, "L1D_M_EVICT"
        )

        # DRAM controller
        encoding[
            14 * num_statistics : 15 * num_statistics
        ] = ProfilingEncoding._normalize(data, "MEM_INST_RETIRED_ALL_LOADS")
        encoding[
            15 * num_statistics : 16 * num_statistics
        ] = ProfilingEncoding._normalize(data, "MEM_INST_RETIRED_ALL_STORES")

        return encoding
