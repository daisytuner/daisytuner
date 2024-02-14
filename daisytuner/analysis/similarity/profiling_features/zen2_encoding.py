# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import numpy as np
import pandas as pd

from daisytuner.analysis.similarity.profiling_encoding import ProfilingEncoding


class Zen2Encoding(ProfilingEncoding):
    def _vectorize(self, data: pd.DataFrame) -> np.ndarray:
        # min, max, sum, mean, std
        num_statistics = 4
        num_counters = 16

        encoding = np.zeros((num_statistics * num_counters,))

        # Total instructions
        encoding[
            0 * num_statistics : 1 * num_statistics
        ] = ProfilingEncoding._normalize(data, "RETIRED_INSTRUCTIONS")

        # FLOPS_SP
        encoding[
            1 * num_statistics : 2 * num_statistics
        ] = ProfilingEncoding._normalize(data, "RETIRED_SSE_AVX_FLOPS_ALL")
        encoding[2 * num_statistics : 5 * num_statistics] = 0.0

        # FLOPS_DP
        encoding[
            5 * num_statistics : 6 * num_statistics
        ] = ProfilingEncoding._normalize(data, "RETIRED_SSE_AVX_FLOPS_ALL")
        encoding[6 * num_statistics : 9 * num_statistics] = 0.0

        # BRANCH
        encoding[
            9 * num_statistics : 10 * num_statistics
        ] = ProfilingEncoding._normalize(data, "RETIRED_BRANCH_INSTR")
        encoding[
            10 * num_statistics : 11 * num_statistics
        ] = ProfilingEncoding._normalize(data, "RETIRED_MISP_BRANCH_INSTR")

        # MEM Volume
        encoding[
            11 * num_statistics : 12 * num_statistics
        ] = ProfilingEncoding._normalize(
            data, "DRAM_CHANNEL_0"
        ) + ProfilingEncoding._normalize(
            data, "DRAM_CHANNEL_1"
        )

        # L3 Volume
        encoding[
            12 * num_statistics : 13 * num_statistics
        ] = ProfilingEncoding._normalize(data, "L3_ACCESS")

        # L2 Volume
        encoding[
            13 * num_statistics : 14 * num_statistics
        ] = ProfilingEncoding._normalize(data, "REQUESTS_TO_L2_GRP1_ALL_NO_PF")

        # DRAM Controller
        encoding[
            14 * num_statistics : 15 * num_statistics
        ] = ProfilingEncoding._normalize(data, "LS_DISPATCH_LOADS")
        encoding[
            15 * num_statistics : 16 * num_statistics
        ] = ProfilingEncoding._normalize(data, "LS_DISPATCH_STORES")

        return encoding
