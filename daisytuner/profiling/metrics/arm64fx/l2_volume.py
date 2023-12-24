# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import platform
import numpy as np

from daisytuner.profiling.metrics.metric import Metric


class L2Volume(Metric):
    def __init__(
        self, sdfg: dace.SDFG, hostname: str = platform.node(), cache=None
    ) -> None:
        super().__init__(
            sdfg,
            [
                "L1D_CACHE_REFILL",
                "L1D_CACHE_WB",
                "L1I_CACHE_REFILL",
            ],
            dace.DeviceType.CPU,
            hostname,
            "arm64fx",
            cache=cache,
        )

    def compute(self) -> float:
        counters = self.values()

        volume = 0.0
        for state in self._sdfg.states():
            volume += (
                sum(
                    [
                        measurements[0]
                        for thread_id, measurements in counters["L1D_CACHE_REFILL"][
                            state
                        ].items()
                    ]
                )
                * 256
            )
            volume += (
                sum(
                    [
                        measurements[0]
                        for thread_id, measurements in counters["L1D_CACHE_WB"][
                            state
                        ].items()
                    ]
                )
                * 256
            )
            volume += (
                sum(
                    [
                        measurements[0]
                        for thread_id, measurements in counters["L1I_CACHE_REFILL"][
                            state
                        ].items()
                    ]
                )
                * 256
            )

        metric = 1e-6 * volume
        return metric

    def compute_per_thread(self) -> np.ndarray:
        counters = self.values()

        volume = []
        for state in self._sdfg.states():
            volume.append(
                np.array(
                    [
                        measurements[0]
                        for thread_id, measurements in counters["L1D_CACHE_REFILL"][
                            state
                        ].items()
                    ]
                )
                * 256
            )
            volume.append(
                np.array(
                    [
                        measurements[0]
                        for thread_id, measurements in counters["L1D_CACHE_WB"][
                            state
                        ].items()
                    ]
                )
                * 256
            )
            volume.append(
                np.array(
                    [
                        measurements[0]
                        for thread_id, measurements in counters["L1I_CACHE_REFILL"][
                            state
                        ].items()
                    ]
                )
                * 256
            )

        metric = 1e-6 * np.vstack(volume).sum(axis=0, keepdims=False)
        return metric
