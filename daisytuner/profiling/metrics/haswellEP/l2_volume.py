# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import numpy as np
import dace
import platform


from daisytuner.profiling.metrics.metric import Metric


class L2Volume(Metric):
    def __init__(
        self, sdfg: dace.SDFG, hostname: str = platform.node(), cache=None
    ) -> None:
        super().__init__(
            sdfg,
            [
                "L2_TRANS_L1D_WB",
                "L1D_REPLACEMENT",
                "ICACHE_MISSES",
            ],
            dace.DeviceType.CPU,
            hostname,
            "haswellEP",
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
                        for thread_id, measurements in counters["L2_TRANS_L1D_WB"][
                            state
                        ].items()
                    ]
                )
                * 64
            )
            volume += (
                sum(
                    [
                        measurements[0]
                        for thread_id, measurements in counters["L1D_REPLACEMENT"][
                            state
                        ].items()
                    ]
                )
                * 64
            )
            volume += (
                sum(
                    [
                        measurements[0]
                        for thread_id, measurements in counters["ICACHE_MISSES"][
                            state
                        ].items()
                    ]
                )
                * 64
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
                        for thread_id, measurements in counters["L2_TRANS_L1D_WB"][
                            state
                        ].items()
                    ]
                )
                * 64
            )
            volume.append(
                np.array(
                    [
                        measurements[0]
                        for thread_id, measurements in counters["L1D_REPLACEMENT"][
                            state
                        ].items()
                    ]
                )
                * 64
            )
            volume.append(
                np.array(
                    [
                        measurements[0]
                        for thread_id, measurements in counters["ICACHE_MISSES"][
                            state
                        ].items()
                    ]
                )
                * 64
            )

        metric = 1e-6 * np.vstack(volume).sum(axis=0, keepdims=False)
        return metric
