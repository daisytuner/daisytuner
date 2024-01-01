# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import platform
import numpy as np

from daisytuner.profiling.metrics.metric import Metric


class FLOP(Metric):
    def __init__(
        self, sdfg: dace.SDFG, hostname: str = platform.node(), cache=None
    ) -> None:
        super().__init__(
            sdfg,
            [
                "FP_FIXED_OPS_SPEC",
                "FP_SCALE_OPS_SPEC",
            ],
            dace.DeviceType.CPU,
            hostname,
            "nvidia_grace",
            cache=cache,
        )

    def compute(self) -> float:
        counters = self.values()

        volume = 0.0
        for state in self._sdfg.states():
            volume += sum(
                [
                    measurements[0]
                    for thread_id, measurements in counters["FP_FIXED_OPS_SPEC"][
                        state
                    ].items()
                ]
            )
            volume += sum(
                [
                    measurements[0]
                    for thread_id, measurements in counters["FP_SCALE_OPS_SPEC"][
                        state
                    ].items()
                ]
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
                        for thread_id, measurements in counters["FP_FIXED_OPS_SPEC"][
                            state
                        ].items()
                    ]
                )
            )
            volume.append(
                np.array(
                    [
                        measurements[0]
                        for thread_id, measurements in counters["FP_SCALE_OPS_SPEC"][
                            state
                        ].items()
                    ]
                )
            )

        metric = 1e-6 * np.vstack(volume).sum(axis=0, keepdims=False)
        return metric
