# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import platform
import numpy as np

from daisytuner.profiling.metrics.metric import Metric


class BranchMispredictionRatio(Metric):
    def __init__(
        self, sdfg: dace.SDFG, hostname: str = platform.node(), cache=None
    ) -> None:
        super().__init__(
            sdfg,
            [
                "BR_MIS_PRED",
                "BR_PRED",
            ],
            dace.DeviceType.CPU,
            hostname,
            "arm64fx",
            cache=cache,
        )

    def compute(self) -> float:
        counters = self.values()

        volume_misp = 0.0
        volume_pred = 0.0
        for state in self._sdfg.states():
            if state not in counters["BR_MIS_PRED"]:
                continue

            volume_misp += sum(
                [
                    measurements[0]
                    for thread_id, measurements in counters["BR_MIS_PRED"][
                        state
                    ].items()
                ]
            )
            volume_pred += sum(
                [
                    measurements[0]
                    for thread_id, measurements in counters["BR_PRED"][state].items()
                ]
            )

        metric = volume_misp / (volume_pred + volume_misp)
        return metric

    def compute_per_thread(self) -> np.ndarray:
        counters = self.values()

        volume_misp = []
        volume_pred = []
        for state in self._sdfg.states():
            if state not in counters["BR_MIS_PRED"]:
                continue

            volume_misp.append(
                np.array(
                    [
                        measurements[0]
                        for thread_id, measurements in counters["BR_MIS_PRED"][
                            state
                        ].items()
                    ]
                )
            )
            volume_pred.append(
                np.array(
                    [
                        measurements[0]
                        for thread_id, measurements in counters["BR_PRED"][
                            state
                        ].items()
                    ]
                )
            )

        volume_misp = np.vstack(volume_misp).sum(axis=0, keepdims=False)
        volume_pred = np.vstack(volume_pred).sum(axis=0, keepdims=False)

        metric = volume_misp / (volume_pred + volume_misp)
        return metric
