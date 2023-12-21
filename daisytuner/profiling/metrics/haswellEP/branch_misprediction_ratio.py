# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import numpy as np
import dace
import platform


from daisytuner.profiling.metrics.metric import Metric


class BranchMispredictionRatio(Metric):
    def __init__(
        self, sdfg: dace.SDFG, hostname: str = platform.node(), cache=None
    ) -> None:
        super().__init__(
            sdfg,
            [
                "BR_MISP_RETIRED_ALL_BRANCHES",
                "BR_INST_RETIRED_ALL_BRANCHES",
            ],
            dace.DeviceType.CPU,
            hostname,
            "haswellEP",
            cache=cache,
        )

    def compute(self) -> float:
        counters = self.values()

        volume_misp = 0.0
        volume_branch = 0.0
        for state in self._sdfg.states():
            if state not in counters["BR_MISP_RETIRED_ALL_BRANCHES"]:
                continue

            volume_misp += sum(
                [
                    measurements[0]
                    for thread_id, measurements in counters[
                        "BR_MISP_RETIRED_ALL_BRANCHES"
                    ][state].items()
                ]
            )
            volume_branch += sum(
                [
                    measurements[0]
                    for thread_id, measurements in counters[
                        "BR_INST_RETIRED_ALL_BRANCHES"
                    ][state].items()
                ]
            )

        metric = volume_misp / volume_branch
        return metric

    def compute_per_thread(self) -> np.ndarray:
        counters = self.values()

        volume_misp = []
        volume_branch = []
        for state in self._sdfg.states():
            if state not in counters["BR_MISP_RETIRED_ALL_BRANCHES"]:
                continue

            volume_misp.append(
                np.array(
                    [
                        measurements[0]
                        for thread_id, measurements in counters[
                            "BR_MISP_RETIRED_ALL_BRANCHES"
                        ][state].items()
                    ]
                )
            )
            volume_branch.append(
                np.array(
                    [
                        measurements[0]
                        for thread_id, measurements in counters[
                            "BR_INST_RETIRED_ALL_BRANCHES"
                        ][state].items()
                    ]
                )
            )

        metric = np.vstack(volume_misp).sum(axis=0, keepdims=False) / np.vstack(
            volume_branch
        ).sum(axis=0, keepdims=False)
        return metric
