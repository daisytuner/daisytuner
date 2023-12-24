# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import numpy as np
import dace
import platform


from daisytuner.profiling.metrics.metric import Metric


class Branches(Metric):
    def __init__(
        self, sdfg: dace.SDFG, hostname: str = platform.node(), cache=None
    ) -> None:
        super().__init__(
            sdfg,
            [
                "BR_INST_RETIRED_ALL_BRANCHES",
            ],
            dace.DeviceType.CPU,
            hostname,
            "skylakeX",
            cache=cache,
        )

    def compute(self) -> float:
        counters = self.values()

        volume_branches = 0.0
        for state in self._sdfg.states():
            if not state in counters["BR_INST_RETIRED_ALL_BRANCHES"]:
                continue

            volume_branches += sum(
                [
                    measurements[0]
                    for thread_id, measurements in counters[
                        "BR_INST_RETIRED_ALL_BRANCHES"
                    ][state].items()
                ]
            )

        metric = volume_branches
        return metric

    def compute_per_thread(self) -> np.ndarray:
        counters = self.values()

        volume_branches = []
        for state in self._sdfg.states():
            if not state in counters["BR_INST_RETIRED_ALL_BRANCHES"]:
                continue

            volume_branches.append(
                np.array(
                    [
                        measurements[0]
                        for thread_id, measurements in counters[
                            "BR_INST_RETIRED_ALL_BRANCHES"
                        ][state].items()
                    ]
                )
            )

        metric = np.vstack(volume_branches).sum(axis=0, keepdims=False)
        return metric
