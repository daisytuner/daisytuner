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
                "RETIRED_BRANCH_INSTR",
            ],
            dace.DeviceType.CPU,
            hostname,
            "zen3",
            cache=cache,
        )

    def compute(self) -> float:
        counters = self.values()

        volume_branches = 0.0
        for state in self._sdfg.states():
            if not state in counters["RETIRED_BRANCH_INSTR"]:
                continue

            volume_branches += sum(
                [
                    measurements[0]
                    for thread_id, measurements in counters["RETIRED_BRANCH_INSTR"][
                        state
                    ].items()
                ]
            )

        metric = volume_branches
        return metric
