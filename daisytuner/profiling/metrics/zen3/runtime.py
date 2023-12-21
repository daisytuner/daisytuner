# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import numpy as np
import dace
import platform


from daisytuner.profiling.metrics.metric import Metric


class Runtime(Metric):
    def __init__(
        self, sdfg: dace.SDFG, hostname: str = platform.node(), cache=None
    ) -> None:
        super().__init__(
            sdfg,
            [
                "Timer",
            ],
            dace.DeviceType.CPU,
            hostname,
            "zen3",
            cache=cache,
        )

    def compute(self) -> float:
        counters = self.values()

        runtime = 0.0
        for state in self._sdfg.states():
            runtime += max(
                [
                    measurements[0]
                    for thread_id, measurements in counters["Timer"][state].items()
                ]
            )

        metric = 1e-3 * runtime
        return metric

    def compute_per_thread(self) -> np.ndarray:
        counters = self.values()

        runtime = []
        for state in self._sdfg.states():
            runtime.append(
                np.array(
                    [
                        measurements[0]
                        for thread_id, measurements in counters["Timer"][state].items()
                    ]
                )
            )

        metric = 1e-3 * np.vstack(runtime).sum(axis=0, keepdims=False)
        return metric
