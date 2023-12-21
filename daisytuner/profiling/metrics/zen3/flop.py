# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import numpy as np
import dace
import platform


from daisytuner.profiling.metrics.metric import Metric


class FLOP(Metric):
    def __init__(
        self, sdfg: dace.SDFG, hostname: str = platform.node(), cache=None
    ) -> None:
        super().__init__(
            sdfg,
            [
                "RETIRED_SSE_AVX_FLOPS_ALL",
            ],
            dace.DeviceType.CPU,
            hostname,
            "zen3",
            cache=cache,
        )

    def compute(self) -> float:
        counters = self.values()

        volume = 0.0
        for state in self._sdfg.states():
            volume += sum(
                [
                    measurements[0]
                    for thread_id, measurements in counters[
                        "RETIRED_SSE_AVX_FLOPS_ALL"
                    ][state].items()
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
                        for thread_id, measurements in counters[
                            "RETIRED_SSE_AVX_FLOPS_ALL"
                        ][state].items()
                    ]
                )
            )

        metric = 1e-6 * np.vstack(volume).sum(axis=0, keepdims=False)
        return metric
