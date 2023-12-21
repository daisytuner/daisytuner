# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import numpy as np

from daisytuner.profiling.metrics.metric import Metric


class CompoundMetric(Metric):
    def __init__(
        self,
        metric_a: Metric,
        metric_b: Metric,
    ):
        super().__init__(
            sdfg=metric_a._sdfg,
            device=metric_a._device,
            hostname=metric_a._hostname,
            codename=metric_a._codename,
            counters=metric_a._counters + metric_b._counters,
            cache={},
        )
        self._metric_a = metric_a
        self._metric_b = metric_b

    def compute(self) -> float:
        return self._metric_a.compute() / self._metric_b.compute()

    def compute_per_thread(self) -> np.array:
        return self._metric_a.compute() / self._metric_b.compute()

    def has_values(self) -> bool:
        return self._metric_a.has_values() and self._metric_b.has_values()
