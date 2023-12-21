# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import platform

from daisytuner.profiling.metrics.compund_metric import CompoundMetric
from daisytuner.profiling.metrics.skylakeX.loads import Loads
from daisytuner.profiling.metrics.skylakeX.stores import Stores


class LoadStoreRatio(CompoundMetric):
    def __init__(self, sdfg: dace.SDFG, hostname: str = platform.node(), cache=None):
        super().__init__(
            metric_a=Loads(sdfg, hostname, cache=cache),
            metric_b=Stores(sdfg, hostname, cache=cache),
        )

    def compute(self) -> float:
        return self._metric_a.compute() / max(self._metric_b.compute(), 1)
