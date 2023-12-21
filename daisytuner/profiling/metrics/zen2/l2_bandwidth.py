# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import platform

from daisytuner.profiling.metrics.compund_metric import CompoundMetric
from daisytuner.profiling.metrics.zen2.l2_volume import L2Volume
from daisytuner.profiling.metrics.zen2.runtime import Runtime


class L2Bandwidth(CompoundMetric):
    def __init__(self, sdfg: dace.SDFG, hostname: str = platform.node(), cache=None):
        super().__init__(
            metric_a=L2Volume(sdfg, hostname, cache=cache),
            metric_b=Runtime(sdfg, hostname, cache=cache),
        )
