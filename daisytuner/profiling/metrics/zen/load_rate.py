# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import platform

from daisytuner.profiling.metrics.compund_metric import CompoundMetric
from daisytuner.profiling.metrics.zen.loads import Loads
from daisytuner.profiling.metrics.zen.instructions import Instructions


class LoadRate(CompoundMetric):
    def __init__(self, sdfg: dace.SDFG, hostname: str = platform.node(), cache=None):
        super().__init__(
            metric_a=Loads(sdfg, hostname), metric_b=Instructions(sdfg, hostname)
        )
