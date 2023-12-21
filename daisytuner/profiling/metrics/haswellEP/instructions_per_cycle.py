# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import platform

from daisytuner.profiling.metrics.compund_metric import CompoundMetric
from daisytuner.profiling.metrics.haswellEP.instructions import (
    Instructions,
)
from daisytuner.profiling.metrics.haswellEP.cycles import Cycles


class InstructionsPerCycle(CompoundMetric):
    def __init__(self, sdfg: dace.SDFG, hostname: str = platform.node(), cache=None):
        super().__init__(
            metric_a=Instructions(sdfg, hostname, cache=cache),
            metric_b=Cycles(sdfg, hostname, cache=cache),
        )
