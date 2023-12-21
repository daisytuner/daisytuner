# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import copy
import numpy as np

from typing import Dict

from daisytuner.analysis.performance_modeling.performance_model import PerformanceModel
from daisytuner.copilot.rewards.reward import Reward
from daisytuner.copilot.state import State


class Performance(Reward):
    def __init__(
        self, performance_model: PerformanceModel = None, arguments: Dict = None
    ) -> None:
        super().__init__()
        self._performance_model = performance_model
        self._arguments = arguments
        assert self._performance_model is not None or self._arguments is not None

    def compute(self, state: State) -> float:
        if state.terminated():
            schedule = state.generate_schedule()
            if self._performance_model:
                runtime = self._performance_model.compute(schedule)
            else:
                schedule.instrument = dace.InstrumentationType.Timer

                csdfg = schedule.compile()
                args = copy.deepcopy(self._arguments)
                csdfg(**args)

                report = schedule.get_latest_report()
                durations = list(report.durations.values())[0]
                durations = list(durations.values())[0]
                durations = list(durations.values())[0]
                durations = np.array(durations)
                runtime = np.median(durations)

                schedule.instrument = dace.InstrumentationType.No_Instrumentation

            return 1.0 / (runtime + 1e-9)
        elif not state.valid():
            return -1.0
        else:
            return 0.0
