# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import numpy as np

from abc import abstractmethod

from daisytuner.profiling.performance_counters import PerformanceCounters


class Metric(PerformanceCounters):
    @abstractmethod
    def compute(self) -> float:
        pass

    @abstractmethod
    def compute_per_thread(self) -> np.array:
        pass
