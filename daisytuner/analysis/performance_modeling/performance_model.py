# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
from abc import ABC, abstractmethod

import dace


class PerformanceModel(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def compute(self, state: dace.SDFG) -> float:
        pass
