# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import pandas as pd

from typing import List

from abc import ABC, abstractmethod


class Metrics(ABC):
    def __init__(self, arch: str, groups: List[str]) -> None:
        self._arch = arch
        self._groups = groups

    @abstractmethod
    def compute(self, counters: pd.DataFrame) -> pd.DataFrame:
        pass
