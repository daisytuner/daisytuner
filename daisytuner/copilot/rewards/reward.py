# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
from abc import ABC, abstractmethod

from daisytuner.copilot.state import State


class Reward(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def compute(self, state: State) -> float:
        pass
