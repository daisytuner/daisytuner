# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
from daisytuner.copilot.rewards.reward import Reward
from daisytuner.copilot.state import State


class Validity(Reward):
    def compute(self, state: State) -> float:
        if state.terminated():
            return 1.0
        elif not state.valid():
            return -1.0
        else:
            return 0.0
