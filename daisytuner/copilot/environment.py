# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import gymnasium as gym

from typing import Tuple, Any

from daisytuner.copilot.action import Action
from daisytuner.copilot.rewards import Reward
from daisytuner.copilot.state import State


class Environment(gym.Env):
    """
    An environment for the device mapping problem.
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self, sdfg: dace.SDFG, reward_function: Reward, render_mode: str = None
    ) -> None:
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self._sdfg = sdfg
        self._current_state = State(sdfg=self._sdfg)
        self._reward_function = reward_function

    def step(self, action: Action) -> Tuple[State, float, bool, bool, dict[str, Any]]:
        assert (
            self._current_state is not None
            and not self._current_state.terminated()
            and self._current_state.valid()
        ), "Reset environment before using"

        # Update environment
        info = self._current_state.update(action)

        # Compute reward
        reward = self._reward_function.compute(state=self._current_state)
        terminated = self._current_state.terminated() or not self._current_state.valid()

        return self._current_state, reward, terminated, False, info

    def reset(
        self, seed: int = None, options: dict[str, Any] = None
    ) -> Tuple[State, dict[str, Any]]:
        self._current_state = State(self._sdfg)
        return self._current_state, {}

    def render(self) -> None:
        raise NotImplementedError
