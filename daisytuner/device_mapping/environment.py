# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import numpy as np

import gymnasium as gym

from typing import Tuple, Any

from daisytuner.device_mapping.action import Action
from daisytuner.device_mapping.state import State, InvalidStateException


class Environment(gym.Env):
    """
    A gym environment for the heterogeneous device mapping problem.
    A state consists of a Graph of Map Nests (GoM) and a currently
    selected map nest. An agent decides whether to schedule on CPU,
    GPU or select another map nest. Map Nests are selected in topo-
    logical order and map nests must be scheduled
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, sdfg: dace.SDFG, render_mode: str = None) -> None:
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.action_space = gym.spaces.Discrete(len(Action))
        self.observation_space = gym.spaces.Dict(
            {
                "gom": gym.spaces.Graph(
                    node_space=gym.spaces.Box(
                        low=-np.inf, high=np.inf, shape=(256,), dtype=np.float32
                    ),
                    edge_space=gym.spaces.Box(
                        low=-np.inf, high=np.inf, shape=(256,), dtype=np.float32
                    ),
                ),
                "cpu_schedule": gym.spaces.Sequence(
                    gym.spaces.Box(
                        low=-np.inf, high=np.inf, shape=(256,), dtype=np.float32
                    )
                ),
                "gpu_schedule": gym.spaces.Sequence(
                    gym.spaces.Box(
                        low=-np.inf, high=np.inf, shape=(256,), dtype=np.float32
                    )
                ),
            }
        )

        self._sdfg = sdfg
        self._current_state = State(sdfg=self._sdfg)

    def step(self, action: Action) -> Tuple[State, float, bool, bool, dict[str, Any]]:
        assert self._current_state is not None, "Reset environment before using"
        assert (
            not self._current_state.terminated()
        ), "Environment has reached final state"

        # Update state
        info = {}
        try:
            self._current_state.update(action)
            terminated = self._current_state.terminated()
            truncated = False
        except InvalidStateException as e:
            truncated = True
            terminated = True
            info["error"] = e

        if terminated and not truncated:
            scheduled_sdfg = self._current_state.generate_schedule()
            info["scheduled_sdfg"] = scheduled_sdfg
            reward = 1.0
        elif truncated:
            reward = -1.0
        else:
            reward = 0.0

        return self._current_state, reward, terminated, truncated, info

    def reset(
        self, seed: int = None, options: dict[str, Any] = None
    ) -> Tuple[State, dict[str, Any]]:
        self._current_state = State(self._sdfg)
        return self._current_state, {}

    def render(self) -> None:
        raise NotImplementedError
