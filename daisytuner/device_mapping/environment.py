# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import gymnasium as gym

from typing import Tuple, Any

from daisytuner.device_mapping.action import Action
from daisytuner.device_mapping.state import GraphOfStates, InvalidScheduleException
from daisytuner.analysis.performance_modeling import PerformanceModel


class Environment(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        sdfg: dace.SDFG,
        reward_function: PerformanceModel = None,
        render_mode: str = None,
    ) -> None:
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self._sdfg = sdfg
        self._reward_function = reward_function
        self._state = GraphOfStates(sdfg)

    @property
    def state(self) -> GraphOfStates:
        return self._state

    def step(
        self, action: Tuple[Action, Any]
    ) -> Tuple[GraphOfStates, float, bool, bool, dict[str, Any]]:
        assert (
            self._state is not None and not self._state.terminated
        ), "Reset environment before using"

        # Default values
        reward = 0
        terminated = False
        info = {"error": None, "schedule": None}

        # Update environment
        action_type, item = action
        # try:
        if action_type == Action.NEXT_STATE:
            self._state.next_state()
        elif action_type == Action.SCHEDULE_MAP_NEST_HOST:
            _, active_gom = self._state.active()
            active_gom.schedule_map_nest(item, dace.DeviceType.CPU)
        elif action_type == Action.SCHEDULE_MAP_NEST_DEVICE:
            _, active_gom = self._state.active()
            active_gom.schedule_map_nest(item, dace.DeviceType.GPU)
        elif action_type == Action.COPY_HOST_TO_DEVICE:
            _, active_gom = self._state.active()
            active_gom.schedule_array(item, dace.DeviceType.GPU)
        elif action_type == Action.COPY_DEVICE_TO_HOST:
            _, active_gom = self._state.active()
            active_gom.schedule_array(item, dace.DeviceType.CPU)
        else:
            raise ValueError(f"Invalid action type {action_type}")

        # Check if game has ended
        terminated = self._state.terminated
        if terminated:
            schedule = self._state.generate()
            info["schedule"] = schedule

            if self._reward_function is not None:
                reward = self._reward_function.compute(schedule)
            else:
                reward = 1
        # except InvalidScheduleException as e:
        #     info["error"] = e
        #     terminated = True
        #     reward = -1

        return self._state, reward, terminated, False, info

    def reset(
        self, seed: int = None, options: dict[str, Any] = None
    ) -> Tuple[GraphOfStates, dict[str, Any]]:
        self._state = GraphOfStates(self._sdfg)
        return self._state, {}

    def render(self) -> None:
        raise NotImplementedError
