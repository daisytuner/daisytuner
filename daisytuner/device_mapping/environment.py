# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import copy
import dace
import json
import gymnasium as gym

from typing import Tuple, Any, List

from daisytuner.analysis.similarity import MapNestModel
from daisytuner.analysis.similarity.benchmarking import CPUBenchmark, GPUBenchmark
from daisytuner.analysis.performance_modeling import PerformanceModel

from daisytuner.transfer_tuning.transfer_tuner import TransferTuner

from daisytuner.transformations import MapWrapping, ComponentFusion

from daisytuner.device_mapping.action import Action
from daisytuner.device_mapping.state import GraphOfStates, InvalidScheduleException
from daisytuner.device_mapping.state.identity_transfer_tuner import (
    IdentityTransferTuner,
)


class Environment(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        sdfg: dace.SDFG,
        cpu_benchmark: CPUBenchmark,
        gpu_benchmark: GPUBenchmark,
        transfer_tuner: TransferTuner = None,
        reward_function: PerformanceModel = None,
        render_mode: str = None,
    ) -> None:
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self._sdfg = sdfg
        self._cpu_benchmark = cpu_benchmark
        self._gpu_benchmark = gpu_benchmark
        self._transfer_tuner = transfer_tuner
        self._reward_function = reward_function

        print("Initializing game...")
        self._state = GraphOfStates(
            self._sdfg,
            cpu_benchmark=self._cpu_benchmark,
            gpu_benchmark=self._gpu_benchmark,
        )
        self._state.init(
            host_model=MapNestModel.create(dace.DeviceType.CPU),
            device_model=MapNestModel.create(dace.DeviceType.GPU),
            transfer_tuner=self._transfer_tuner
            if self._transfer_tuner is not None
            else IdentityTransferTuner(),
        )

        self._history = []

    @property
    def state(self) -> GraphOfStates:
        return self._state

    @property
    def history(self) -> List:
        return self._history

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
        self._history.append(
            [
                json.dumps(action_type),
                item.to_json(parent=self._state.active()[0])
                if isinstance(item, dace.nodes.Node)
                else json.dumps(item),
            ]
        )
        try:
            if action_type == Action.NEXT_STATE:
                self._state.next_state()
            elif action_type == Action.SCHEDULE_MAP_NEST_HOST:
                _, active_gom = self._state.active()
                active_gom.schedule_map_nest_host(item)
            elif action_type == Action.SCHEDULE_MAP_NEST_DEVICE:
                _, active_gom = self._state.active()
                active_gom.schedule_map_nest_device(item)
            elif action_type == Action.COPY_HOST_TO_DEVICE:
                _, active_gom = self._state.active()
                active_gom.copy_host_to_device(item)
            elif action_type == Action.COPY_DEVICE_TO_HOST:
                _, active_gom = self._state.active()
                active_gom.copy_device_to_host(item)
            elif action_type == Action.FREE_DEVICE:
                _, active_gom = self._state.active()
                active_gom.free_device(item)
            elif action_type == Action.FREE_HOST:
                _, active_gom = self._state.active()
                active_gom.free_host(item)
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
        except InvalidScheduleException as e:
            info["error"] = e
            terminated = True
            reward = -1

        return self._state, reward, terminated, False, info

    def reset(
        self, seed: int = None, options: dict[str, Any] = None
    ) -> Tuple[GraphOfStates, dict[str, Any]]:
        print("Initializing game...")
        self._state = GraphOfStates(
            self._sdfg,
            cpu_benchmark=self._cpu_benchmark,
            gpu_benchmark=self._gpu_benchmark,
        )
        self._state.init(
            host_model=MapNestModel.create(dace.DeviceType.CPU),
            device_model=MapNestModel.create(dace.DeviceType.GPU),
            transfer_tuner=self._transfer_tuner
            if self._transfer_tuner is not None
            else IdentityTransferTuner(),
        )
        return self._state, {}

    def render(self) -> None:
        raise NotImplementedError

    @staticmethod
    def preprocess(sdfg: dace.SDFG) -> dace.SDFG:
        sdfg_ = copy.deepcopy(sdfg)

        # Convert top-level code into kernels
        sdfg_.apply_transformations_repeated(MapWrapping)

        # Fuse components across states with happens-before memlets
        sdfg_.apply_transformations_repeated(ComponentFusion, validate=False)

        return sdfg_
