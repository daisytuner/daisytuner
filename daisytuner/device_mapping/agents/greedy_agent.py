# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace

from daisytuner.device_mapping.state import GraphOfStates
from daisytuner.device_mapping import Action


class GreedyAgent:
    def __init__(self) -> None:
        self._copied = False
        self._decisions = None
        self._last_schedule = None

    def action(self, state: GraphOfStates) -> Action:
        active_state, active_gom = state.active()
        if self._decisions is None:
            self._decisions = {}
            for map_nest, schedule in state.map_nest_schedules().items():
                runtime_host = schedule["host"]["runtime"] / schedule["host"]["speedup"]
                runtime_device = (
                    schedule["device"]["runtime"] / schedule["device"]["speedup"]
                )
                if runtime_host <= runtime_device:
                    self._decisions[map_nest] = Action.SCHEDULE_MAP_NEST_HOST
                else:
                    self._decisions[map_nest] = Action.SCHEDULE_MAP_NEST_DEVICE

        active_maps = active_gom.active()
        if active_maps:
            active_map = active_maps[0]
            active_map_nest = active_gom.map_nests[active_map]
            decision = self._decisions[active_map_nest]

            if decision == Action.SCHEDULE_MAP_NEST_HOST:
                for inp in active_map_nest.inputs():
                    if active_gom.array_table[inp.data] != dace.DeviceType.CPU:
                        return Action.COPY_DEVICE_TO_HOST, inp.data

                for outp in active_map_nest.outputs():
                    if active_gom.array_table[outp.data] != dace.DeviceType.CPU:
                        return Action.COPY_DEVICE_TO_HOST, outp.data

                self._last_schedule = decision
                return decision, active_map
            else:
                for inp in active_map_nest.inputs():
                    if active_gom.array_table[inp.data] != dace.DeviceType.GPU:
                        return Action.COPY_HOST_TO_DEVICE, inp.data

                for outp in active_map_nest.outputs():
                    if active_gom.array_table[outp.data] != dace.DeviceType.GPU:
                        return Action.COPY_HOST_TO_DEVICE, outp.data

                self._last_schedule = decision
                return decision, active_map

        if self._last_schedule == Action.SCHEDULE_MAP_NEST_DEVICE:
            for array in active_gom.array_table:
                if active_gom.array_table[array] != dace.DeviceType.CPU:
                    return Action.COPY_DEVICE_TO_HOST, array

            self._last_schedule = None

        # Terminal states: Copy-back
        if state.is_terminal():
            for array in active_gom.array_table:
                if state.sdfg.arrays[array].transient:
                    continue

                if active_gom.array_table[array] != dace.DeviceType.CPU:
                    return Action.COPY_DEVICE_TO_HOST, array

        return Action.NEXT_STATE, None
