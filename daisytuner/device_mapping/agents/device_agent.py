# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace

from daisytuner.device_mapping.state import GraphOfStates
from daisytuner.device_mapping import Action


class DeviceAgent:
    def __init__(self) -> None:
        self._copied = False

    def action(self, state: GraphOfStates) -> Action:
        active_state, active_gom = state.active()
        active_maps = active_gom.active()

        # Copy-in
        if active_state == state._start_state and not self._copied:
            for array in active_gom.array_table:
                if not active_gom.array_table[array].is_device():
                    return Action.COPY_HOST_TO_DEVICE, array

            for array in active_gom.array_table:
                if active_gom.array_table[array].is_host():
                    return Action.FREE_HOST, array

            self._copied = True

        # Schedule
        if active_maps:
            return Action.SCHEDULE_MAP_NEST_DEVICE, active_maps[0]

        # Terminal states: Copy-back
        if state.is_terminal():
            for array in active_gom.array_table:
                if state.sdfg.arrays[array].transient:
                    continue

                if not active_gom.array_table[array].is_host():
                    return Action.COPY_DEVICE_TO_HOST, array

        # Next state
        return Action.NEXT_STATE, None
