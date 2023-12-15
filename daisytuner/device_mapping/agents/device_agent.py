# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace

from daisytuner.device_mapping.state import State
from daisytuner.device_mapping.state import Action


class DeviceAgent:
    def __init__(self) -> None:
        pass

    def action(self, state: State) -> Action:
        map_nest = state.graph_of_map_nests.map_nests[state.selected_map_nest]

        host_array = set()
        for inp in map_nest.inputs():
            if state.array_table[inp.data] == dace.DeviceType.CPU:
                host_array.add(inp.data)

        for outp in map_nest.outputs():
            if state.array_table[outp.data] == dace.DeviceType.CPU:
                host_array.add(outp.data)

        if not host_array:
            return Action.SCHEDULE_DEVICE
        elif state.selected_array in host_array:
            return Action.COPY_HOST_TO_DEVICE
        elif not state.selected_array in host_array:
            return Action.NEXT_ARRAY
        else:
            raise ValueError("No action possible")
