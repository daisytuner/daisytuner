# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace

from daisytuner.copilot.state import State, BeginState, EndState, EmptyNode
from daisytuner.copilot.state import Action


class DeviceAgent:
    def __init__(self) -> None:
        pass

    def action(self, state: State) -> Action:
        if isinstance(state.selected_node, BeginState):
            return Action.SCHEDULE_NONE
        if isinstance(state.selected_node, EmptyNode):
            return Action.SCHEDULE_NONE
        if isinstance(state.selected_node, EndState):
            device_arrays = {
                k for k, v in state.array_table.items() if v == dace.DeviceType.GPU
            }
            if not device_arrays:
                return Action.SCHEDULE_NONE
            elif state.selected_array in device_arrays:
                return Action.COPY_DEVICE_TO_HOST
            elif state.selected_array not in device_arrays:
                return Action.NEXT_ARRAY

        map_nest = state.graph_of_map_nests.map_nests[state.selected_node]

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
        elif state.selected_array not in host_array:
            return Action.NEXT_ARRAY
        else:
            raise ValueError("No action possible")
