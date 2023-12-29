# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
from daisytuner.device_mapping.state import GraphOfStates
from daisytuner.device_mapping import Action


class HostAgent:
    def action(self, state: GraphOfStates) -> Action:
        _, active_gom = state.active()
        active_maps = active_gom.active()
        if active_maps:
            return Action.SCHEDULE_MAP_NEST_HOST, active_maps[0]

        return Action.NEXT_STATE, None
