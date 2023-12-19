# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace

from daisytuner.copilot.state import State
from daisytuner.copilot.state import Action


class HostAgent:
    def __init__(self) -> None:
        pass

    def action(self, state: State) -> Action:
        if not isinstance(state.selected_node, dace.nodes.MapEntry):
            return Action.SCHEDULE_NONE

        return Action.SCHEDULE_HOST
