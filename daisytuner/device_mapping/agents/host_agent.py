# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
from daisytuner.device_mapping.state import State
from daisytuner.device_mapping.state import Action


class HostAgent:
    def __init__(self) -> None:
        pass

    def action(self, state: State) -> Action:
        return Action.SCHEDULE_HOST
