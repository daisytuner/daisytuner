# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
from enum import Enum


class Action(Enum):
    """
    Schedule decisions for the current selection.
    """

    SCHEDULE_NONE = 0
    SCHEDULE_HOST = 1
    SCHEDULE_DEVICE = 2
    COPY_HOST_TO_DEVICE = 3
    COPY_DEVICE_TO_HOST = 4
    NEXT_MAP_NEST = 5
    NEXT_ARRAY = 6
