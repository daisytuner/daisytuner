# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
from enum import IntEnum


class Action(IntEnum):
    NEXT_STATE = 0
    SCHEDULE_MAP_NEST_HOST = 1
    SCHEDULE_MAP_NEST_DEVICE = 2
    COPY_HOST_TO_DEVICE = 3
    COPY_DEVICE_TO_HOST = 4
    FREE_DEVICE = 5
    FREE_HOST = 6
    FUSE_MAPS = 7
