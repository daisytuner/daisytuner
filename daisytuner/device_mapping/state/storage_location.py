# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
from enum import Enum


class StorageLocation(Enum):
    HOST = 0
    DEVICE = 1
    BOTH = 2

    def is_host(self) -> bool:
        return self == StorageLocation.HOST or self == StorageLocation.BOTH

    def is_device(self) -> bool:
        return self == StorageLocation.DEVICE or self == StorageLocation.BOTH
