# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace

from abc import ABC, abstractmethod

from typing import Dict, List, Tuple


class TransferTuner(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def predict(self, batch, device: dace.DeviceType) -> List[Tuple[Dict, float]]:
        pass

    @abstractmethod
    def tune(self, batch, device: dace.DeviceType) -> List[dace.SDFG]:
        pass
