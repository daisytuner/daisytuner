# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace

from typing import Dict, List, Tuple

from dace.dtypes import DeviceType
from daisytuner.transfer_tuning.transfer_tuner import TransferTuner


class IdentityTransferTuner(TransferTuner):
    def __init__(self) -> None:
        super().__init__()

    def predict(self, batch, device: dace.DeviceType) -> List[Tuple[Dict, float]]:
        schedules = []
        speedups = []
        for _ in batch:
            speedups.append(1.0)
            schedules.append(None)

        return list(zip(schedules, speedups))

    def tune(self, batch, device: dace.DeviceType) -> List[dace.SDFG]:
        raise NotImplementedError
