# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import random

from typing import Dict, List, Tuple
from daisytuner.transfer_tuning.transfer_tuner import TransferTuner


class SamplingTransferTuner(TransferTuner):
    def __init__(self) -> None:
        super().__init__()

    def predict(self, batch, device: dace.DeviceType) -> List[Tuple[Dict, float]]:
        schedules = []
        speedups = []
        for _ in batch:
            speedup = random.normalvariate(mu=1.5, sigma=10.0)
            speedup = max(speedup, 1.0)

            speedups.append(speedup)
            schedules.append(None)

        return schedules, speedups
