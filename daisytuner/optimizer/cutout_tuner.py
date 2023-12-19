# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace

from abc import ABC, abstractmethod
from typing import Dict, Tuple


class CutoutTuner(ABC):
    """
    A cutout tuner optimizes a small 'cutout' of an SDFG.
    """

    @abstractmethod
    def can_be_tuned(self, cutout: dace.SDFG) -> bool:
        """
        Checks whether the cutout can be tuned by the tuner.
        For instance, some tuners may require certain properties
        like a polyhedral representation.
        """
        pass

    @abstractmethod
    def tune(self, cutout: dace.SDFG, arguments: Dict) -> Tuple[dace.SDFG, Dict]:
        """
        Given a cutout, this method returns an optimized SDFG. This method
        should not apply in-place / change the input cutout.
        """
        pass
