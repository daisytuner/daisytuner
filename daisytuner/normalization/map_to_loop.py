# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace

from typing import Dict, Any

from dace.transformation import pass_pipeline as ppl

from dace.transformation.dataflow import MapToForLoop
from dace.transformation.interstate import InlineMultistateSDFG


class MapToLoop(ppl.Pass):
    """
    A pass converting the outermost maps to loops. The pass requires
    the SDFG to be in map-expanded-form.
    """

    CATEGORY: str = "Normalization"

    recursive = True

    def __init__(self) -> None:
        super().__init__()

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.Everything

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Dict[str, Any]) -> int:
        applied = sdfg.apply_transformations_repeated(
            (MapToForLoop, InlineMultistateSDFG)
        )
        if applied > 0:
            return applied

        return None
