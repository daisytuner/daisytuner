# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace

from typing import Dict, Any

from dace.transformation import pass_pipeline as ppl

from daisytuner.transformations import ComponentFission


class StateReordering(ppl.Pass):
    """ """

    CATEGORY: str = "Normalization"

    recursive = True

    def __init__(self) -> None:
        super().__init__()

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.Everything

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Dict[str, Any]) -> int:
        sdfg.apply_transformations_repeated(ComponentFission)

        return None
