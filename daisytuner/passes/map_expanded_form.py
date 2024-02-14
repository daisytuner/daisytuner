# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace

from typing import Dict, Any

from dace.transformation import pass_pipeline as ppl

from dace.transformation.dataflow import MapExpansion
from dace.transformation.optimizer import Optimizer


class MapExpandedForm(ppl.Pass):
    """
    Expands maps to an expanded form, which is used for auto-scheduling.
    """

    CATEGORY: str = "Normalization"

    recursive = True

    def __init__(self) -> None:
        super().__init__()

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Scopes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.Scopes

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Dict[str, Any]) -> int:
        applied = sdfg.apply_transformations_repeated(MapExpansion)
        if applied > 0:
            return applied
        else:
            return None

    @staticmethod
    def is_expanded_form(sdfg: dace.SDFG) -> bool:
        xforms = Optimizer(sdfg=sdfg).get_pattern_matches(patterns=(MapExpansion))
        return not any(True for _ in xforms)
