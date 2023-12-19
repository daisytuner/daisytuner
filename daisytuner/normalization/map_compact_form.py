# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace

from typing import Dict, Any

from dace.transformation import pass_pipeline as ppl
from dace.transformation.dataflow import (
    MapCollapse,
    TrivialMapElimination,
)
from dace.transformation.optimizer import Optimizer

from daisytuner.transformations import MapReroller, MapUntiling


class MapCompactForm(ppl.Pass):
    """
    Squashes maps to a compact form, for which many transformations are designed.
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
        applied = 0

        applied += sdfg.apply_transformations_repeated(
            TrivialMapElimination, validate=False
        )
        applied += sdfg.apply_transformations_repeated(MapCollapse, validate=False)
        applied += sdfg.apply_transformations_repeated(MapReroller, validate=False)
        applied += sdfg.apply_transformations_repeated(MapUntiling, validate=False)
        applied += sdfg.apply_transformations_repeated(MapCollapse, validate=False)

        if applied > 0:
            sdfg.validate()
            return applied
        else:
            return None

    @staticmethod
    def is_compact_form(sdfg: dace.SDFG) -> bool:
        xforms = Optimizer(sdfg=sdfg).get_pattern_matches(
            patterns=(MapCollapse, MapReroller, MapUntiling)
        )
        return not any(True for _ in xforms)
