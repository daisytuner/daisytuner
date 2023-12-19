# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace

from typing import Dict, Any

from dace.transformation.dataflow import AugAssignToWCR, PruneConnectors, PruneSymbols
from dace.transformation import pass_pipeline as ppl
from dace.transformation.interstate import (
    LoopToMap,
    MoveLoopIntoMap,
)
from daisytuner.transformations import (
    GreedyTaskletFusion,
    TaskletSimplification,
)


class DataflowMaximization(ppl.Pass):
    """
    Auto-Parallelization pass converting control-flow to parametric dataflow.
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
        applied = 0

        applied += sdfg.apply_transformations_repeated(
            (GreedyTaskletFusion, TaskletSimplification), validate=False
        )
        applied += sdfg.apply_transformations_repeated(
            (AugAssignToWCR, PruneConnectors), validate=False, permissive=True
        )
        applied += sdfg.apply_transformations_repeated(LoopToMap, validate=False)
        if applied > 0:
            dace.propagate_memlets_sdfg(sdfg)
            sdfg.validate()
            return applied

        # Move loop into map to facilitate detection of parallel reductions
        applied += sdfg.apply_transformations_repeated(MoveLoopIntoMap, validate=False)
        if applied > 0:
            dace.propagate_memlets_sdfg(sdfg)
            sdfg.validate()
            return applied

        return None
