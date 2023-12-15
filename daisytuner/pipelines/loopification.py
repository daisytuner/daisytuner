# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace

from typing import Dict, Any

from dace.transformation import pass_pipeline as ppl
from dace.transformation.pass_pipeline import FixedPointPipeline

from daisytuner.passes.map_expanded_form import MapExpandedForm
from daisytuner.passes.map_to_loop import MapToLoop
from daisytuner.transformations.greedy_tasklet_fusion import GreedyTaskletFusion


class Loopification(ppl.Pass):
    """
    Loopficiation pass converting parametric dataflow to control-flow.
    """

    CATEGORY: str = "Normalization"

    recursive = True

    def __init__(self) -> None:
        super().__init__()

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Dict[str, Any]) -> int:
        # Pre-condition for loopification
        pipeline = MapExpandedForm()
        pipeline.apply_pass(sdfg, pipeline_results)

        # Loopify SDFG
        pipeline = FixedPointPipeline((MapToLoop(),))
        pipeline.apply_pass(sdfg, pipeline_results)

        sdfg.apply_transformations_repeated(GreedyTaskletFusion)
        sdfg.simplify()

        return None
