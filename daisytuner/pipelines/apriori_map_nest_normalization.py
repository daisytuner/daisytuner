# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace

from typing import Any, Dict

from dace.transformation import pass_pipeline as ppl

from dace.sdfg.utils import consolidate_edges
from dace.transformation.pass_pipeline import FixedPointPipeline
from dace.transformation.passes import SimplifyPass
from dace.transformation.dataflow import CopyToMap

from daisytuner.transformations import (
    GreedyTaskletFusion,
    PerfectMapFusion,
    CopyToTasklet,
    ScalarFission,
)
from daisytuner.passes import (
    DataDependentSymbolAnalysis,
    DataflowMaximization,
    IndirectionPropagation,
    MapCompactForm,
    StrideMinimization,
)
from daisytuner.passes import MapInlining, MaximalMapFission


class APrioriMapNestNormalization(ppl.Pass):
    """
    A normalization pass for SDFGs which must be run before the optimization.
    """

    CATEGORY: str = "Normalization"

    def __init__(self, assumptions: Dict = None) -> None:
        super().__init__()
        self.assumptions = assumptions

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Dict[str, Any]) -> int:
        # Normalization
        results = {}

        # Early Canonicalization
        # 1. Expand library nodes
        # 2. Ensure compact form: collapse, rerolling, untiling
        # 3. Simplify memlets, i.e., no copy-syntax
        # 4. Fission scalars to minimize RAW-dependencies
        # 5. Fuse tasklets to minimize number of nodes
        dace.libraries.blas.default_implementation = "pure"
        sdfg.expand_library_nodes(recursive=True)

        pipeline = MapCompactForm()
        pipeline.apply_pass(sdfg, results)

        sdfg.apply_transformations_repeated(CopyToMap, validate=False)
        sdfg.apply_transformations_repeated(CopyToTasklet, validate=False)
        sdfg.apply_transformations_repeated(ScalarFission, validate=False)
        sdfg.apply_transformations_repeated(GreedyTaskletFusion, validate=False)
        sdfg.validate()

        # Normalization I: Maximal Map Fission
        # - Maximize dataflow (Loop to map, inlining)
        # - Inline maps into nested SDFGs
        # - Simplification
        # - Compact form
        # - (Actual) map fissioning
        pipeline = FixedPointPipeline(
            [
                SimplifyPass(),
                DataflowMaximization(),
                MapInlining(),
                MapCompactForm(),
                MaximalMapFission(),
            ]
        )
        pipeline.apply_pass(sdfg, results)
        sdfg.validate()

        # Canonizalization through Fusion with conditions:
        # - Locality: either ... or ..
        #       - Chain: One-to-one producer-consumer relation
        #       - Siblings: Contiguous memory locations (TODO)
        # - Perfect: All tasklets are at same depth of map levels
        #       - Don't violate maximal fission property
        sdfg.apply_transformations_repeated(PerfectMapFusion, validate=False)
        sdfg.apply_transformations_repeated(GreedyTaskletFusion, validate=False)

        pipeline = MapCompactForm()
        pipeline.apply_pass(sdfg, results)

        consolidate_edges(sdfg)
        sdfg.validate()

        # Normalization II: Stride Minimization
        pipeline = FixedPointPipeline(
            [StrideMinimization(assumptions=self.assumptions)]
        )
        pipeline.apply_pass(sdfg, results)

        return results

    @staticmethod
    def is_normalized(sdfg):
        if not MaximalMapFission.is_normalized(sdfg):
            return False

        if not StrideMinimization.is_normalized(sdfg):
            return False

        return True
