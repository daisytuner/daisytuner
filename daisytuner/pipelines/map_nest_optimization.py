# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace

from typing import Any, Dict, List

from dace.transformation import pass_pipeline as ppl

from daisytuner.analysis.similarity.map_nest import MapNest
from daisytuner.passes.map_expanded_form import MapExpandedForm
from daisytuner.profiling.measure import create_data_report, arguments_from_data_report
from daisytuner.tuning import CutoutTuner


@dace.properties.make_properties
class MapNestOptimization(ppl.Pass):
    """
    An optimization pipeline moving a sliding window over the map nests
    of an SDFG. It cuts out the map nest and replaces it by an optimized
    SDFG. Different tuners can be used.
    """

    CATEGORY: str = "Optimization"

    def __init__(self, tuner: CutoutTuner, arguments: Dict = None) -> None:
        super().__init__()
        self._tuner = tuner
        self._arguments = arguments

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Scopes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Dict[str, Any]) -> int:
        map_nests = MapNest.enumerate_map_nests(sdfg)

        # Create data report if necessary
        if self._arguments is not None and any(
            [map_nest.is_data_dependent() for map_nest in map_nests]
        ):
            data_report = create_data_report(sdfg, arguments=self._arguments)

        results = {}
        for map_nest in map_nests:
            cutout = map_nest.as_cutout()

            pipeline = MapExpandedForm()
            pipeline.apply_pass(cutout, {})

            if not self._tuner.can_be_tuned(cutout):
                continue

            arguments = None
            if self._arguments is not None and map_nest.is_data_dependent():
                arguments = arguments_from_data_report(sdfg, data_report)

            optimized_cutout = self._tuner.tune(cutout, arguments=arguments)
            MapNestOptimization.replace_map_nest(map_nest, optimized_cutout)

            # TODO: Add tuning report
            results[map_nest.root] = None

        if len(results) > 0:
            return results
        else:
            return None

    @staticmethod
    def replace_map_nest(map_nest: MapNest, cutout: dace.SDFG) -> None:
        outer_state: dace.SDFGState = map_nest.state
        outer_sdfg: dace.SDFG = outer_state.parent
        sdfg = map_nest.sdfg

        # Collect inputs and outputs
        outermost_map_entry = None
        for node in map_nest.nodes():
            if not isinstance(node, dace.nodes.MapEntry):
                continue
            if map_nest.entry_node(node) is not None:
                continue

            outermost_map_entry = node
            break

        access_nodes = set()
        inputs = set()
        outputs = set()

        for iedge in outer_state.in_edges(outermost_map_entry):
            access_nodes.add(iedge.src)
            inputs.add(iedge.src.data)

        outermost_map_exit = outer_state.exit_node(outermost_map_entry)
        for iedge in outer_state.out_edges(outermost_map_exit):
            access_nodes.add(iedge.dst)
            outputs.add(iedge.dst.data)

        nsdfg_node = outer_state.add_nested_sdfg(
            cutout, parent=outer_sdfg, inputs=inputs, outputs=outputs
        )

        # Connect Nested SDFG
        for iedge in outer_state.in_edges(outermost_map_entry):
            outer_state.add_edge(
                iedge.src,
                None,
                nsdfg_node,
                iedge.data.data,
                dace.Memlet.from_array(
                    iedge.data.data, outer_sdfg.arrays[iedge.data.data]
                ),
            )

        outermost_map_exit = outer_state.exit_node(outermost_map_entry)
        for iedge in outer_state.out_edges(outermost_map_exit):
            outer_state.add_edge(
                nsdfg_node,
                iedge.data.data,
                iedge.dst,
                None,
                dace.Memlet.from_array(
                    iedge.data.data, outer_sdfg.arrays[iedge.data.data]
                ),
            )

        for edge in map_nest.edges():
            outer_state.remove_edge(edge)

        for node in map_nest.nodes():
            if node in access_nodes:
                continue

            outer_state.remove_node(node)

        dace.propagate_memlets_sdfg(sdfg)
