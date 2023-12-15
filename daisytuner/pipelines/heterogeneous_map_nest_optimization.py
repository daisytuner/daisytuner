# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import copy

from typing import Any, Dict
from tqdm import tqdm

from dace.transformation import pass_pipeline as ppl

from daisytuner.analysis.similarity.map_nest import MapNest
from daisytuner.passes.map_expanded_form import MapExpandedForm
from daisytuner.profiling.measure import (
    create_data_report,
    arguments_from_data_report,
    random_arguments,
    measure,
)
from daisytuner.transformations.map_wrapping import MapWrapping
from daisytuner.tuning import CutoutTuner


@dace.properties.make_properties
class HeterogeneousMapNestOptimization(ppl.Pass):
    """ """

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
        # Create a data report to capture all intermediate results
        if self._arguments is not None:
            arguments = self._arguments
        else:
            arguments = random_arguments(sdfg)

        data_report = create_data_report(sdfg, arguments=arguments)

        # Pre-processing
        pipeline = MapExpandedForm()
        pipeline.apply_pass(sdfg, {})
        sdfg.apply_transformations_repeated(MapWrapping)

        # Collect best map nest schedules for mapping decision
        map_nests = MapNest.enumerate_map_nests(sdfg)
        best_schedules = {}
        for map_nest in tqdm(map_nests):
            in_degree = map_nest.state.in_degree(map_nest.root)
            from_range, to_range, _ = map_nest.root.map.range.ranges[0]
            if in_degree == 0 or from_range == to_range:
                cutout = map_nest.as_cutout()
                args = arguments_from_data_report(cutout, data_report)

                # CPU
                instrumented_node = None
                for node in cutout.start_state.nodes():
                    if (
                        isinstance(node, dace.nodes.MapEntry)
                        and cutout.start_state.entry_node(node) is None
                    ):
                        node.map.instrument = dace.InstrumentationType.Timer
                        break

                cpu_runtime, *_ = measure(cutout, args, measurements=1)

                # GPU
                instrumented_node.map.instrument = (
                    dace.InstrumentationType.No_Instrumentation
                )
                cutout.apply_gpu_transformations()

                # Only measure computation, no data transfer
                for state in cutout.states():
                    for node in state.nodes():
                        if (
                            isinstance(node, dace.nodes.MapEntry)
                            and state.entry_node(node) is None
                        ):
                            node.instrument = dace.InstrumentationType.Timer
                            break

                args = copy.deepcopy(args)
                gpu_runtime, *_ = measure(cutout, args, measurements=1)

                best_schedules[map_nest.root] = {
                    dace.DeviceType.CPU: {
                        "initial": cpu_runtime,
                        "optimized": cpu_runtime,
                    },
                    dace.DeviceType.GPU: {
                        "initial": gpu_runtime,
                        "optimized": gpu_runtime,
                    },
                }
            else:
                pass

        return None


if __name__ == "__main__":
    sdfg = dace.SDFG.from_file("example.sdfg")
    sdfg.specialize({"KLEV": 137, "NPROMA": 128, "_for_it_24": 1})

    for dnode in sdfg.start_state.data_nodes():
        if sdfg.start_state.out_degree(dnode) == 0:
            sdfg.arrays[dnode.data].transient = False

    from daisytuner.pipelines.apriori_map_nest_normalization import (
        APrioriMapNestNormalization,
    )

    pipeline = APrioriMapNestNormalization(expand=False)
    pipeline.apply_pass(sdfg, {})

    pipeline = HeterogeneousMapNestOptimization(None, None)
    pipeline.apply_pass(sdfg, {})
