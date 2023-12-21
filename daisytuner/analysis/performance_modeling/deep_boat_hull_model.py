# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace

from dace.sdfg.work_depth_analysis.work_depth import (
    tasklet_work,
    scope_work_depth,
)

from daisytuner.analysis.similarity.map_nest import MapNest
from daisytuner.analysis.similarity.map_nest_model import MapNestModel
from daisytuner.analysis.performance_modeling.boat_hull_model import BoatHullModel
from daisytuner.analysis.similarity.profiling_features.norm_coeffs import (
    TARGETS_MEAN_CPU,
    TARGETS_MEAN_GPU,
    TARGETS_STD_CPU,
    TARGETS_STD_GPU,
)


class DeepBoatHullModel(BoatHullModel):
    """ """

    def __init__(
        self,
        cpu_model: MapNestModel,
        gpu_model: MapNestModel,
        interconnect_bandwidth: float,
    ) -> None:
        super().__init__(
            host_peakflops=1,
            device_peakflops=1,
            host_cores=1,
            device_cores=1,
            host_memory_bandwidth=1,
            device_memory_bandwidth=1,
            interconnect_bandwidth=interconnect_bandwidth,
        )

        self._cpu_model = cpu_model
        self._gpu_model = gpu_model

    def _model_map_nest(
        self, sdfg: dace.SDFG, state: dace.SDFGState, node: dace.nodes.MapEntry
    ) -> float:
        # Predict runtime
        map_nest = MapNest(state, node)
        if node.map.schedule == dace.ScheduleType.CPU_Multicore:
            pred, *_ = self._cpu_model.predict(map_nest)
        else:
            pred, *_ = self._gpu_model.predict(map_nest)

        runtime = pred[0]
        return runtime
