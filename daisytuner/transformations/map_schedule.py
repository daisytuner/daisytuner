# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace

from dace.sdfg import SDFG
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.state import SDFGState
from dace.transformation import transformation
from dace.properties import make_properties, Property, EnumProperty


@make_properties
class MapSchedule(transformation.SingleStateTransformation):
    map_entry = transformation.PatternNode(nodes.MapEntry)

    schedule_type = EnumProperty(
        dtype=dace.ScheduleType,
        default=dace.ScheduleType.Default,
        desc="Schedule type of the map",
    )
    collapse = Property(dtype=int, default=1)
    unroll = Property(dtype=bool, default=False)

    omp_schedule_type = EnumProperty(
        dtype=dace.OMPScheduleType,
        default=dace.OMPScheduleType.Default,
        desc="OMP schedule type",
    )
    omp_chunk_size = Property(dtype=int, default=0)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry)]

    def can_be_applied(self, state, expr_index, sdfg, permissive=False):
        # Ensure fixe point in dace passes
        if self.map_entry.map.schedule == self.schedule_type:
            if self.map_entry.map.collapse == self.collapse:
                if self.map_entry.map.unroll == self.unroll:
                    if self.schedule_type != dace.ScheduleType.CPU_Multicore:
                        return False

                    if self.map_entry.map.omp_schedule == self.omp_schedule_type:
                        if self.map_entry.map.omp_chunk_size == self.omp_chunk_size:
                            return False

        return True

    def apply(self, state: SDFGState, sdfg: SDFG):
        self.map_entry.map.schedule = self.schedule_type
        self.map_entry.map.collapse = self.collapse
        self.map_entry.map.unroll = self.unroll

        if self.schedule_type == dace.ScheduleType.CPU_Multicore:
            self.map_entry.map.omp_schedule = self.omp_schedule_type
            self.map_entry.map.omp_chunk_size = self.omp_chunk_size
