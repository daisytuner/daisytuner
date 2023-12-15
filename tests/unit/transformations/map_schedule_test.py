# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace

from daisytuner.transformations import MapSchedule


def test_omp_schedule():
    @dace.program
    def sdfg_with_map(B: dace.float64[32]):
        for i in dace.map[0:32]:
            with dace.tasklet:
                b >> B[i]
                b = 0

    sdfg = sdfg_with_map.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations(
        MapSchedule,
        options={
            "schedule_type": dace.ScheduleType.CPU_Multicore,
            "omp_schedule_type": dace.OMPScheduleType.Dynamic,
            "omp_chunk_size": 4,
        },
    )
    assert applied == 1

    for node in sdfg.start_state.nodes():
        if isinstance(node, dace.nodes.MapEntry):
            assert node.map.schedule == dace.ScheduleType.CPU_Multicore
            assert node.map.omp_schedule == dace.OMPScheduleType.Dynamic
            assert node.map.omp_chunk_size == 4


def test_sequential_schedule():
    @dace.program
    def sdfg_with_map(B: dace.float64[32]):
        for i in dace.map[0:32]:
            with dace.tasklet:
                b >> B[i]
                b = 0

    sdfg = sdfg_with_map.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations(
        MapSchedule,
        options={
            "schedule_type": dace.ScheduleType.Sequential,
            "omp_schedule_type": dace.OMPScheduleType.Dynamic,
            "omp_chunk_size": 4,
        },
    )
    assert applied == 1

    for node in sdfg.start_state.nodes():
        if isinstance(node, dace.nodes.MapEntry):
            assert node.map.schedule == dace.ScheduleType.Sequential
            assert node.map.omp_schedule != dace.OMPScheduleType.Dynamic
            assert node.map.omp_chunk_size != 4
