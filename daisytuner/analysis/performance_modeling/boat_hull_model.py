# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
# The class adapts several parts of dace's work-depth computation
import dace
import dace.sdfg.propagation as propagation

from dace.sdfg.work_depth_analysis.work_depth import (
    get_tasklet_work,
    scope_work_depth,
    find_loop_guards_tails_exits,
)

from daisytuner.analysis.performance_modeling.performance_model import PerformanceModel


class BoatHullModel(PerformanceModel):
    """
    Boat Hull Model

    Model:
        - runtime_sdfg = max(runtime_path) for all paths (start_state -> final_state)
        - runtime_path = sum(runtime_state) for all states along path
        - runtime_state = max(runtime_host, runtime_device)
        - runtime_host = sum(runtime_map_nest) + data_transfer_to_host for all host_map_nests
        - runtime_device = sum(runtime_map_nest) + data_transfer_to_device for all device_map_nests
        - runtime_map_nest = max(compute, memory)
        - compute = max(1, parallelism / cores) * flops / peakflops_per_core
        - memory = sum(memlet.volume) / write_main_memory_bandwidth
    """

    def __init__(
        self,
        host_peakflops: float,
        device_peakflops: float,
        host_cores: int,
        device_cores: int,
        host_memory_bandwidth: float,
        device_memory_bandwidth: float,
        interconnect_bandwidth: float,
    ) -> None:
        super().__init__()

        self._host_peakflops = host_peakflops / host_cores
        self._device_peakflops = device_peakflops / device_cores
        self._host_cores = host_cores
        self._device_cores = device_cores
        self._host_memory_bandwidth = host_memory_bandwidth
        self._device_memory_bandwidth = device_memory_bandwidth
        self._interconnect_bandwidth = interconnect_bandwidth

    def compute(self, schedule: dace.SDFG) -> float:
        # Preprocess
        dace.sdfg.infer_types.infer_connector_types(schedule)
        dace.sdfg.infer_types.set_default_schedule_and_storage_types(schedule, None)
        for sd in schedule.all_sdfgs_recursive():
            propagation.propagate_states(sd, concretize_dynamic_unbounded=True)

        # 1. Performance modeling for each state
        # Assumptions:
        #    - runtime_device = max(compute, memory) + data_transfers_to_device
        #    - runtime of state = max(runtime_cpu, runtime_gpu)
        perf_model = {}
        for sdfg_state in schedule.nodes():
            # a. max(compute, memory)
            host_runtime = 0.0
            device_runtime = 0.0
            for node in sdfg_state.nodes():
                if (
                    not isinstance(node, dace.nodes.MapEntry)
                    or sdfg_state.entry_node(node) is not None
                ):
                    continue

                runtime = self._model_map_nest(schedule, sdfg_state, node)
                if node.map.schedule == dace.ScheduleType.CPU_Multicore:
                    host_runtime += runtime
                else:
                    device_runtime += runtime

            # b. data_transfers_to_device
            to_device_transfers = 0.0
            from_device_transfers = 0.0
            for dnode in sdfg_state.data_nodes():
                if (
                    sdfg_state.in_degree(dnode) == 0
                    and sdfg_state.out_degree(dnode) == 1
                ):
                    desc = schedule.arrays[dnode.data]
                    edge = sdfg_state.out_edges(dnode)[0]
                    if not isinstance(edge.dst, dace.nodes.AccessNode):
                        continue

                    from_memory = schedule.arrays[edge.src.data].storage
                    to_memory = schedule.arrays[edge.dst.data].storage
                    if to_memory == from_memory:
                        continue
                    elif (
                        to_memory == dace.StorageType.GPU_Global
                        and from_memory == dace.StorageType.CPU_Heap
                    ):
                        from_device_transfers += desc.total_size * desc.dtype.bytes
                    elif (
                        to_memory == dace.StorageType.GPU_Global
                        and from_memory == dace.StorageType.CPU_Heap
                    ):
                        to_device_transfers += desc.total_size * desc.dtype.bytes
                elif (
                    sdfg_state.out_degree(dnode) == 0
                    and sdfg_state.in_degree(dnode) == 1
                ):
                    edge = sdfg_state.in_edges(dnode)[0]
                    if not isinstance(edge.src, dace.nodes.AccessNode):
                        continue

                    from_memory = schedule.arrays[edge.src.data].storage
                    to_memory = schedule.arrays[edge.dst.data].storage
                    if to_memory == from_memory:
                        continue
                    elif (
                        to_memory == dace.StorageType.GPU_Global
                        and from_memory == dace.StorageType.CPU_Heap
                    ):
                        from_device_transfers += desc.total_size * desc.dtype.bytes
                    elif (
                        to_memory == dace.StorageType.GPU_Global
                        and from_memory == dace.StorageType.CPU_Heap
                    ):
                        to_device_transfers += desc.total_size * desc.dtype.bytes

                to_device_transfers = (
                    1e-6 * to_device_transfers / self._interconnect_bandwidth
                )
                from_device_transfers = (
                    1e-6 * from_device_transfers / self._interconnect_bandwidth
                )

            host_runtime += from_device_transfers
            device_runtime += to_device_transfers

            perf_model[sdfg_state] = (
                max(host_runtime, device_runtime) * sdfg_state.executions
            )

        # 2. Convert the SDFG into a DAG
        nodes_oNodes_exits = find_loop_guards_tails_exits(schedule._nx)

        for node, oNode, exits in nodes_oNodes_exits:
            schedule.remove_edge(schedule.edges_between(oNode, node)[0])
            for e in exits:
                if len(schedule.edges_between(oNode, e)) == 0:
                    # no edge there yet
                    schedule.add_edge(oNode, e, dace.InterstateEdge())
                if len(schedule.edges_between(node, e)) > 0:
                    # edge present --> remove it
                    schedule.remove_edge(schedule.edges_between(node, e)[0])

        # add a dummy exit to the SDFG, such that each path ends there.
        dummy_exit = schedule.add_state("dummy_exit")
        for state in schedule.nodes():
            if len(schedule.out_edges(state)) == 0 and state != dummy_exit:
                schedule.add_edge(state, dummy_exit, dace.InterstateEdge())

        # 3. Performance modeling for SDFG: Compute slowest path through schedule
        # Assumption:
        #   - sdfg_runtime = sum(state_runtime)
        max_runtime = 0.0
        for path in schedule.all_simple_paths(
            schedule.start_state, dummy_exit, as_edges=False
        ):
            runtime = 0.0
            for state in path:
                runtime += perf_model[state]

            max_runtime = max(runtime, max_runtime)

        max_runtime += perf_model[schedule.start_state]
        return max_runtime

    def _model_map_nest(
        self, sdfg: dace.SDFG, state: dace.SDFGState, node: dace.nodes.MapEntry
    ) -> float:
        # i. compute
        mflops, compute = self._model_compute(sdfg, state, node)

        # ii. memory
        memory = self._model_memory(state, node)

        # iii. roofline
        runtime = max(compute, memory)

        return mflops, runtime

    def _model_compute(
        self, sdfg: dace.SDFG, state: dace.SDFGState, node: dace.nodes.MapEntry
    ) -> float:
        w_d_map = {}
        scope_work_depth(
            state,
            w_d_map,
            get_tasklet_work,
            sdfg.constants,
            {},
            {},
            entry=node,
            detailed_analysis=False,
        )
        work, depth = w_d_map[node]
        avg_parallelism = work / depth
        mflops = 1e-6 * work

        if node.map.schedule == dace.ScheduleType.CPU_Multicore:
            runtime = (
                (avg_parallelism / self._host_cores) * mflops / self._host_peakflops
            )
        elif node.map.schedule == dace.ScheduleType.GPU_Device:
            runtime = (
                (avg_parallelism / self._device_cores) * mflops / self._device_peakflops
            )
        else:
            raise ValueError("Invalid schedule type")

        return runtime

    def _model_memory(
        self, sdfg: dace.SDFG, state: dace.SDFGState, node: dace.nodes.MapEntry
    ) -> float:
        volume = 0.0
        for iedge in state.in_edges(node):
            desc = sdfg.arrays[iedge.data.data]
            volume += iedge.data.volume * desc.dtype.bytes
        for oedge in state.out_edges(node):
            desc = sdfg.arrays[iedge.data.data]
            volume += oedge.data.volume * desc.dtype.bytes

        if node.map.schedule == dace.ScheduleType.CPU_Multicore:
            runtime = 1e-6 * volume / self._host_memory_bandwidth
        elif node.map.schedule == dace.ScheduleType.GPU_Device:
            runtime = 1e-6 * volume / self._device_memory_bandwidth
        else:
            raise ValueError("Invalid schedule type")

        return runtime
