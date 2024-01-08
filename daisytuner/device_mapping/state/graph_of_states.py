# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import copy
import dace
import networkx as nx

from typing import Tuple, Dict

from dace.sdfg.work_depth_analysis.work_depth import find_loop_guards_tails_exits
from dace.sdfg.graph import OrderedMultiDiGraph

from dace.transformation.optimizer import Optimizer
from dace.transformation.interstate import StateFusion

from daisytuner.analysis.similarity.map_nest_model import MapNest, MapNestModel
from daisytuner.analysis.similarity.benchmarking import CPUBenchmark, GPUBenchmark
from daisytuner.transfer_tuning import TransferTuner

from daisytuner.device_mapping.action import Action
from daisytuner.device_mapping.state.storage_location import StorageLocation
from daisytuner.device_mapping.state.graph_of_maps import GraphOfMaps
from daisytuner.device_mapping.state.invalid_schedule_exception import (
    InvalidScheduleException,
)


class GraphOfStates(OrderedMultiDiGraph):
    def __init__(
        self,
        sdfg: dace.SDFG,
        cpu_benchmark: CPUBenchmark,
        gpu_benchmark: GPUBenchmark,
        host_model: MapNestModel,
        device_model: MapNestModel,
        transfer_tuner: TransferTuner,
    ):
        super().__init__()
        self._sdfg = sdfg

        # Create DAG from top-level states
        self._arrays = set()
        self._graphs_of_maps: Dict[dace.SDFGState, GraphOfMaps] = {}

        dfs_tree = nx.dfs_tree(self._sdfg, source=self._sdfg.start_state)
        for state in dfs_tree.nodes():
            self.add_node(state)
            self._graphs_of_maps[state] = GraphOfMaps(
                state=state,
                cpu_benchmark=cpu_benchmark,
                gpu_benchmark=gpu_benchmark,
                host_model=host_model,
                device_model=device_model,
                transfer_tuner=transfer_tuner,
            )

            for dnode in state.data_nodes():
                if state.entry_node(dnode) is not None:
                    continue
                self._arrays.add(dnode.data)

        for src, dst in dfs_tree.edges():
            self.add_edge(src, dst, data=None)

        self._start_state = self._sdfg.start_state
        self._terminal_states = set(
            [node for node in self._sdfg.states() if self._sdfg.out_degree(node) == 0]
        )

        # Scheduling structures
        self._terminated = False
        self._active_state: dace.SDFGState = None
        self._visited = []
        self._walker = None

        # Move to first state
        self._walker = self._walk()
        self.next_state()

    @property
    def sdfg(self) -> dace.SDFG:
        return self._sdfg

    @property
    def terminated(self) -> bool:
        return self._terminated

    def is_terminal(self) -> bool:
        return self._active_state in self._terminal_states

    def active(self) -> Tuple[dace.SDFGState, GraphOfMaps]:
        return self._active_state, self._graphs_of_maps[self._active_state]

    def next_state(self) -> Tuple[dace.SDFGState, GraphOfMaps]:
        return next(self._walker)

    def map_nest_schedules(self) -> Dict[MapNest, Dict]:
        return self._graphs_of_maps[self._active_state].map_nest_schedules

    def _walk(self):
        for node in nx.dfs_preorder_nodes(self, source=self._start_state):
            # Skip visited states and artificial final state
            if self._graphs_of_maps[node].frozen():
                continue

            self._active_state = node

            # Set initial array table
            active_gom = self._graphs_of_maps[self._active_state]
            if self._active_state == self._start_state:
                active_gom.init({arr: StorageLocation.HOST for arr in self._arrays})
            else:
                for pred in self.predecessors(self._active_state):
                    pred_gom = self._graphs_of_maps[pred]
                    if pred_gom.frozen():
                        active_gom.init(pred_gom.array_table)
                        break

                assert pred_gom.array_table is not None

            yield self._active_state, active_gom

            # Post-processing: Check if valid
            self._visited.append(self._active_state)

            # All maps scheduled
            gom = self._graphs_of_maps[self._active_state]
            gom.freeze()
            if gom.active():
                raise InvalidScheduleException(
                    f"{self._active_state} not fully scheduled"
                )

            # Array tables are consistent across branches
            for node in self._sdfg.states():
                in_array_tables = []
                for inedge in self._sdfg.in_edges(node):
                    in_gom = self._graphs_of_maps[inedge.src]
                    if not in_gom.frozen():
                        continue

                    in_array_tables.append(in_gom.array_table)

                if not all([in_array_tables[0] == at for at in in_array_tables]):
                    raise InvalidScheduleException(
                        f"Inconsistent array tables at state {self._active_state}"
                    )

        # All non-transients are moved back to CPU
        for node in self._terminal_states:
            gom = self._graphs_of_maps[node]
            for array in self._arrays:
                if not self._sdfg.arrays[array].transient:
                    if not gom.array_table[array].is_host():
                        raise InvalidScheduleException(
                            f"{array} not on host in final state {node}"
                        )

        self._active_state = None
        self._terminated = True
        yield self._active_state, None
        raise StopIteration

    def generate(self) -> dace.SDFG:
        assert self.terminated

        schedule = dace.SDFG(self._sdfg.name + "_scheduled")
        schedule.specialize(self._sdfg.constants)
        for arr, desc in self._sdfg.arrays.items():
            schedule.add_datadesc(arr, copy.deepcopy(desc))

        state_action_mapping = {}
        state_start_end_nodes = {}
        for node in self._visited:
            state_start = schedule.add_state(node.label + "_start")
            state_end = schedule.add_state(node.label + "_end")
            state_start_end_nodes[node] = (state_start, state_end)

            gom = self._graphs_of_maps[node]
            state_action_mapping.update(gom.generate(schedule, state_start, state_end))
            state_action_mapping[state_start] = None
            state_action_mapping[state_end] = None

        for edge in self._sdfg.edges():
            schedule.add_edge(
                state_start_end_nodes[edge.src][1],
                state_start_end_nodes[edge.dst][0],
                data=copy.deepcopy(edge.data),
            )

        dace.sdfg.infer_types.infer_connector_types(schedule)
        dace.sdfg.infer_types.set_default_schedule_and_storage_types(schedule, None)

        schedule.validate()

        # State fusion to utilize concurrent utilization
        # 1. Fuse same types of states
        # - Fuses data transfers of same direction into same states
        # - Fuses maps of same schedule into samee state
        while True:
            xforms = Optimizer(sdfg=schedule).get_pattern_matches(
                patterns=[StateFusion]
            )
            xform = None
            for xf in xforms:
                if (
                    state_action_mapping[xf.first_state]
                    == state_action_mapping[xf.second_state]
                    or state_action_mapping[xf.second_state] is None
                ):
                    xform = xf
                    break

            if xform is None:
                break

            xform.apply(xform._sdfg, xform._sdfg)

        # 2. Overlap data transfers and computation
        while True:
            xforms = Optimizer(sdfg=schedule).get_pattern_matches(
                patterns=[StateFusion]
            )
            xform = None
            for xf in xforms:
                if (
                    state_action_mapping[xf.first_state] == Action.COPY_HOST_TO_DEVICE
                    and state_action_mapping[xf.second_state]
                    == Action.SCHEDULE_MAP_NEST_HOST
                ):
                    xform = xf
                    break
                elif (
                    state_action_mapping[xf.first_state] == Action.COPY_DEVICE_TO_HOST
                    and state_action_mapping[xf.second_state]
                    == Action.SCHEDULE_MAP_NEST_DEVICE
                ):
                    xform = xf
                    break

            if xform is None:
                break

            xform.apply(xform._sdfg, xform._sdfg)

        return schedule
