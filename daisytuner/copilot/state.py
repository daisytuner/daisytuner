# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import copy
import dace
import networkx as nx

from typing import List, Dict, Union

from daisytuner.analysis.gom.graph_of_maps import (
    GraphOfMaps,
    BeginState,
    EndState,
    EmptyNode,
)
from daisytuner.copilot.action import Action


class InvalidStateException(Exception):
    pass


class State:
    def __init__(self, sdfg: dace.SDFG) -> None:
        self._sdfg: dace.SDFG = sdfg

        # Schedule to be constructed
        self._schedule = []
        self._scheduled_nodes = set()

        # Set up arrays
        self._arrays = list(sdfg.arrays)
        self._array_table = {arr: dace.DeviceType.CPU for arr in sdfg.arrays}
        self._selected_array = self._arrays[0]

        ##### Construction Graph of Map Nests #####
        self._graph_of_map_nests = GraphOfMaps(self._sdfg)

        # Define active maps
        self._active_nodes = []
        for node in self._graph_of_map_nests.nodes():
            if self._graph_of_map_nests.in_degree(node) == 0:
                self._active_nodes.append(node)

        self._selected_node = self._active_nodes[0]

    @property
    def sdfg(self) -> dace.SDFG:
        return self._sdfg

    @property
    def arrays(self) -> List[str]:
        return self._arrays

    @property
    def array_table(self) -> Dict[str, dace.DeviceType]:
        return self._array_table

    @property
    def graph_of_map_nests(self) -> nx.DiGraph:
        return self._graph_of_map_nests

    @property
    def selected_array(self) -> str:
        return self._selected_array

    @property
    def selected_node(self) -> dace.nodes.MapEntry:
        return self._selected_node

    def terminated(self) -> bool:
        return self._selected_node == None

    def _next_array(self) -> str:
        index = self._arrays.index(self._selected_array)
        self._selected_array = self._arrays[(index + 1) % len(self._arrays)]
        return self._selected_array

    def _next_map_nest(self) -> dace.nodes.MapEntry:
        active_nodes = self.active_nodes()
        index = active_nodes.index(self._selected_node)
        self._selected_node = active_nodes[(index + 1) % len(active_nodes)]
        return self._selected_node

    def _copy_to_host(self, array: str) -> None:
        if self._array_table[array] != dace.DeviceType.GPU:
            raise InvalidStateException(f"COPY_HOST: {array} already on host")

        self._schedule.append((Action.COPY_DEVICE_TO_HOST, array))
        self._array_table[array] = dace.DeviceType.CPU

    def _copy_to_device(self, array: str) -> None:
        if self._array_table[array] != dace.DeviceType.CPU:
            raise InvalidStateException(f"COPY_DEVICE: {array} already on gpu")

        self._schedule.append((Action.COPY_HOST_TO_DEVICE, array))
        self._array_table[array] = dace.DeviceType.GPU

    def _schedule_none(self, node: Union[EmptyNode, BeginState, EndState]) -> None:
        if not isinstance(node, (BeginState, EndState, EmptyNode)):
            raise InvalidStateException(
                f"SCHEDULE_NODE: {node} does not support schedule type"
            )

        self._schedule.append((Action.SCHEDULE_NONE, node))
        self._update_active(node)

    def _schedule_host(self, map_nest: dace.nodes.MapEntry) -> None:
        if not isinstance(map_nest, dace.nodes.MapEntry):
            raise InvalidStateException(
                f"SCHEDULE_HOST: {map_nest} does not support schedule type"
            )

        _map_nest = self._graph_of_map_nests.map_nests[map_nest]
        for inp in _map_nest.inputs():
            if self._array_table[inp.data] != dace.DeviceType.CPU:
                raise InvalidStateException(
                    f"SCHEDULE_HOST: input dependency {inp.data} not on host"
                )
        for outp in _map_nest.outputs():
            if self._array_table[outp.data] != dace.DeviceType.CPU:
                raise InvalidStateException(
                    f"SCHEDULE_HOST: input dependency {outp.data} not on host"
                )

        self._schedule.append((Action.SCHEDULE_HOST, map_nest))
        self._update_active(map_nest)

    def _schedule_device(self, map_nest: dace.nodes.MapEntry) -> None:
        if not isinstance(map_nest, dace.nodes.MapEntry):
            raise InvalidStateException(
                f"SCHEDULE_DEVICE: {map_nest} does not support schedule type"
            )

        _map_nest = self._graph_of_map_nests.map_nests[map_nest]
        for inp in _map_nest.inputs():
            if self._array_table[inp.data] != dace.DeviceType.GPU:
                raise InvalidStateException(
                    f"SCHEDULE_DEVICE: input dependency {inp.data} not on device"
                )
        for outp in _map_nest.outputs():
            if self._array_table[outp.data] != dace.DeviceType.GPU:
                raise InvalidStateException(
                    f"SCHEDULE_DEVICE: input dependency {outp.data} not on device"
                )

        self._schedule.append((Action.SCHEDULE_DEVICE, map_nest))
        self._update_active(map_nest)

    def _update_active(self, node) -> None:
        self._scheduled_nodes.add(node)
        self._active_nodes.remove(node)
        self._selected_node = None

        successors = set()
        for (src, dst) in self._graph_of_map_nests.out_edges(node):
            # TODO: Distinguish edges by attributes
            if False:
                continue

            successors.add(dst)

        for succ in successors:
            if all(
                [
                    pred in self._scheduled_nodes
                    for pred in self._graph_of_map_nests.predecessors(succ)
                ]
            ):
                self._active_nodes.append(succ)

        if self._active_nodes:
            if isinstance(node, EndState):
                self._selected_node = self._active_nodes[0]
            else:
                self._selected_node = list(
                    filter(lambda n: not isinstance(n, BeginState), self._active_nodes)
                )[0]

    def active_nodes(self) -> List:
        if isinstance(self._selected_node, BeginState):
            return self._active_nodes
        else:
            return list(
                filter(lambda n: not isinstance(n, BeginState), self._active_nodes)
            )

    def update(self, action: Action) -> None:
        if action == Action.SCHEDULE_NONE:
            self._schedule_none(self._selected_node)
        elif action == Action.SCHEDULE_HOST:
            self._schedule_host(self._selected_node)
        elif action == Action.SCHEDULE_DEVICE:
            self._schedule_device(self._selected_node)
        elif action == Action.COPY_HOST_TO_DEVICE:
            self._copy_to_device(self._selected_array)
        elif action == Action.COPY_DEVICE_TO_HOST:
            self._copy_to_host(self._selected_array)
        elif action == Action.NEXT_ARRAY:
            self._next_array()
        elif action == Action.NEXT_MAP_NEST:
            self._next_map_nest()
        else:
            raise ValueError(f"Invalid action: {action}")

    def generate_schedule(self) -> dace.SDFG:
        assert self.terminated(), "Can only schedule when terminated"

        scheduled_sdfg = dace.SDFG(
            name=self._sdfg.name + "scheduled",
        )
        scheduled_sdfg.specialize(self._sdfg.constants)
        for array, desc in self._sdfg.arrays.items():
            scheduled_sdfg.add_datadesc(array, copy.deepcopy(desc))

        # Map states to original SDFG
        state_mapping = {}

        # Add initial state
        action, item = self._schedule.pop(0)
        assert isinstance(item, BeginState)

        init_state = scheduled_sdfg.add_state("start_0", is_start_block=True)
        state_mapping[item.state] = [init_state, None]

        # Generate states
        last_state = init_state
        for i, (action, item) in enumerate(self._schedule):
            if isinstance(item, BeginState):
                # Control nodes mark the beginning of a new state
                state = scheduled_sdfg.add_state(f"begin_{i+1}")
                state_mapping[item.state] = [state, None]
            elif isinstance(item, EmptyNode):
                # Control nodes mark the beginning of a new state
                state = scheduled_sdfg.add_state(f"empty_{i+1}")
                scheduled_sdfg.add_edge(last_state, state, dace.InterstateEdge())
            elif isinstance(item, EndState):
                state = scheduled_sdfg.add_state(f"end_{i+1}")
                scheduled_sdfg.add_edge(last_state, state, dace.InterstateEdge())
                state_mapping[item.state][1] = state
            else:
                state = scheduled_sdfg.add_state(f"action_{i+1}")
                self._generate_state(scheduled_sdfg, state, action=action, item=item)
                scheduled_sdfg.add_edge(last_state, state, dace.InterstateEdge())

            last_state = state

        for edge in self._sdfg.edges():
            scheduled_sdfg.add_edge(
                state_mapping[edge.src][1],
                state_mapping[edge.dst][0],
                copy.deepcopy(edge.data),
            )

        device_arrays = [
            k
            for k, v in self._array_table.items()
            if v == dace.DeviceType.GPU and not self._sdfg.arrays[k].transient
        ]
        if device_arrays:
            copy_back_state = scheduled_sdfg.add_state("copy_back_state")
            for arr in device_arrays:
                self._generate_copy_device_to_host(scheduled_sdfg, copy_back_state, arr)

            final_states = []
            for state in scheduled_sdfg.states():
                if state != copy_back_state and scheduled_sdfg.out_degree(state) == 0:
                    final_states.append(state)

            for state in final_states:
                scheduled_sdfg.add_edge(state, copy_back_state, dace.InterstateEdge())

        # Postprocessing
        # scheduled_sdfg.simplify()
        scheduled_sdfg.validate()
        return scheduled_sdfg

    def _generate_state(
        self,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        action: Action,
        item: Union[dace.nodes.MapEntry, BeginState, EndState, str],
    ) -> None:
        if action == Action.SCHEDULE_HOST:
            self._generate_schedule_host(sdfg, state, item)
        elif action == Action.SCHEDULE_HOST:
            self._generate_schedule_host(sdfg, state, item)
        elif action == Action.SCHEDULE_DEVICE:
            self._generate_schedule_device(sdfg, state, item)
        elif action == Action.COPY_HOST_TO_DEVICE:
            self._generate_copy_host_to_device(sdfg, state, item)
        elif action == Action.COPY_DEVICE_TO_HOST:
            self._generate_copy_device_to_host(sdfg, state, item)
        else:
            raise ValueError(f"Invalid action {action}")

    def _generate_schedule_host(
        self, sdfg: dace.SDFG, state: dace.SDFGState, item: dace.nodes.MapEntry
    ) -> None:
        map_nest = self._graph_of_map_nests.map_nests[item]

        node_mapping = {}
        for node in map_nest.nodes():
            node_ = copy.deepcopy(node)
            state.add_node(node_)

            node_mapping[node] = node_

        for edge in map_nest.edges():
            state.add_edge(
                node_mapping[edge.src],
                edge.src_conn,
                node_mapping[edge.dst],
                edge.dst_conn,
                copy.deepcopy(edge.data),
            )

    def _generate_schedule_device(
        self, sdfg: dace.SDFG, state: dace.SDFGState, item: dace.nodes.MapEntry
    ) -> None:
        map_nest = self._graph_of_map_nests.map_nests[item]

        node_mapping = {}
        for node in map_nest.nodes():
            node_ = copy.deepcopy(node)
            if (
                isinstance(node, dace.nodes.AccessNode)
                and map_nest.entry_node(node) is None
            ):
                node_.data = "device_" + node.data
            elif isinstance(node_, dace.nodes.MapEntry):
                node_.map.schedule = dace.ScheduleType.GPU_Device

            state.add_node(node_)

            node_mapping[node] = node_

        for edge in map_nest.edges():
            memlet = copy.deepcopy(edge.data)
            if memlet.data is not None and "device_" + memlet.data in sdfg.arrays:
                memlet.data = "device_" + memlet.data

            state.add_edge(
                node_mapping[edge.src],
                edge.src_conn,
                node_mapping[edge.dst],
                edge.dst_conn,
                memlet,
            )

    def _generate_copy_host_to_device(
        self, sdfg: dace.SDFG, state: dace.SDFGState, item: str
    ) -> None:
        host_array = item
        host_desc = sdfg.arrays[host_array]
        host_node = state.add_access(host_array)

        device_array = "device_" + item
        if device_array not in sdfg.arrays:
            device_desc = copy.deepcopy(host_desc)
            device_desc.storage = dace.StorageType.GPU_Global
            device_desc.transient = True
            sdfg.add_datadesc(device_array, device_desc)
        else:
            device_desc = sdfg.arrays[device_array]

        device_node = state.add_access(device_array)

        state.add_edge(
            host_node,
            None,
            device_node,
            None,
            dace.Memlet.from_array(host_array, host_desc),
        )

    def _generate_copy_device_to_host(
        self, sdfg: dace.SDFG, state: dace.SDFGState, item: str
    ) -> None:
        host_array = item
        host_desc = sdfg.arrays[host_array]
        host_node = state.add_access(host_array)

        device_array = "device_" + item
        assert device_array in sdfg.arrays
        device_desc = sdfg.arrays[device_array]
        device_node = state.add_access(device_array)

        state.add_edge(
            device_node,
            None,
            host_node,
            None,
            dace.Memlet.from_array(device_array, device_desc),
        )
