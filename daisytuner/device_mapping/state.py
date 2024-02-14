# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import copy
import dace
import networkx as nx

from typing import List, Dict, Union

from daisytuner.analysis.gom.graph_of_maps import GraphOfMaps, BarrierNode
from daisytuner.device_mapping.action import Action


class InvalidStateException(Exception):
    pass


class State:
    def __init__(self, sdfg: dace.SDFG) -> None:
        self._sdfg: dace.SDFG = sdfg

        # Schedule to be constructed
        self._schedule = []
        self._scheduled_maps = set()

        # Set up arrays
        self._arrays = list(sdfg.arrays)
        self._array_table = {arr: dace.DeviceType.CPU for arr in sdfg.arrays}
        self._selected_array = self._arrays[0]

        ##### Construction Graph of Map Nests #####
        self._graph_of_map_nests = GraphOfMaps(self._sdfg)

        # Define active maps
        self._active_maps = []
        for node in self._graph_of_map_nests.nodes():
            if self._graph_of_map_nests.in_degree(node) == 0:
                self._active_maps.append(node)

        self._selected_map_nest = self._active_maps[0]

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
    def selected_map_nest(self) -> dace.nodes.MapEntry:
        return self._selected_map_nest

    def terminated(self) -> bool:
        return self._selected_map_nest == None

    def _next_array(self) -> str:
        index = self._arrays.index(self._selected_array)
        self._selected_array = self._arrays[(index + 1) % len(self._arrays)]
        return self._selected_array

    def _next_map_nest(self) -> dace.nodes.MapEntry:
        index = self._active_maps.index(self._selected_map_nest)
        self._selected_map_nest = self._active_maps[
            (index + 1) % len(self._active_maps)
        ]
        return self._selected_map_nest

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

    def _schedule_host(self, map_nest: dace.nodes.MapEntry) -> None:
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

    def _update_active(self, map_nest: dace.nodes.MapEntry) -> None:
        self._selected_map_nest = None
        self._scheduled_maps.add(map_nest)
        self._active_maps.remove(map_nest)

        for succ in self._graph_of_map_nests.successors(map_nest):
            if all(
                [
                    pred in self._scheduled_maps
                    for pred in self._graph_of_map_nests.predecessors(succ)
                ]
            ):
                if isinstance(succ, BarrierNode):
                    self._scheduled_maps.add(succ)
                    for succ_ in self._graph_of_map_nests.successors(succ):
                        if succ_ not in self._active_maps:
                            self._active_maps.append(succ_)
                else:
                    if succ not in self._active_maps:
                        self._active_maps.append(succ)

        if self._active_maps:
            self._selected_map_nest = self._active_maps[0]

    def update(self, action: Action) -> None:
        if action == Action.SCHEDULE_HOST:
            self._schedule_host(self._selected_map_nest)
        elif action == Action.SCHEDULE_DEVICE:
            self._schedule_device(self._selected_map_nest)
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

        # Add initial state
        action, item = self._schedule.pop(0)
        init_state = scheduled_sdfg.add_state("action_0", is_start_block=True)
        self._generate_state(scheduled_sdfg, init_state, action=action, item=item)

        # Generate chain of states
        last_state = init_state
        for i, (action, item) in enumerate(self._schedule):
            state = scheduled_sdfg.add_state(f"action_{i+1}", is_start_block=False)
            self._generate_state(scheduled_sdfg, state, action=action, item=item)

            scheduled_sdfg.add_edge(last_state, state, dace.InterstateEdge())
            last_state = state

        device_arrays = [
            k
            for k, v in self._array_table.items()
            if v == dace.DeviceType.GPU and not self._sdfg.arrays[k].transient
        ]
        if device_arrays:
            copy_back_state = scheduled_sdfg.add_state("copy_back_state")
            scheduled_sdfg.add_edge(last_state, copy_back_state, dace.InterstateEdge())
            for arr in device_arrays:
                self._generate_copy_device_to_host(scheduled_sdfg, copy_back_state, arr)

        # Postprocessing
        # scheduled_sdfg.simplify()
        scheduled_sdfg.validate()
        return scheduled_sdfg

    def _generate_state(
        self,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        action: Action,
        item: Union[dace.nodes.MapEntry, str],
    ) -> None:
        if action == Action.SCHEDULE_HOST:
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
