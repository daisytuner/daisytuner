# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import copy

from typing import Dict, List

from dace.sdfg.graph import OrderedMultiDiGraph

from daisytuner.analysis.similarity.map_nest import MapNest

from daisytuner.device_mapping.action import Action
from daisytuner.device_mapping.state.storage_location import StorageLocation
from daisytuner.device_mapping.state.invalid_schedule_exception import (
    InvalidScheduleException,
)


class GraphOfMaps(OrderedMultiDiGraph):
    def __init__(self, state: dace.SDFGState) -> None:
        super().__init__()
        self._state = state

        # 1. Nodes = map nests
        self._map_nests: Dict[dace.nodes.MapEntry, MapNest] = {}
        for node in state.nodes():
            if not isinstance(node, dace.nodes.MapEntry):
                continue
            if not state.entry_node(node) is None:
                continue

            map_nest = MapNest(state=state, root=node)
            self._map_nests[node] = map_nest
            self.add_node(node)

        # 2. Edges = data dependencies
        for node in self._map_nests:
            exit_node = state.exit_node(node)
            for oedge in state.out_edges(exit_node):
                assert isinstance(oedge.dst, dace.nodes.AccessNode)
                for oedge_ in state.out_edges(oedge.dst):
                    assert isinstance(oedge_.dst, dace.nodes.MapEntry)
                    self.add_edge(node, oedge_.dst, data=None)

        # Scheduling structures
        self._frozen = False
        self._visited = set()
        self._array_table: Dict[str, StorageLocation] = None
        self._last_actions = []

    @property
    def state(self) -> dace.SDFGState:
        return self._state

    @property
    def map_nests(self) -> Dict[dace.nodes.MapEntry, MapNest]:
        return self._map_nests

    @property
    def array_table(self) -> Dict[str, StorageLocation]:
        return self._array_table

    def init(self, array_table: Dict[str, StorageLocation]) -> None:
        assert self._array_table is None
        self._array_table = copy.deepcopy(array_table)

    def frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> None:
        assert not self._frozen
        self._frozen = True

    def schedule_map_nest_host(self, node: dace.nodes.MapEntry) -> None:
        assert not self.frozen()
        assert node in self._map_nests

        # Check if scheduling of map is valid
        map_nest = self._map_nests[node]
        for inp in map_nest.inputs():
            location = self._array_table[inp.data]
            if not location.is_host():
                raise InvalidScheduleException(f"Dependencies not fulfilled for {node}")

        for outp in map_nest.outputs():
            location = self._array_table[outp.data]
            if not location.is_host():
                raise InvalidScheduleException(f"Dependencies not fulfilled for {node}")

        # Written array are now exclusive to schedule device
        for outp in map_nest.outputs():
            self._array_table[outp.data] = StorageLocation.HOST

        # Bookkeeping
        self._visited.add(node)
        self._last_actions.append((node, Action.SCHEDULE_MAP_NEST_HOST))

    def schedule_map_nest_device(self, node: dace.nodes.MapEntry) -> None:
        assert not self.frozen()
        assert node in self._map_nests

        # Check if scheduling of map is valid
        map_nest = self._map_nests[node]
        for inp in map_nest.inputs():
            location = self._array_table[inp.data]
            if not location.is_device():
                raise InvalidScheduleException(f"Dependencies not fulfilled for {node}")

        for outp in map_nest.outputs():
            location = self._array_table[outp.data]
            if not location.is_device():
                raise InvalidScheduleException(f"Dependencies not fulfilled for {node}")

        # Written array are now exclusive to schedule device
        for outp in map_nest.outputs():
            self._array_table[outp.data] = StorageLocation.DEVICE

        # Bookkeeping
        self._visited.add(node)
        self._last_actions.append((node, Action.SCHEDULE_MAP_NEST_DEVICE))

    def copy_host_to_device(self, array: str) -> None:
        assert not self.frozen()
        assert array in self._array_table

        if self._array_table[array].is_device():
            raise InvalidScheduleException(f"{array} already on device")

        self._array_table[array] = StorageLocation.BOTH

        # Bookkeeping
        self._last_actions.append((array, Action.COPY_HOST_TO_DEVICE))

    def copy_device_to_host(self, array: str) -> None:
        assert not self.frozen()
        assert array in self._array_table

        if self._array_table[array].is_host():
            raise InvalidScheduleException(f"{array} already on host")

        self._array_table[array] = StorageLocation.BOTH

        # Bookkeeping
        self._last_actions.append((array, Action.COPY_DEVICE_TO_HOST))

    def free_device(self, array: str) -> None:
        assert not self.frozen()
        assert array in self._array_table

        if self._array_table[array] != StorageLocation.BOTH:
            raise InvalidScheduleException(f"Cannot free {array} on device")

        self._array_table[array] = StorageLocation.HOST

        # Bookkeeping
        self._last_actions.append((array, Action.FREE_DEVICE))

    def active(self) -> List[dace.nodes.MapEntry]:
        active_nodes = []
        for node in self.nodes():
            if node in self._visited:
                continue

            deps = self.in_edges(node)

            active = True
            for dep in deps:
                if dep.src not in self._visited:
                    active = False
                    break

            if active:
                active_nodes.append(node)

        return active_nodes

    def generate(
        self, sdfg: dace.SDFG, start: dace.SDFGState, end: dace.SDFGState
    ) -> None:
        state_action_mapping = {}
        last_state = start
        for i, (item, action) in enumerate(self._last_actions):
            state = sdfg.add_state(self._state.name + f"_{i}")

            if action == Action.SCHEDULE_MAP_NEST_HOST:
                self._generate_schedule_host(sdfg, state, item)
            elif action == Action.SCHEDULE_MAP_NEST_DEVICE:
                self._generate_schedule_device(sdfg, state, item)
            elif action == Action.COPY_HOST_TO_DEVICE:
                self._generate_copy_host_to_device(sdfg, state, item)
            elif action == Action.COPY_DEVICE_TO_HOST:
                self._generate_copy_device_to_host(sdfg, state, item)
            elif action == Action.FREE_DEVICE:
                # TODO
                pass
            else:
                raise ValueError(f"Invalid action {item}")

            sdfg.add_edge(last_state, state, dace.InterstateEdge())
            last_state = state
            state_action_mapping[state] = action

        sdfg.add_edge(last_state, end, dace.InterstateEdge())

        return state_action_mapping

    def _generate_schedule_host(
        self, sdfg: dace.SDFG, state: dace.SDFGState, item: dace.nodes.MapEntry
    ) -> None:
        map_nest = self._map_nests[item]

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
        map_nest = self._map_nests[item]

        node_mapping = {}
        for node in map_nest.nodes():
            node_ = copy.deepcopy(node)
            if (
                isinstance(node, dace.nodes.AccessNode)
                and map_nest.entry_node(node) is None
            ):
                node_.data = "device_" + node.data
            elif isinstance(node_, dace.nodes.MapEntry):
                if map_nest.entry_node(node) is None:
                    node_.map.schedule = dace.ScheduleType.GPU_Device
                else:
                    node_.map.schedule = dace.ScheduleType.Sequential

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
