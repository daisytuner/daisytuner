# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import copy

from typing import Dict, List, Tuple

from p_tqdm import p_map
from torch_geometric.data import Batch

from dace.sdfg.graph import OrderedMultiDiGraph
from dace.sdfg.state import StateSubgraphView
from dace.transformation.optimizer import Optimizer
from dace.transformation.dataflow import OTFMapFusion
from dace.transformation.subgraph import SubgraphFusion

from daisytuner.analysis.similarity.map_nest_model import MapNest, MapNestModel
from daisytuner.analysis.similarity.benchmarking import CPUBenchmark, GPUBenchmark
from daisytuner.transfer_tuning import TransferTuner

from daisytuner.device_mapping.action import Action
from daisytuner.device_mapping.state.storage_location import StorageLocation
from daisytuner.device_mapping.state.invalid_schedule_exception import (
    InvalidScheduleException,
)


class GraphOfMaps(OrderedMultiDiGraph):
    def __init__(
        self,
        state: dace.SDFGState,
        cpu_benchmark: CPUBenchmark,
        gpu_benchmark: GPUBenchmark,
        host_model: MapNestModel,
        device_model: MapNestModel,
        transfer_tuner: TransferTuner,
    ) -> None:
        super().__init__()
        self._state = state
        self._host_model = host_model
        self._device_model = device_model
        self._transfer_tuner = transfer_tuner
        self._cpu_benchmark = cpu_benchmark
        self._gpu_benchmark = gpu_benchmark

        # Scheduling structures
        self._frozen = False
        self._visited = set()
        self._array_table: Dict[str, StorageLocation] = None
        self._last_actions = []
        self._fusable_maps = {}
        self._map_nest_schedules = {}

        self._init_graph()
        self._init_map_nest_schedules()
        self._init_fusable_maps()

    @property
    def state(self) -> dace.SDFGState:
        return self._state

    @property
    def map_nests(self) -> Dict[dace.nodes.MapEntry, MapNest]:
        return self._map_nests

    @property
    def array_table(self) -> Dict[str, StorageLocation]:
        return self._array_table

    @property
    def map_nest_schedules(self):
        return self._map_nest_schedules

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

    def free_host(self, array: str) -> None:
        assert not self.frozen()
        assert array in self._array_table

        if self._array_table[array] != StorageLocation.BOTH:
            raise InvalidScheduleException(f"Cannot free {array} on host")

        self._array_table[array] = StorageLocation.DEVICE

        # Bookkeeping
        self._last_actions.append((array, Action.FREE_HOST))

    def fuse_maps(self, maps: Tuple[dace.nodes.MapEntry, dace.nodes.MapEntry]) -> None:
        if not maps in self._fusable_maps:
            raise InvalidScheduleException(f"Cannot fuse {maps}")

        if isinstance(self._fusable_maps[maps], OTFMapFusion):
            self._fusable_maps[maps].apply(graph=self._state, sdfg=self._state.parent)
        else:
            self._fusable_maps[maps].apply(sdfg=self._state.parent)

        # Reset maps
        self._init_graph()
        self._init_map_nest_schedules()
        self._init_fusable_maps()

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
            elif action == Action.FREE_HOST:
                # TODO
                pass
            elif action == Action.FUSE_MAPS:
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

    def _init_graph(self):
        # Clear
        for edge in self.edges():
            self.remove_edge(edge)
        for node in self.nodes():
            self.remove_node(node)

        # 1. Nodes = map nests
        self._map_nests: Dict[dace.nodes.MapEntry, MapNest] = {}
        for node in self._state.nodes():
            if not isinstance(node, dace.nodes.MapEntry):
                continue
            if not self._state.entry_node(node) is None:
                continue

            map_nest = MapNest(state=self._state, root=node)
            self._map_nests[node] = map_nest
            self.add_node(node)

        # 2. Edges = data dependencies
        for node in self._map_nests:
            exit_node = self._state.exit_node(node)
            queue = list(self._state.out_edges(exit_node))
            while queue:
                edge = queue.pop(0)
                if isinstance(edge.dst, dace.nodes.MapEntry):
                    self.add_edge(node, edge.dst, data=None)
                else:
                    queue.extend(self._state.out_edges(edge.dst))

    def _init_map_nest_schedules(self):
        self._map_nest_schedules.clear()

        def preprocess(map_nest: MapNest):
            map_nest_encoding_cpu = self._host_model.preprocess(
                map_nest, self._cpu_benchmark
            )
            map_nest_encoding_gpu = self._device_model.preprocess(
                map_nest, self._gpu_benchmark
            )
            assert not map_nest_encoding_cpu.is_data_dependent
            return map_nest_encoding_cpu, map_nest_encoding_gpu

        # Create embeddings and runtime estimates
        all_map_nests = list(self._map_nests.values())
        if not all_map_nests:
            return

        # encodings = p_map(preprocess, all_map_nests)
        encodings = list(map(preprocess, all_map_nests))
        host_encodings, device_encodings = zip(*encodings)
        host_batch = Batch.from_data_list(host_encodings)
        device_batch = Batch.from_data_list(device_encodings)
        preds_host = self._host_model.predict_batch(host_batch)
        preds_device = self._device_model.predict_batch(device_batch)

        # Query best schedules
        host_schedules = self._transfer_tuner.predict(preds_host, dace.DeviceType.CPU)
        device_schedules = self._transfer_tuner.predict(
            preds_device, dace.DeviceType.GPU
        )

        # Store results in global dict
        self._map_nest_schedules = {}
        for i in range(len(all_map_nests)):
            map_nest = all_map_nests[i]
            runtime_host, embedding_host, node_embeddings_host = preds_host[i]
            runtime_device, embedding_device, node_embeddings_device = preds_device[i]
            schedule_host, speedup_host = host_schedules[i]
            schedule_device, speedup_device = device_schedules[i]

            self._map_nest_schedules[map_nest] = {
                "host": {
                    "runtime": runtime_host,
                    "embedding": embedding_host,
                    "node_embeddings": node_embeddings_host,
                    "best_schedule": schedule_host,
                    "speedup": speedup_host,
                },
                "device": {
                    "runtime": runtime_device,
                    "embedding": embedding_device,
                    "node_embeddings": node_embeddings_device,
                    "best_schedule": schedule_device,
                    "speedup": speedup_device,
                },
            }

    def _init_fusable_maps(self):
        self._fusable_maps.clear()

        # OTFMapFusion
        otf_fusable_maps = Optimizer(sdfg=self._state.parent).get_pattern_matches(
            states=[self._state], patterns=[OTFMapFusion]
        )
        for xform in otf_fusable_maps:
            first_map_entry = self._state.entry_node(xform.first_map_exit)
            second_map_entry = xform.second_map_entry

            if first_map_entry in self._visited:
                continue
            if second_map_entry in self._visited:
                continue

            self._fusable_maps[(first_map_entry, second_map_entry)] = xform

        # SubgraphFusion
        for map_entry, map_nest in self._map_nests.items():
            if map_entry in self._visited:
                continue

            for map_entry_, map_nest_ in self._map_nests.items():
                if map_entry_ in self._visited:
                    continue
                if map_entry == map_entry_:
                    continue

                nodes = set(map_nest.nodes()).union(map_nest_.nodes())
                subgraph_view = StateSubgraphView(self._state, nodes)
                xform = SubgraphFusion()
                xform.setup_match(
                    subgraph_view,
                    sdfg_id=self._state.parent.sdfg_id,
                    state_id=self._state.parent.node_id(self._state),
                )
                if xform.can_be_applied(self._state.parent, subgraph_view):
                    self._fusable_maps[(map_entry, map_entry_)] = xform
