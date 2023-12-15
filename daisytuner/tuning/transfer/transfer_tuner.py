# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import copy
import json
import dace
import math
import torch
import sympy
import copy
import requests
import numpy as np
import networkx as nx

from typing import Dict, List
from scipy.optimize import linear_sum_assignment

from dace.transformation import PatternTransformation
from dace.transformation.optimizer import Optimizer
from dace.transformation.dataflow import (
    MapExpansion,
    TrivialMapElimination,
    InLocalStorage,
    OutLocalStorage,
    AccumulateTransient,
)
from dace.transformation.auto.auto_optimize import make_transients_persistent


from dace.sdfg.state import StateSubgraphView
from dace.sdfg.analysis.cutout import SDFGCutout
from dace.frontend.operations import detect_reduction_type
from dace.transformation.optimizer import Optimizer

from daisytuner.analysis.similarity import MapNest
from daisytuner.model import DaisyNet
from daisytuner.profiling.measure import random_arguments, measure
from daisytuner.tuning.cutout_tuner import CutoutTuner

TT_API_URI = "https://tune-bhqsvyw3sa-uc.a.run.app/"


class TransferTuner(CutoutTuner):
    def __init__(self, topK: int = 3, use_profiling_features: bool = False) -> None:
        self._model = DaisyNet.create()
        self._topK = topK
        self._use_profiling_features = use_profiling_features

    def tune(
        self,
        loop_nest: MapNest,
        arguments: Dict = None,
    ) -> str:
        if arguments is None:
            arguments = random_arguments(loop_nest.cutout)

        # Benchmark
        initial_sdfg = loop_nest.cutout
        initial_runtime, initial_process_time, _ = measure(
            initial_sdfg, arguments=arguments, measurements=1
        )
        if initial_runtime == math.inf:
            return None

        nns = self._query_nearest_neighbors(
            loop_nest,
            topK=self._topK,
            use_profiling_features=self._use_profiling_features,
        )
        if not nns:
            return None

        best_neighbor = None
        best_schedule = None
        best_runtime = initial_runtime
        best_process_time = initial_process_time
        for nn in nns:
            schedule = json.loads(nn["metadata"]["tuning"]["schedule"])
            recom, applied_schedule = self.apply_schedule(
                loop_nest,
                schedule,
                in_place=False,
                use_profiling_features=self._use_profiling_features,
            )
            if not applied_schedule:
                continue

            recom.openmp_sections = False
            dace.sdfg.infer_types.infer_connector_types(recom)
            dace.sdfg.infer_types.set_default_schedule_and_storage_types(recom, None)
            make_transients_persistent(recom, dace.DeviceType.CPU)

            args = copy.deepcopy(arguments)

            try:
                runtime, process_time, _ = measure(
                    recom,
                    arguments=args,
                    timeout=best_process_time * 1.5,
                    measurements=1,
                )
            except:
                runtime = math.inf

            if runtime == math.inf:
                continue

            if best_runtime / runtime > 1.1:
                best_runtime = runtime
                best_process_time = process_time
                best_schedule = schedule
                best_neighbor = nn

        if best_schedule is not None:
            _, applied_schedule = self.apply_schedule(
                loop_nest,
                best_schedule,
                in_place=True,
                use_profiling_features=self._use_profiling_features,
            )
        else:
            applied_schedule = []

        return applied_schedule, best_neighbor

    def _query_nearest_neighbors(
        self, loop_nest: MapNest, topK: int, use_profiling_features: bool
    ) -> Dict:
        if use_profiling_features:
            emb, _, _, _, runner = self._model.predict(
                loop_nest=loop_nest, use_profiling_features=use_profiling_features
            )
            runner = runner.tolist()
        else:
            runner = "__static"
            emb, _ = self._model.predict(
                loop_nest=loop_nest, use_profiling_features=use_profiling_features
            )

        outermost_map = next(nx.topological_sort(loop_nest.tree))
        payload = {
            "type": "cpu",
            "runner": runner,
            "embedding": emb.tolist(),
            "outermost_dims": len(outermost_map.map.params),
            "topK": topK,
            "collections": ["polybench"],
        }
        res = requests.post(TT_API_URI, json=payload)
        if res.status_code != requests.codes.ok:
            raise ValueError("TT request failed")

        return res.json()

    @torch.inference_mode()
    def apply_schedule(
        self,
        loop_nest: MapNest,
        schedule: List[Dict],
        in_place: bool,
        use_profiling_features: bool,
    ) -> dace.SDFG:
        if in_place:
            sdfg = loop_nest.sdfg
            subgraph_view = loop_nest.view
        else:
            schedule = copy.deepcopy(schedule)
            sdfg = copy.deepcopy(loop_nest.cutout)
            subgraph_view = StateSubgraphView(
                sdfg.start_state, sdfg.start_state.nodes()
            )

        subgraph_view = TransferTuner._preprocess(sdfg, subgraph_view)

        applied_schedule = []
        in_local_buffers = set()
        out_local_buffers = set()
        in_strides, out_strides = TransferTuner._analyze_accesses(loop_nest)
        for trans in schedule:
            # Create a temporary SDFG to compute node embeddings on subgraph-only
            cutout_ = SDFGCutout.singlestate_cutout(
                subgraph_view.graph,
                *subgraph_view.nodes(),
                make_copy=False,
                make_side_effects_global=False,
                use_alibi_nodes=False
            )
            cutout_.specialize(sdfg.constants)
            map_entry = None
            for node in cutout_.start_state.nodes():
                if (
                    isinstance(node, dace.nodes.MapEntry)
                    and cutout_.start_state.entry_node(node) is None
                ):
                    map_entry = node
                    break

            loop_nest_ = MapNest.create(
                cutout_,
                cutout_.start_state,
                map_entry=map_entry,
                build_folder=loop_nest.cutout.build_folder,
            )

            # Compute node embeddings
            if use_profiling_features:
                _, node_embeddings, *_ = self._model.predict(
                    loop_nest_, use_profiling_features=use_profiling_features
                )
            else:
                _, node_embeddings = self._model.predict(
                    loop_nest_, use_profiling_features=use_profiling_features
                )

            # Subgraph matching of pattern nodes to actual nodes
            matching = self._subgraph_matching(
                cutout_,
                trans["_subgraph"],
                node_embeddings,
            )
            if matching is None:
                continue

            # Convert node ids back to original SDFG
            trans["_subgraph"] = {}
            for pattern_node, node_id in matching.items():
                node = cutout_.start_state.node(node_id)
                trans["_subgraph"][pattern_node] = subgraph_view.node_id(node)

            # Replace sdfg-specific options
            if not self._find_options(
                sdfg,
                subgraph_view,
                trans,
                in_local_buffers,
                out_local_buffers,
                in_strides,
                out_strides,
            ):
                continue

            # Parse transformation
            xform = PatternTransformation.from_json(trans)
            xform._sdfg = sdfg
            xform.state_id = sdfg.node_id(subgraph_view.graph)

            if xform.can_be_applied(subgraph_view.graph, sdfg=sdfg, expr_index=0):
                xform.apply(subgraph_view.graph, sdfg)
                applied_schedule.append(xform.to_json())

                # Update state of matching
                if isinstance(xform, InLocalStorage):
                    in_local_buffers.add(xform.array)
                elif isinstance(xform, OutLocalStorage):
                    out_local_buffers.add(xform.array)
                elif isinstance(xform, AccumulateTransient):
                    out_local_buffers.add(xform.array)

                subgraph_view = update_subgraph_view(sdfg, subgraph_view)
            else:
                pass

        return sdfg, applied_schedule

    def _subgraph_matching(
        self,
        sdfg: dace.SDFG,
        pattern_subgraph: Dict,
        node_embeddings: Dict[dace.nodes.Node, np.ndarray],
    ) -> Dict:
        pattern_nodes = list(pattern_subgraph.keys())
        elements = list(
            filter(lambda e: isinstance(e, dace.nodes.Node), node_embeddings.keys())
        )

        cost_matrix = np.zeros((len(pattern_nodes), len(elements)))
        for i in range(cost_matrix.shape[0]):
            pattern_node = pattern_subgraph[pattern_nodes[i]]

            for j in range(cost_matrix.shape[1]):
                element = elements[j]
                element_desc = TransferTuner._element_description(
                    sdfg.start_state, element
                )
                element_embedding = node_embeddings[element]
                if element_desc["type"] == pattern_node["type"]:
                    if element_desc["scope_levels"] == pattern_node["scope_levels"]:
                        cost_matrix[i, j] = np.linalg.norm(
                            (
                                np.array(pattern_node["node_embedding"])
                                - element_embedding
                            ),
                            ord=2.0,
                        )
                        continue

                cost_matrix[i, j] = np.inf

        # Bi-partite matching between pattern node embeddings and node embeddings
        try:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        except:
            return None

        subgraph = {}
        for i in range(len(row_ind)):
            pattern_node = pattern_nodes[row_ind[i]]
            element = elements[col_ind[i]]
            element_id = sdfg.start_state.node_id(element)
            subgraph[pattern_node] = element_id

        if len(subgraph) != len(pattern_nodes):
            return None

        return subgraph

    def _find_options(
        self,
        sdfg: dace.SDFG,
        subgraph_view: StateSubgraphView,
        trans: Dict,
        in_local_buffers: set,
        out_local_buffers: set,
        in_strides: Dict,
        out_strides: Dict,
    ) -> None:
        subgraph = trans["_subgraph"]
        transformation = trans["transformation"]

        if transformation == "MapDimShuffle":
            map_entry = subgraph_view.graph.node(subgraph["0"])
            map_params = [map_entry.map.params[i] for i in trans["parameters"]]
            trans["parameters"] = map_params
        elif transformation == "MapSchedule":
            trans["unroll"] = False
        elif transformation == "StripMining":
            map_entry = subgraph_view.graph.node(subgraph["0"])
            dim_idx = trans["dim_idx"]
            tile_size = int(trans["tile_size"])
            start, stop, step = map_entry.map.range[dim_idx]
            try:
                map_extend = dace.symbolic.int_floor((stop + 1 - start), step)
                map_extend = dace.symbolic.evaluate(map_extend, symbols=sdfg.constants)
                divides_evenly = map_extend / tile_size
                trans["divides_evenly"] = divides_evenly.is_integer
            except:
                trans["divides_evenly"] = False
        elif transformation == "MapTiling":
            map_entry = subgraph_view.graph.node(subgraph["0"])
            tile_sizes = trans["tile_sizes"]
            divides_evenly = True
            for i, (start, stop, step) in enumerate(map_entry.map.range):
                try:
                    map_extend = dace.symbolic.int_floor((stop + 1 - start), step)
                    map_extend = dace.symbolic.evaluate(
                        map_extend, symbols=sdfg.constants
                    )
                    divisor = map_extend / int(tile_sizes[i])
                    divides_evenly = divides_evenly and divisor.is_integer
                except:
                    divides_evenly = False
            trans["divides_evenly"] = divides_evenly
            trans["prefix"] = "tile"
            trans["tile_trivial"] = True
        elif transformation == "Vectorization":
            map_entry = subgraph_view.graph.node(subgraph["0"])
            start, stop, step = map_entry.map.range[-1]
            try:
                map_extend = dace.symbolic.int_floor((stop + 1 - start), step)
                map_extend = dace.symbolic.evaluate(map_extend, symbols=sdfg.constants)
                divisor = map_extend / int(trans["vector_len"])
                divides_evenly = divisor.is_integer
            except:
                divides_evenly = False
            trans["preamble"] = False
            trans["postamble"] = not divides_evenly

            tasklet = next(subgraph_view.graph.out_edges(map_entry).__iter__()).dst
            if not isinstance(tasklet, dace.nodes.Tasklet):
                return False

            code = tasklet.code.as_string
            if "min" in code or "max" in code or "sqrt" in code:
                return False

            wcr_edge = next(subgraph_view.graph.out_edges(tasklet).__iter__())
            memlet: dace.Memlet = wcr_edge.data
            if memlet.wcr is not None and ("min" in memlet.wcr or "max" in memlet.wcr):
                return False
        elif transformation == "InLocalStorage":
            first_map_entry = subgraph_view.graph.node(subgraph["0"])
            second_map_entry = subgraph_view.graph.node(subgraph["1"])

            # Potential arrays
            options = set(
                [
                    edge.data.data
                    for edge in subgraph_view.graph.edges_between(
                        first_map_entry, second_map_entry
                    )
                    if edge.data is not None
                    and edge.data.data is not None
                    and edge.data.wcr is None
                ]
            )

            # Arrays sorted by stride of innermost map
            strides = sorted(in_strides.items(), key=lambda item: item[1], reverse=True)

            # Pick from options according to strides
            array = None
            for option, _ in strides:
                if option not in options or option in in_local_buffers:
                    continue

                array = option
                break

            if array is None:
                # Try again with a second buffer
                for option, _ in strides:
                    if option not in options:
                        continue

                    array = option
                    break

                if array is None:
                    return False

            trans["array"] = array
        elif transformation == "OutLocalStorage":
            first_map_exit = subgraph_view.graph.node(subgraph["0"])
            second_map_exit = subgraph_view.graph.node(subgraph["1"])

            # Potential arrays
            options = set(
                [
                    edge.data.data
                    for edge in subgraph_view.graph.edges_between(
                        first_map_exit, second_map_exit
                    )
                    if edge.data is not None
                    and edge.data.data is not None
                    and edge.data.wcr is None
                ]
            )

            # Arrays sorted by stride of innermost map
            strides = sorted(
                out_strides.items(), key=lambda item: item[1], reverse=True
            )

            # Pick from options according to strides
            array = None
            for option, _ in strides:
                if option not in options or option in out_local_buffers:
                    continue

                array = option
                break

            if array is None:
                # Try again with a second buffer
                for option, _ in strides:
                    if option not in options:
                        continue

                    array = option
                    break

                if array is None:
                    return False

            trans["array"] = array
        elif transformation == "AccumulateTransient":
            first_map_exit = subgraph_view.graph.node(subgraph["0"])
            second_map_exit = subgraph_view.graph.node(subgraph["1"])

            # Heuristic: largest strided access
            edges = [
                (e, e.data.get_stride(sdfg, first_map_exit.map, dim=-1))
                for e in subgraph_view.graph.edges_between(
                    first_map_exit, second_map_exit
                )
                if e.data.data is not None and e.data.wcr is not None
            ]
            edges = sorted(
                edges,
                key=lambda item: int(
                    dace.symbolic.evaluate(item[1], symbols=sdfg.constants)
                ),
                reverse=True,
            )
            if not edges or edges[0][1] >= 1:
                return False

            trans["array"] = edges[0][0].data.data

            edge = edges[0][0]
            reduction_type = detect_reduction_type(edge.data.wcr)
            if reduction_type == dace.dtypes.ReductionType.Custom:
                trans["identity"] = None
            else:
                dtype = sdfg.arrays[trans["array"]].dtype
                identity = dace.dtypes.reduction_identity(dtype, reduction_type)
                trans["identity"] = identity

        return True

    @staticmethod
    def _analyze_accesses(loop_nest: MapNest) -> Dict:
        in_arrays = {}
        out_arrays = {}
        for dnode in loop_nest.view.data_nodes():
            if loop_nest.view.entry_node(dnode) is not None:
                continue

            if loop_nest.view.in_degree(dnode) == 0:
                in_arrays[dnode.data] = 0
            elif loop_nest.view.out_degree(dnode) == 0:
                out_arrays[dnode.data] = 0

        innermost_maps = (
            node for node, out_degree in loop_nest.tree.out_degree() if out_degree == 0
        )
        for innermost_map in innermost_maps:
            for edge in loop_nest.view.out_edges(innermost_map):
                if edge.data.data is None:
                    continue

                stride = edge.data.get_stride(loop_nest.sdfg, innermost_map.map, dim=-1)
                # Overapproximate
                stride = stride.replace(sympy.Max, sympy.Add)
                stride = stride.replace(sympy.Min, sympy.Add)

                stride = int(
                    dace.symbolic.evaluate(stride, symbols=loop_nest.sdfg.constants)
                )
                if stride == 0:
                    stride = math.inf

                if in_arrays[edge.data.data] < stride:
                    in_arrays[edge.data.data] = stride

        for innermost_map in innermost_maps:
            for edge in loop_nest.view.in_edges(
                loop_nest.view.exit_node(innermost_map)
            ):
                if edge.data.data is None:
                    continue

                stride = edge.data.get_stride(loop_nest.sdfg, innermost_map.map, dim=-1)
                stride = int(
                    dace.symbolic.evaluate(stride, symbols=loop_nest.sdfg.constants)
                )
                if stride == 0:
                    stride = math.inf

                if out_arrays[edge.data.data] < stride:
                    out_arrays[edge.data.data] = stride

        return in_arrays, out_arrays

    @staticmethod
    def _element_description(state: dace.SDFGState, element: dace.nodes.Node) -> Dict:
        desc = element.to_json(state)

        scope_levels = 0
        if "scope_entry" in desc:
            scope_entry = desc["scope_entry"]
            while not scope_entry is None:
                scope_levels += 1

                scope = state.node(int(scope_entry))
                scope_entry = scope.to_json(state)["scope_entry"]

        cs = {"type": desc["type"], "scope_levels": scope_levels}
        return cs

    @staticmethod
    def _preprocess(
        sdfg: dace.SDFG, subgraph_view: StateSubgraphView
    ) -> StateSubgraphView:
        for node in subgraph_view.nodes():
            if not isinstance(node, dace.nodes.MapEntry):
                continue

            node.schedule = dace.ScheduleType.Sequential
            node.collapse = 1

        subgraph_view = TransferTuner._apply_transformations_repeated_on_subgraph(
            MapExpansion, sdfg, subgraph_view
        )

        return subgraph_view

    @staticmethod
    def _apply_transformations_repeated_on_subgraph(
        transformation: PatternTransformation,
        sdfg: dace.SDFG,
        subgraph_view: StateSubgraphView,
        options: Dict = None,
    ) -> StateSubgraphView:
        while True:
            xforms = Optimizer(sdfg).get_pattern_matches(
                patterns=(transformation,), options=options
            )
            applied = False
            for xform in xforms:
                # check same state
                if (
                    xform._sdfg != sdfg
                    or sdfg.node(xform.state_id) != subgraph_view.graph
                ):
                    continue

                contained = True
                for _, node_id in xform.subgraph.items():
                    if subgraph_view.graph.node(node_id) not in subgraph_view.nodes():
                        contained = False
                        break

                if not contained:
                    continue

                xform.apply(subgraph_view.graph, sdfg)
                applied = True
                break

            if not applied:
                break

        return update_subgraph_view(sdfg, subgraph_view)


def update_subgraph_view(
    sdfg: dace.SDFG, subgraph_view: StateSubgraphView
) -> StateSubgraphView:
    # TODO: Find better solution

    state: dace.SDFGState = subgraph_view.graph
    map_entry = None
    for node in subgraph_view.nodes():
        if node not in state.nodes():
            continue

        if isinstance(node, (dace.nodes.MapExit, dace.nodes.AccessNode)):
            continue

        map_entry = node
        break

    if map_entry is None:
        return None

    # Assumption, working on top-level maps only
    while state.entry_node(map_entry) is not None:
        map_entry = state.entry_node(map_entry)

    map_exit = state.exit_node(map_entry)
    subgraph_nodes = set(state.all_nodes_between(map_entry, map_exit))
    subgraph_nodes.add(map_entry)
    subgraph_nodes.add(map_exit)

    for edge in state.in_edges(map_entry):
        subgraph_nodes.add(edge.src)
    for edge in state.out_edges(map_exit):
        subgraph_nodes.add(edge.dst)

    subgraph_nodes = list(subgraph_nodes)

    view: StateSubgraphView = StateSubgraphView(state, subgraph_nodes)
    return view
