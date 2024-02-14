# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import ast
import copy
import dace
import itertools

from typing import List, Dict, Tuple
from collections import OrderedDict

from dace.transformation.dataflow import (
    MapInterchange,
    MapExpansion,
    MapTiling,
    InLocalStorage,
    OutLocalStorage,
    Vectorization,
    AccumulateTransient,
    TrivialMapElimination,
)
from dace.frontend.operations import detect_reduction_type

from daisytuner.transformations import MapSchedule


class ScheduleSpace:

    _TILE_SIZES = [1, 2, 4, 6, 8, 12, 13, 16, 24, 26, 32, 64, 128, 256]
    _VECTOR_SIZES = [1, 2, 4]
    _OMP_CHUNK_SIZES = [4, 8, 16, 32, 64]

    def __init__(self, cutout: dace.SDFG, extra_tile_sizes: List[int] = None) -> None:
        # Preprocessing
        self._cutout = copy.deepcopy(cutout)

        # SDFG-specific tile sizes
        self._tile_sizes = set(ScheduleSpace._TILE_SIZES)
        if extra_tile_sizes is not None:
            self._tile_sizes = self._tile_sizes.union(set(extra_tile_sizes))
        self._tile_sizes = sorted(list(self._tile_sizes))

    def enumerate(self, state: List[int] = None) -> Tuple[dace.SDFG, List[int], str]:
        if state is None:
            state = [0, 0, 0, 0, 0, 0]

        for i, first_level_tiling in enumerate(self._first_level_tilings(self._cutout)):
            if i < state[0]:
                continue

            for j, second_level_tiling in enumerate(
                self._second_level_tilings(self._cutout, first_level_tiling)
            ):
                if j < state[1]:
                    continue

                # Apply: Tiling
                tiled_cutout = copy.deepcopy(self._cutout)
                ScheduleSpace._apply_tilings(
                    tiled_cutout, (first_level_tiling, second_level_tiling)
                )

                for k, permutation in enumerate(self._permutations(tiled_cutout)):
                    if k < state[2]:
                        continue

                    # Apply: Permutation
                    permuted_cutout = copy.deepcopy(tiled_cutout)
                    ScheduleSpace._apply_permutation(permuted_cutout, permutation)

                    for l, parallelization in enumerate(
                        self._parallelizations(permuted_cutout)
                    ):
                        if l < state[3]:
                            continue

                        # Apply: Parallelization
                        parallelized_cutout = copy.deepcopy(permuted_cutout)
                        ScheduleSpace._apply_parallelization(
                            parallelized_cutout, parallelization
                        )

                        for m, vector_length in enumerate(
                            self._vectorizations(parallelized_cutout)
                        ):
                            if m < state[4]:
                                continue

                            # Apply: Vectorization
                            vectorized_cutout = copy.deepcopy(parallelized_cutout)
                            ScheduleSpace._apply_vectorization(
                                vectorized_cutout, vector_length
                            )

                            for n, local_storage in enumerate(
                                self._local_storages(vectorized_cutout)
                            ):
                                if n < state[5]:
                                    continue

                                # Apply: Local Storage
                                loc_cutout = copy.deepcopy(vectorized_cutout)
                                ScheduleSpace._apply_local_storage(
                                    loc_cutout, local_storage
                                )

                                try:
                                    loc_cutout.validate()
                                except:
                                    continue

                                state = [i, j, k, l, m, n]
                                desc = f"{first_level_tiling}#{second_level_tiling}#{permutation}#{parallelization}#{vector_length}#{local_storage}"
                                yield loc_cutout, copy.copy(state), desc

                            state[5] = 0
                        state[4] = 0
                    state[3] = 0
                state[2] = 0
            state[1] = 0
        state[0] = 0

    def _first_level_tilings(self, cutout: dace.SDFG) -> Dict[str, int]:
        params, _ = _maps(cutout)

        all_strategies = itertools.product(self._tile_sizes, repeat=len(params))
        first_level_strategy = {}
        for strategy in all_strategies:
            for i, param in enumerate(params):
                first_level_strategy[param] = strategy[i]

            yield first_level_strategy

            first_level_strategy.clear()

    def _second_level_tilings(
        self, cutout: dace.SDFG, first_level_tiling: Dict[str, int]
    ) -> Dict[str, int]:
        params = [
            "tile_" + param
            for param in first_level_tiling
            if first_level_tiling[param] > 1
        ]

        all_strategies = itertools.product(self._tile_sizes, repeat=len(params))
        second_level_strategy = {}
        for strategy in all_strategies:
            for i, param in enumerate(params):
                second_level_strategy[param] = strategy[i]

            yield second_level_strategy

            second_level_strategy.clear()

    def _permutations(self, cutout: dace.SDFG) -> List[str]:
        params, _ = _maps(cutout)

        begins = {}
        ends = {}
        for param, map_entry in params.items():
            b, e, _ = map_entry.map.range[0]
            begins[param] = set([str(sym) for sym in b.free_symbols])
            if isinstance(e, dace.symbolic.SymExpr):
                e = e.approx

            ends[param] = set([str(sym) for sym in e.free_symbols])

        for perm in itertools.permutations(list(params.keys())):
            valid = True
            for i, param in enumerate(perm[:-1]):
                lower_params = set(perm[i + 1 :])
                if not lower_params.isdisjoint(begins[param]):
                    valid = False
                    break
                if not lower_params.isdisjoint(ends[param]):
                    valid = False
                    break

            if not valid:
                continue

            yield perm

    def _parallelizations(self, cutout: dace.SDFG) -> Tuple[int]:
        params, _ = _maps(cutout)

        # Sequential, static, dynamic_size_1, dynamic_size_2, ...
        encoding = list(range(0, 2 + len(ScheduleSpace._OMP_CHUNK_SIZES), 1))
        return itertools.product(encoding, repeat=len(params))

    def _vectorizations(self, cutout: dace.SDFG) -> int:
        params, _ = _maps(cutout)
        innermost_map_entry = params[list(params.keys())[-1]]

        xform = Vectorization()
        xform._sdfg = cutout
        xform.state_id = cutout.node_id(cutout.start_state)
        xform.map_entry = innermost_map_entry
        if not xform.can_be_applied(cutout.start_state, sdfg=cutout, expr_index=0):
            yield 1
            return

        for vec in ScheduleSpace._VECTOR_SIZES:
            yield vec

    def _local_storages(self, cutout: dace.SDFG):
        try:
            params, _ = _maps(cutout)
        except:
            yield {"in": {}, "out": {}}
            return

        in_arrays, out_arrays = _arrays(cutout)
        op = {"in": {}, "out": {}}
        if len(params) <= 1 or (len(in_arrays) + len(out_arrays)) == 0:
            yield op
            return

        in_options = []
        for array in in_arrays:
            array_options = []

            option = [0] * (len(params) - 1)
            array_options.append(option)

            for i in range(len(params) - 1):
                option = [0] * (len(params) - 1)
                option[i] = 1
                array_options.append(option)

            in_options.append(array_options)
        in_options = itertools.product(*in_options)

        out_options = []
        for array in out_arrays:
            array_options = []

            option = [0] * (len(params) - 1)
            array_options.append(option)

            for i in range(len(params) - 1):
                option = [0] * (len(params) - 1)
                option[i] = 1
                array_options.append(option)

            out_options.append(array_options)
        out_options = itertools.product(*out_options)

        options = itertools.product(in_options, out_options)
        for in_option, out_option in options:
            op = {"in": {}, "out": {}}
            for i, array in enumerate(in_arrays):
                option = in_option[i]
                op["in"][array] = option
            for i, array in enumerate(out_arrays):
                option = out_option[i]
                op["out"][array] = option

            yield op

    def find_state(self, desc: str) -> Tuple[dace.SDFG, List[int]]:
        (
            flt_desc,
            slt_desc,
            perm_desc,
            par_desc,
            vec_desc,
            loc_storage_desc,
        ) = desc.split("#")

        flt_ref = ast.literal_eval(flt_desc)
        for i, first_level_tiling in enumerate(self._first_level_tilings(self._cutout)):
            if first_level_tiling != flt_ref:
                continue

            slt_ref = ast.literal_eval(slt_desc)
            for j, second_level_tiling in enumerate(
                self._second_level_tilings(
                    self._cutout, first_level_tiling=first_level_tiling
                )
            ):
                if second_level_tiling != slt_ref:
                    continue

                # Apply: Tiling
                tiled_cutout = copy.deepcopy(self._cutout)
                ScheduleSpace._apply_tilings(
                    tiled_cutout, (first_level_tiling, second_level_tiling)
                )

                for k, permutation in enumerate(self._permutations(tiled_cutout)):
                    if str(permutation) != perm_desc:
                        continue

                    # Apply: Permutation
                    permuted_cutout = copy.deepcopy(tiled_cutout)
                    ScheduleSpace._apply_permutation(permuted_cutout, permutation)

                    for l, parallelization in enumerate(
                        self._parallelizations(permuted_cutout)
                    ):
                        if str(parallelization) != par_desc:
                            continue

                        # Apply: Parallelization
                        parallelized_cutout = copy.deepcopy(permuted_cutout)
                        ScheduleSpace._apply_parallelization(
                            parallelized_cutout, parallelization
                        )

                        vector_lengths = list(self._vectorizations(permuted_cutout))
                        if int(vec_desc) in vector_lengths:
                            m = vector_lengths.index(int(vec_desc))
                        else:
                            m = len(vector_lengths) - 1

                        # Apply: Vectorization
                        vec_cutout = copy.deepcopy(parallelized_cutout)
                        ScheduleSpace._apply_vectorization(
                            vec_cutout, vector_len=int(vec_desc)
                        )

                        local_storage_ref = ast.literal_eval(loc_storage_desc)
                        for n, local_storage in enumerate(
                            self._local_storages(vec_cutout)
                        ):
                            if local_storage != local_storage_ref:
                                continue

                            # Apply: Local Storage
                            loc_cutout = copy.deepcopy(vec_cutout)
                            ScheduleSpace._apply_local_storage(
                                loc_cutout, local_storage=local_storage
                            )
                            try:
                                loc_cutout.validate()
                            except:
                                return None

                            return loc_cutout, [i, j, k, l, m, n]

        return None

    @classmethod
    def _apply_tilings(
        cls, cutout: dace.SDFG, tiling: Tuple[Dict[str, int], Dict[str, int]]
    ):
        params, _ = _maps(cutout)

        # First level tiling
        first_level_tiling = tiling[0]
        tiled_params = {}
        for param in params:
            tile_size = first_level_tiling[param]
            if tile_size == 1:
                continue

            map_entry = params[param]
            start, stop, step = map_entry.map.range[0]
            map_extend = dace.symbolic.int_floor((stop + 1 - start), step)
            try:
                map_extend = dace.symbolic.evaluate(
                    map_extend, symbols=cutout.constants
                )
                d = map_extend / tile_size
                divides_evenly = d.is_integer
            except:
                divides_evenly = False

            outer_map_entry = MapTiling.apply_to(
                sdfg=cutout,
                options={
                    "prefix": "tile",
                    "tile_sizes": (tile_size,),
                    "divides_evenly": divides_evenly,
                },
                map_entry=map_entry,
                save=True,
                verify=False,
            )
            tiled_params["tile_" + param] = outer_map_entry

        # Second level tiling
        second_level_tiling = tiling[1]
        for param in tiled_params:
            tile_size = second_level_tiling[param]
            if tile_size == 1:
                continue

            map_entry = tiled_params[param]
            start, stop, step = map_entry.map.range[0]
            map_extend = dace.symbolic.int_floor((stop + 1 - start), step)
            try:
                map_extend = dace.symbolic.evaluate(
                    map_extend, symbols=cutout.constants
                )
                d = map_extend / tile_size
                divides_evenly = d.is_integer
            except:
                divides_evenly = False

            outer_map_entry = MapTiling.apply_to(
                sdfg=cutout,
                options={
                    "prefix": "tile",
                    "tile_sizes": (tile_size,),
                    "divides_evenly": divides_evenly,
                },
                map_entry=map_entry,
                save=True,
                verify=False,
            )

    @classmethod
    def _apply_permutation(cls, cutout: dace.SDFG, permutation: List[str]):
        params, _ = _maps(cutout)

        current_permutation = list(params.keys())
        for i, param in enumerate(permutation):
            if current_permutation[i] == param:
                continue

            map_entry = params[param]
            j = current_permutation.index(param)
            for k in range(j - 1, i - 1, -1):
                upper_param = current_permutation[k]
                upper_map_entry = params[upper_param]

                MapInterchange.apply_to(
                    sdfg=cutout,
                    outer_map_entry=upper_map_entry,
                    inner_map_entry=map_entry,
                    verify=False,
                    save=True,
                )

            current_permutation.remove(param)
            current_permutation.insert(i, param)

    @classmethod
    def _apply_parallelization(cls, cutout: dace.SDFG, parallelization: Tuple[int]):
        params, _ = _maps(cutout)

        ordered_params = list(params.keys())
        for i, flag in enumerate(parallelization):
            if flag == 0:
                continue

            options = {"schedule_type": dace.ScheduleType.CPU_Multicore, "collapse": 1}
            if flag > 1:
                options["omp_schedule_type"] = dace.OMPScheduleType.Dynamic
                options["omp_chunk_size"] = ScheduleSpace._OMP_CHUNK_SIZES[flag - 2]

            map_entry = params[ordered_params[i]]
            MapSchedule.apply_to(
                sdfg=cutout,
                map_entry=map_entry,
                options=options,
                annotate=False,
                save=True,
                verify=False,
            )

    @classmethod
    def _apply_vectorization(cls, cutout: dace.SDFG, vector_len: int):
        if vector_len == 1:
            return

        params, _ = _maps(cutout)
        innermost_map_entry = params[list(params.keys())[-1]]
        start, stop, step = innermost_map_entry.map.range[-1]
        if isinstance(stop, dace.symbolic.SymExpr):
            divides_evenly = False
        else:
            map_extend = dace.symbolic.int_floor((stop + 1 - start), step)
            map_extend = dace.symbolic.evaluate(map_extend, symbols=cutout.constants)
            divisor = map_extend / vector_len
            divides_evenly = divisor.is_integer

        preamble = False
        postamble = not divides_evenly
        Vectorization.apply_to(
            sdfg=cutout,
            map_entry=innermost_map_entry,
            options={
                "vector_len": vector_len,
                "preamble": preamble,
                "postamble": postamble,
            },
            save=True,
            verify=False,
        )

    @classmethod
    def _apply_local_storage(cls, cutout: dace.SDFG, local_storage):
        try:
            params, chain = _maps(cutout)
        except:
            return
        map_entries = list(params.values())

        in_local_storage = local_storage["in"]
        for array in in_local_storage:
            option = in_local_storage[array]
            for i, flag in enumerate(option):
                if flag == 0:
                    continue

                outer_map_entry = map_entries[i]
                inner_map_entry = map_entries[i + 1]

                InLocalStorage.apply_to(
                    sdfg=cutout,
                    node_a=outer_map_entry,
                    node_b=inner_map_entry,
                    options={"array": array},
                    save=True,
                    verify=False,
                )

        out_local_storage = local_storage["out"]
        for array in out_local_storage:
            option = out_local_storage[array]
            for i, flag in enumerate(option):
                if flag == 0:
                    continue

                outer_map_exit = cutout.start_state.exit_node(map_entries[i])
                inner_map_exit = cutout.start_state.exit_node(map_entries[i + 1])

                xform = OutLocalStorage()
                xform._sdfg = cutout
                xform.state_id = cutout.node_id(cutout.start_state)
                xform.node_a = inner_map_exit
                xform.node_b = outer_map_exit
                xform.array = array
                if xform.can_be_applied(cutout.start_state, sdfg=cutout, expr_index=0):
                    OutLocalStorage.apply_to(
                        sdfg=cutout,
                        node_a=inner_map_exit,
                        node_b=outer_map_exit,
                        options={"array": array},
                        save=True,
                        verify=False,
                    )
                else:
                    xform = AccumulateTransient()
                    xform._sdfg = cutout
                    xform.state_id = cutout.node_id(cutout.start_state)
                    xform.map_exit = inner_map_exit
                    xform.outer_map_exit = outer_map_exit
                    xform.array = array
                    if xform.can_be_applied(
                        cutout.start_state, sdfg=cutout, expr_index=0
                    ):
                        edge = next(
                            cutout.start_state.edges_between(
                                inner_map_exit, outer_map_exit
                            ).__iter__()
                        )
                        reduction_type = detect_reduction_type(edge.data.wcr)

                        iden = None
                        if reduction_type != dace.dtypes.ReductionType.Custom:
                            dtype = cutout.arrays[array].dtype
                            identity = dace.dtypes.reduction_identity(
                                dtype, reduction_type
                            )
                            iden = identity

                        AccumulateTransient.apply_to(
                            sdfg=cutout,
                            map_exit=inner_map_exit,
                            outer_map_exit=outer_map_exit,
                            options={"array": array, "identity": iden},
                            save=True,
                            verify=False,
                        )


def _maps(cutout: dace.SDFG):
    chain = {}
    for node in cutout.start_state.nodes():
        if not isinstance(node, dace.nodes.MapEntry):
            continue

        parent_map = cutout.start_state.entry_node(node)
        assert parent_map not in chain, "Must be a chain of maps"
        chain[parent_map] = node

    params = OrderedDict()
    map_entry = None
    while map_entry in chain:
        map_entry = chain[map_entry]
        params[str(map_entry.map.params[0])] = map_entry

    return params, chain


def _arrays(cutout: dace.SDFG):
    _, chain = _maps(cutout)
    outermost_map_entry = chain[None]

    in_arrays = set()
    for edge in cutout.start_state.in_edges(outermost_map_entry):
        if not isinstance(edge.src, dace.nodes.AccessNode):
            continue

        in_array = cutout.arrays[edge.data.data]
        if isinstance(in_array, dace.data.Scalar):
            continue

        in_arrays.add(edge.data.data)

    parent_map_exit = cutout.start_state.exit_node(outermost_map_entry)
    out_arrays = set()
    for edge in cutout.start_state.out_edges(parent_map_exit):
        if not isinstance(edge.dst, dace.nodes.AccessNode):
            continue

        out_array = cutout.arrays[edge.data.data]
        if isinstance(out_array, dace.data.Scalar):
            continue

        out_arrays.add(edge.data.data)

    return in_arrays, out_arrays
