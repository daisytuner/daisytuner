# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import os
import sys
import copy
import dace
import json
import sympy
import numpy as np

from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from daisytuner.analysis.similarity import MapNest
from daisytuner.tuning.cutout_tuner import CutoutTuner
from daisytuner.tuning.schedule_space.schedule_space import _arrays, ScheduleSpace
from daisytuner.profiling.measure import random_arguments, measure


class TiramisuTuner(CutoutTuner):
    def __init__(
        self,
        beam_size: int = 10,
        max_depth: int = 15,
        search_method: str = "MCTS",
        topK: int = 3,
    ) -> None:
        self._beam_size = beam_size
        self._max_depth = max_depth
        self._search_method = search_method
        self._topK = topK

        try:
            from daisy_tiramisu import SDFGWrapper, AutoScheduler
        except:
            raise ValueError(
                "Daisytuner was built without support for Tiramisu tuner. Please install daisytuner accordingly."
            )

    def tune(
        self,
        loop_nest: MapNest,
        arguments: Dict = None,
    ) -> List[str]:
        from daisy_tiramisu import SDFGWrapper, AutoScheduler

        cutout = loop_nest.cutout
        cutout.save("tmp.sdfg")

        py_cmd_path = str(sys.executable)
        py_interface_path = str(
            Path(__file__).parent.parent.parent
            / "data"
            / "tiramisu"
            / "scripts"
            / "main.py"
        )

        wrapper = SDFGWrapper(str("tmp.sdfg"))
        self._auto_scheduler = AutoScheduler(
            wrapper,
            py_cmd_path,
            py_interface_path,
            self._search_method,
            self._max_depth,
            self._beam_size,
        )

        os.environ["MEM_SIZE"] = str(1024 * 1024)

        schedules = self._auto_scheduler.tune()
        _, schedules = schedules.split(";", 1)
        schedules = json.loads(schedules)

        candidates = []
        visited = set()
        for sched, score in tqdm(schedules):
            if sched in visited:
                continue

            parsed_schedule = self._parse_schedule(loop_nest.cutout, sched)

            visited.add(sched)
            candidates.append((parsed_schedule, score))

        if self._topK > 1 and arguments is None:
            arguments = random_arguments(loop_nest.cutout)

        if self._topK > 1:
            best_runtime, best_process_time, _ = measure(
                loop_nest.cutout, arguments=arguments, measurements=1
            )
            best_state = None

        evaluated = 0
        for desc, score in candidates:
            schedule_space = ScheduleSpace(cutout=loop_nest.cutout)
            state = schedule_space.find_state(desc=desc)
            if state is None:
                continue

            if self._topK == 1:
                replace_subgraph_by_cutout(loop_nest, state[0])
                print(desc, score, flush=True)
                return desc, score
            else:
                cutout_ = copy.deepcopy(loop_nest.cutout)
                map_entry = None
                for node in cutout_.start_state.nodes():
                    if (
                        not isinstance(node, dace.nodes.MapEntry)
                        or cutout_.start_state.entry_node(node) is not None
                    ):
                        continue

                    map_entry = node
                    break

                loop_nest_ = MapNest.create(
                    sdfg=cutout_, state=cutout_.start_state, map_entry=map_entry
                )

                replace_subgraph_by_cutout(loop_nest_, copy.deepcopy(state[0]))

                runtime, process_time, _ = measure(
                    cutout_,
                    arguments=arguments,
                    measurements=1,
                    timeout=best_process_time * 1.5,
                )
                if best_runtime / runtime > 1.1:
                    best_runtime = runtime
                    best_process_time = process_time
                    best_state = (desc, score, state[0])

                evaluated += 1
                if evaluated == self._topK:
                    break

        print(desc, score)
        if best_state is not None:
            replace_subgraph_by_cutout(loop_nest, best_state[2])
            return best_state[0], best_state[1]
        else:
            return None

    def _parse_schedule(self, sdfg: dace.SDFG, schedule: str):
        map_entry = None
        for node in sdfg.start_state.nodes():
            if not isinstance(node, dace.nodes.MapEntry):
                continue

            assert map_entry is None, "Multiple map entries"
            map_entry = node

        params = list(map_entry.map.params)
        mapping = {f"L{i}": p for i, p in enumerate(params)}
        first_level_tiling = {p: 1 for p in params}
        second_level_tiling = {}
        parallel = set()

        transformations: List[str] = schedule.split("#")
        for trans in transformations:
            if not trans:
                continue

            if trans.startswith("I"):
                trans = trans[1:]
                trans = trans[1:-1]
                trans = trans.split(",")
                l0, l1 = (trans[0], trans[1])
                i = params.index(mapping[l0])
                j = params.index(mapping[l1])
                params[i] = mapping[l1]
                params[j] = mapping[l0]

                mapping[l0] = params[i]
                mapping[l1] = params[j]
            elif trans.startswith("T"):
                trans = trans[1:]
                trans = trans[2:-1]
                trans = trans.split(",")
                levels = trans[: len(trans) // 2]
                tile_sizes = trans[len(trans) // 2 :]

                start = params.index(mapping[levels[0]])
                for i, level in enumerate(levels):
                    param = mapping[level]
                    outer_param = "tile_" + mapping[level]

                    first_level_tiling[param] = int(tile_sizes[i])
                    second_level_tiling[outer_param] = 1

                    mapping["L" + str(len(params))] = mapping[level]
                    mapping[level] = "tile_" + mapping[level]

                    params.insert(start, outer_param)
                    start = start + 1
            elif trans.startswith("P"):
                trans = trans[1:]
                level = trans[1:-1]
                parallel.add(mapping[level])
            else:
                raise ValueError("Unknown transformation")

        # Parallelization: Params to binary repr
        parallelization = [0] * len(params)
        for i, param in enumerate(params):
            if param in parallel:
                parallelization[i] = 1

        in_arrays, out_arrays = _arrays(sdfg)
        loc_storage = {
            "in": {
                arr: [0] * (len(parallelization) - 1)
                for arr in in_arrays
                if len(parallelization) > 1
            },
            "out": {
                arr: [0] * (len(parallelization) - 1)
                for arr in out_arrays
                if len(parallelization) > 1
            },
        }
        desc = f"{first_level_tiling}#{second_level_tiling}#{tuple(params)}#{tuple(parallelization)}#1#{loc_storage}"
        return desc


def encode(sdfg_path: str) -> str:
    sdfg = dace.SDFG.from_file(sdfg_path)
    assert len(sdfg.states()) == 1

    repr = {
        "name": sdfg.name,
        "sdfg_path": sdfg_path,
        "build_folder": sdfg.build_folder,
    }

    # buffers
    buffers = {}
    descs = {}
    for dnode in sdfg.start_state.data_nodes():
        name = dnode.data.replace("_", "")
        if name in buffers:
            continue

        if sdfg.start_state.entry_node(dnode) is not None:
            continue

        if sdfg.start_state.out_degree(dnode) == 0:
            argt = "output"
        elif sdfg.start_state.in_degree(dnode) == 0:
            argt = "input"
        else:
            argt = "temporary"

        desc = sdfg.arrays[dnode.data]
        descs[name] = desc
        buffers[name] = {
            "name": name,
            "dim_sizes": tuple(
                [
                    int(dace.symbolic.evaluate(dim, symbols=sdfg.constants))
                    for dim in desc.shape
                ]
            ),
            "type": desc.dtype.as_numpy_dtype().name,
            "argt": argt,
        }
    repr["buffers"] = list(buffers.values())

    # iteration domain
    maps: Dict[dace.nodes.MapEntry, dace.nodes.MapEntry] = {}
    for node in sdfg.start_state.nodes():
        if not isinstance(node, dace.nodes.MapEntry):
            continue

        parent_map = sdfg.start_state.entry_node(node)
        assert parent_map not in maps

        if parent_map is None:
            maps[None] = node
        else:
            maps[parent_map] = node

    params: List[str] = []
    ranges = []
    parent_map = None
    while parent_map in maps:
        parent_map = maps[parent_map]
        params.extend(parent_map.map.params)
        ranges.extend(parent_map.map.range)

    params = [param.replace("_", "") for param in params]

    iteration_domain = []
    for param, rng in zip(params, ranges):
        b, e, s = rng
        e = e + 1

        try:
            e = dace.symbolic.evaluate(e, symbols=sdfg.constants)
        except:
            pass

        iteration_domain.append(f"{b}<={param}<{e}")

    iteration_domain = " and ".join(iteration_domain)
    iteration_domain = "{ " + "S[" + ",".join(params) + "]: " + iteration_domain + " }"

    # Access string
    map_exit = sdfg.start_state.exit_node(maps[None])
    edges = sdfg.start_state.in_edges(map_exit)
    assert len(edges) == 1

    memlet_out: dace.Memlet = edges[0].data
    accesses = []
    for rng in memlet_out.subset:
        b, _, s = rng

        # Shortcut for now
        assert int(s) == 1
        accesses.append(str(b).replace("_", ""))

    out_name = memlet_out.data.replace("_", "")
    access = ",".join(accesses)
    access = (
        "{ " + "S[" + ",".join(params) + "]" + "->" + f"{out_name}[{access}]" + " }"
    )

    # FLOPS
    flops = {}
    flops["number_of_additions"] = 0
    flops["number_of_subtraction"] = 0
    flops["number_of_multiplication"] = 0
    flops["number_of_division"] = 0
    for tasklet in sdfg.start_state.nodes():
        if not isinstance(tasklet, dace.nodes.Tasklet):
            continue

        f = {
            "+": tasklet.code.as_string.count("+"),
            "*": tasklet.code.as_string.count("*"),
            "-": tasklet.code.as_string.count("-"),
            "/": tasklet.code.as_string.count("/"),
        }

        flops["number_of_additions"] = f["+"]
        flops["number_of_subtraction"] = f["-"]
        flops["number_of_multiplication"] = f["*"]
        flops["number_of_division"] = f["/"]

    if memlet_out.wcr is not None:
        flops["number_of_additions"] += 1

    # Buffer accesses
    innermost_map = maps[None]
    while innermost_map in maps:
        innermost_map = maps[innermost_map]

    buffer_accesses = []
    for edge in sdfg.start_state.out_edges(innermost_map):
        if edge.data.data is None:
            continue

        name = edge.data.data.replace("_", "")
        dims = len(descs[name].shape)
        access_matrix = np.zeros((dims, len(params) + 1), dtype=np.int32)
        for i, rng in enumerate(edge.data.subset):
            b, _, s = rng
            assert int(s) == 1

            for mul, add, var in _encode_expression(
                b, params=params, symbol_map=sdfg.constants
            ):
                # Constant offset
                access_matrix[i, -1] += add

                if var in params:
                    access_matrix[i, params.index(var)] = mul
                elif var is None:
                    continue
                else:
                    raise ValueError(f"Unsupported expression {b}")

        buffer_accesses.append(
            {"buffer": name, "access_matrix": access_matrix.tolist()}
        )

    if memlet_out.wcr is not None:
        name = memlet_out.data.replace("_", "")
        dims = len(descs[name].shape)
        access_matrix = np.zeros((dims, len(params) + 1), dtype=np.int32)
        for i, rng in enumerate(memlet_out.subset):
            b, _, s = rng
            assert int(s) == 1

            for mul, add, var in _encode_expression(
                b, params=params, symbol_map=sdfg.constants
            ):
                # Constant offset
                access_matrix[i, -1] += add

                if var in params:
                    access_matrix[i, params.index(var)] = mul
                elif var is None:
                    continue
                else:
                    raise ValueError(f"Unsupported expression {b}")

        buffer_accesses.append(
            {"buffer": name, "access_matrix": access_matrix.tolist()}
        )

    computation = {
        "iteration_domain": iteration_domain,
        "access": access,
        "type": descs[out_name].dtype.as_numpy_dtype().name,
        "flops": flops,
        "buffer_accesses": buffer_accesses,
    }
    repr["computations"] = [computation]

    return json.dumps(repr)


def _encode_expression(expr: str, params: List[str], symbol_map: Dict) -> None:
    expr = dace.symbolic.overapproximate(expr)

    if isinstance(expr, (sympy.Max, sympy.Min)):
        if expr.args[0].is_Atom:
            expr = expr.args[1]
        else:
            expr = expr.args[0]

    # Special case: Constant
    try:
        const = int(dace.symbolic.evaluate(expr, symbols=symbol_map))
        return [(0, const, None)]
    except:
        pass

    if isinstance(expr, sympy.Mul):
        mult = dace.symbolic.evaluate(expr.args[0], symbols=symbol_map)
        param = str(expr.args[1]).replace("_", "")
        assert param in params
        return [(mult, 0, param)]

    if str(expr).replace("_", "") in params:
        var = str(expr).replace("_", "")
        return [(1, 0, var)]

    accesses = {None: [0, 0, None]}
    for arg in expr.args:
        if isinstance(arg, sympy.Mul):
            mult = int(dace.symbolic.evaluate(arg.args[0], symbols=symbol_map))
            param = str(arg.args[1]).replace("_", "")
            assert param in params
            assert param not in accesses
            accesses[param] = [mult, 0, param]
        elif str(arg).replace("_", "") in params:
            accesses[str(arg)] = [1, 0, str(arg).replace("_", "")]
        else:
            accesses[None][1] += int(dace.symbolic.evaluate(arg, symbols=symbol_map))

    return accesses.values()
