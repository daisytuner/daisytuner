# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import os
import copy
import dace
import json
import ast
import platform

from abc import ABC
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm

from daisytuner.profiling.helpers import measure
from daisytuner.profiling.likwid_helpers import (
    cpu_codename,
    gpu_codename,
    LIKWID_GROUPS,
)


class CountersNotLoadedException(Exception):
    pass


class PerformanceCounters(ABC):
    """
    A class for easily using performance counters in other classes.
    The class takes care of reading already measured counters from the cache
    and measuring new counters if necessary.
    """

    def __init__(
        self,
        sdfg: dace.SDFG,
        counters: List[str],
        device: dace.DeviceType = dace.DeviceType.CPU,
        hostname: str = platform.node(),
        codename: str = cpu_codename(),
        cache: Dict = None,
    ) -> None:
        self._sdfg = sdfg
        self._counters = counters
        self._device = device
        self._hostname = hostname
        self._codename = codename

        self._values = {}
        self._cache_path = (
            Path(sdfg.build_folder)
            / "daisy"
            / "analysis"
            / "instrumentation"
            / self._hostname
        )
        self._cache_path.mkdir(exist_ok=True, parents=True)

        if cache is not None:
            for counter in self._counters:
                if counter in cache:
                    self._values[counter] = cache[counter]
        else:
            self._load_values_from_cache_path()

    def _load_values_from_cache_path(self):
        for group in self._cache_path.glob("*.json"):
            with open(group, "r") as handle:
                try:
                    group_values = json.load(handle)[group.stem]
                except json.decoder.JSONDecodeError:
                    continue

                # Counters
                counter_values = group_values["counters"]
                for node_uuid in counter_values:
                    sdfg_id, state_id, node_id = ast.literal_eval(node_uuid)

                    nsdfg, state, node = (None, None, None)
                    for sd in self._sdfg.all_sdfgs_recursive():
                        if sd.sdfg_id == sdfg_id:
                            nsdfg = sd
                            if state_id == -1:
                                break

                            state = sd.node(state_id)
                            if node_id == -1:
                                break

                            node = state.node(node_id)

                    node_name = next(counter_values[node_uuid].__iter__())
                    values = counter_values[node_uuid][node_name]
                    for counter in values:
                        if counter not in self._counters:
                            continue

                        if node is not None:
                            self._values[counter] = {node: values[counter]}
                        elif state is not None:
                            self._values[counter] = {state: values[counter]}
                        else:
                            self._values[counter] = {nsdfg: values[counter]}

                # Durations
                duration_values = group_values["durations"]
                for node_uuid in duration_values:
                    sdfg_id, state_id, node_id = ast.literal_eval(node_uuid)

                    nsdfg, state, node = (None, None, None)
                    for sd in self._sdfg.all_sdfgs_recursive():
                        if sd.sdfg_id == sdfg_id:
                            nsdfg = sd
                            if state_id == -1:
                                break

                            state = sd.node(state_id)
                            if node_id == -1:
                                break

                            node = state.node(node_id)

                    values = duration_values[node_uuid]
                    for counter in values:
                        if counter not in self._counters:
                            continue

                        if node is not None:
                            self._values[counter] = {node: values[counter]}
                        elif state is not None:
                            self._values[counter] = {state: values[counter]}
                        else:
                            self._values[counter] = {nsdfg: values[counter]}

    def has_values(self) -> bool:
        for c in self._counters:
            if c not in self._values:
                return False
        return True

    def values(self) -> Dict[str, List[float]]:
        if not self.has_values():
            raise CountersNotLoadedException(
                "Requested counters are not loaded. Counters might not be measured yet."
            )
        return self._values

    def measure(self, arguments: Dict, keep_existing: bool = False) -> bool:
        assert self._hostname == platform.node()
        if self._device == dace.DeviceType.CPU:
            assert self._codename == cpu_codename()
        else:
            assert self._codename == gpu_codename()

        if self._device == dace.DeviceType.CPU:
            for state in self._sdfg.states():
                state.instrument = dace.InstrumentationType.LIKWID_CPU
        else:
            for state in self._sdfg.states():
                state.instrument = dace.InstrumentationType.LIKWID_GPU

        existing_groups = [g.stem for g in self._cache_path.glob("*.json")]

        print("Measuring performance counters ...")
        for group in tqdm(LIKWID_GROUPS[self._codename]):
            if group == "PORT_USAGE":
                continue
            if keep_existing and group in existing_groups:
                continue

            # Instruct LIKWID
            if self._device == dace.DeviceType.CPU:
                env = "LIKWID_EVENTS"
                os.environ[env] = group
            else:
                env = "LIKWID_GEVENTS"
                os.environ[env] = group

            # Measure
            args = copy.deepcopy(arguments)
            report = measure(self._sdfg, args)

            # Save report
            report = {
                group: {
                    "durations": {str(k): dict(v) for k, v in report.durations.items()},
                    "counters": {str(k): dict(v) for k, v in report.counters.items()},
                }
            }
            with open(self._cache_path / f"{group}.json", "w") as handle:
                json.dump(report, handle)

        for state in self._sdfg.states():
            state.instrument = dace.InstrumentationType.No_Instrumentation

        self._load_values_from_cache_path()

    @staticmethod
    def load_cache(sdfg: Path, hostname: str) -> Dict:
        cache_path = (
            Path(sdfg.build_folder)
            / "daisy"
            / "analysis"
            / "instrumentation"
            / hostname
        )

        all_values = {}
        for group in cache_path.glob("*.json"):
            with open(group, "r") as handle:
                try:
                    group_values = json.load(handle)[group.stem]
                except json.decoder.JSONDecodeError:
                    continue

                # Counters
                counter_values = group_values["counters"]
                for node_uuid in counter_values:
                    sdfg_id, state_id, node_id = ast.literal_eval(node_uuid)

                    nsdfg, state, node = (None, None, None)
                    for sd in sdfg.all_sdfgs_recursive():
                        if sd.sdfg_id == sdfg_id:
                            nsdfg = sd
                            if state_id == -1:
                                break

                            state = sd.node(state_id)
                            if node_id == -1:
                                break

                            node = state.node(node_id)

                    node_name = next(counter_values[node_uuid].__iter__())
                    values = counter_values[node_uuid][node_name]
                    for counter in values:
                        if node is not None:
                            all_values[counter] = {node: values[counter]}
                        elif state is not None:
                            all_values[counter] = {state: values[counter]}
                        else:
                            all_values[counter] = {nsdfg: values[counter]}

                # Durations
                duration_values = group_values["durations"]
                for node_uuid in duration_values:
                    sdfg_id, state_id, node_id = ast.literal_eval(node_uuid)

                    nsdfg, state, node = (None, None, None)
                    for sd in sdfg.all_sdfgs_recursive():
                        if sd.sdfg_id == sdfg_id:
                            nsdfg = sd
                            if state_id == -1:
                                break

                            state = sd.node(state_id)
                            if node_id == -1:
                                break

                            node = state.node(node_id)

                    values = duration_values[node_uuid]
                    for counter in values:
                        if node is not None:
                            all_values[counter] = {node: values[counter]}
                        elif state is not None:
                            all_values[counter] = {state: values[counter]}
                        else:
                            all_values[counter] = {nsdfg: values[counter]}

        return all_values
