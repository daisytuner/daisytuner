# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import os
import copy
import math
import json
import dace
import platform
import numpy as np
import pandas as pd

from typing import Dict, List
from pathlib import Path

from daisytuner.profiling.measure import measure, random_arguments

from daisytuner.profiling.likwid_helpers import (
    likwid_groups,
    cpu_codename,
    gpu_codename,
)
from daisytuner.profiling.metrics.metrics_factory import MetricsFactory


SPECIAL_COUNTERS = set(
    [
        "CPU_CLK_UNHALTED_CORE",
        "CPU_CLK_UNHALTED_REF",
        "CPU_CLOCKS_UNHALTED",
        "CPU_CYCLES",
        "ACTUAL_CPU_CLOCK",
        "MAX_CPU_CLOCK",
    ]
)

MEASUREMENTS = 10


class Profiling:
    """
    Performance metrics measured through LIKWID.
    """

    def __init__(
        self,
        sdfg: dace.SDFG,
        groups: List[str] = None,
        arguments: Dict = None,
        device: dace.DeviceType = dace.DeviceType.CPU,
        cache_path: Path = None,
        hostname: str = None,
        codename: str = None,
    ) -> None:
        assert platform.system() in [
            "Linux"
        ], f"Profiling is not supported {platform.system()}"

        self._sdfg = sdfg
        self._arguments = arguments
        self._device = device
        if hostname is None:
            self._hostname = platform.node()
            self._codename = (
                cpu_codename()
                if self._device == dace.DeviceType.CPU
                else gpu_codename()
            )
        else:
            self._hostname = hostname
            self._codename = codename
            assert (
                self._codename is not None
            ), "Architecture codename must be provded when hostname is different to current host, e.g., broadwellEP"

        if groups is None:
            assert (
                self._hostname == platform.node()
            ), "Groups must be provided as arguments when hostname is different to current host"
            self._groups = likwid_groups(device)
        else:
            self._groups = groups

        self._cache_path = cache_path
        if self._cache_path is None:
            self._cache_path = Path(sdfg.build_folder) / "daisy"

        self._cache_path = (
            self._cache_path / "analysis" / "instrumentation" / self._hostname
        )

        self._counters = None

    def analyze(self) -> pd.DataFrame:
        raw_data = self._get_raw_data()
        counters = None
        for group in self._groups:
            ex = [] if counters is None else counters.columns
            group_counters = Profiling._process(
                raw_data[group], device=self._device, exclude=ex
            )
            if group_counters is None:
                continue

            if counters is None:
                counters = group_counters
            else:
                counters = pd.merge(
                    left=counters, right=group_counters, on=["THREAD_ID", "REPETITION"]
                )

        self._counters = counters
        return counters

    def _get_raw_data(self) -> Dict:
        report = {}

        for group in self._groups:
            group_cache_path = self._cache_path / f"{group}.json"
            if group_cache_path.is_file():
                group_report = json.load(open(group_cache_path, "r"))
            elif "DAISY_TRAINING" in os.environ and os.environ["DAISY_TRAINING"] == "1":
                raise ValueError("Training Mode: Profiling data missing")
            else:
                assert (
                    self._hostname == platform.node()
                ), "Groups can only be measured on current host"
                group_report = self._measure_group(group)

            report[group] = group_report[group]

        return report

    def _measure_group(self, group: str) -> Dict:
        if self._arguments is None:
            self._arguments = random_arguments(self._sdfg)

        inst = self._sdfg.instrument
        self._sdfg.instrument = dace.InstrumentationType.Timer
        if self._device == dace.DeviceType.CPU:
            for state in self._sdfg.states():
                state.instrument = dace.InstrumentationType.LIKWID_CPU
            os.environ["LIKWID_EVENTS"] = group
        else:
            for state in self._sdfg.states():
                state.instrument = dace.InstrumentationType.LIKWID_GPU
            os.environ["LIKWID_GEVENTS"] = group

        arguments = copy.deepcopy(self._arguments)
        runtime, _, _ = measure(
            self._sdfg, arguments=arguments, measurements=MEASUREMENTS
        )
        if runtime == math.inf:
            raise ValueError(f"Failed to measure {group} group")

        group_report = self._sdfg.get_latest_report()
        group_report = {
            group: {
                "durations": {
                    str(k): dict(v) for k, v in group_report.durations.items()
                },
                "counters": {str(k): dict(v) for k, v in group_report.counters.items()},
            }
        }
        print(group_report)

        group_cache_path = self._cache_path / f"{group}.json"
        self._cache_path.mkdir(exist_ok=True, parents=True)
        with open(group_cache_path, "w") as handle:
            json.dump(group_report, handle)

        self._sdfg.instrument = inst
        for state in self._sdfg.states():
            state.instrument = dace.InstrumentationType.No_Instrumentation

        return group_report

    def performance_metrics(self) -> pd.DataFrame:
        return MetricsFactory.create(self._codename, self._groups).compute(
            self._counters
        )

    @staticmethod
    def _process(data: Dict, device: str, exclude: List = None) -> pd.DataFrame:
        counters = data["counters"]["(0, 0, -1)"]["state_0_0_-1"]

        threads = None
        num_threads = None
        if device == dace.DeviceType.GPU:
            # GPUs
            threads = ["0"]
            num_threads = 1
        else:
            # Hardware threads
            for counter in SPECIAL_COUNTERS:
                if counter in counters:
                    threads = list(counters[counter].keys())
                    num_threads = len(threads)
                    break

        columns = []
        all_values = []
        for counter in counters:
            if exclude is not None and counter in exclude:
                continue

            if "0" in counters[counter]:
                reps = len(counters[counter]["0"])
            else:
                reps = len(counters[counter][0])

            values = np.zeros((num_threads, reps))
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    values[i, j] = counters[counter][threads[i]][-j]

            # Sum performance counters are measured over multiple channels
            if counter == "CAS_COUNT_RD" or counter == "CAS_COUNT_WR":
                # Postfix
                values = np.hstack([values[:, 1:], values[:, 0][:, None]])
                size = all_values[0].shape[1]
                num_channels = values.shape[1] // size
                values = values.reshape(num_threads, -1, num_channels)
                values = np.sum(values, axis=-1).squeeze()
                values = np.flip(values, axis=-1)

            if device == dace.DeviceType.GPU:
                if values.shape[1] > MEASUREMENTS:
                    continue

            columns.append(counter)
            all_values.append(values)

        if not all_values:
            return None

        if device == dace.DeviceType.CPU:
            all_values = np.stack(all_values, axis=2)
        else:
            all_values = np.stack(all_values, axis=1)
            all_values = all_values.transpose((0, 2, 1))

        df = []
        for thread_id in range(all_values.shape[0]):
            for rep in range(all_values.shape[1]):
                row = np.concatenate(
                    [all_values[thread_id, rep, :], np.array([thread_id, rep])]
                )[None, :]
                df.append(row)

        df = np.vstack(df)

        for c in exclude:
            if c in columns:
                columns.remove(c)
        columns += ["THREAD_ID", "REPETITION"]

        df = pd.DataFrame(df, columns=columns)

        if exclude is None or not "TIME" in exclude:
            runtimes = (
                np.array(
                    [
                        np.array(measurements)
                        for _, measurements in data["durations"]["(0, 0, -1)"][
                            "Timer"
                        ].items()
                    ]
                )
                / 1e3  # Convert to seconds
            )
            runtimes = runtimes.reshape(-1)
            df["TIME"] = runtimes

        return df
