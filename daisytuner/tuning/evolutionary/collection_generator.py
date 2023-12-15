# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
from __future__ import annotations

import os
import dace
import shutil
import platform
import pandas as pd
import numpy as np

from tqdm import tqdm
from pathlib import Path
from typing import List

from daisytuner.profiling.measure import (
    measure,
    arguments_from_data_report,
    create_data_report,
    random_arguments,
)

from daisytuner.analysis.similarity import MapNest
from daisytuner.tuning.evolutionary.evolutionary_tuner import EvolutionaryTuner
from daisytuner.tuning.evolutionary.nearest_neighbors_sampler import (
    NearestNeighborsSampler,
)


class CollectionGenerator:
    def __init__(self, batch: List[MapNest]) -> None:
        self._batch = batch
        self._hostname = platform.node()

    def tune(self, epochs: int = 3):
        references = pd.DataFrame.from_dict(
            self._references(), orient="index", columns=["runtime"]
        )

        stats = []
        for epoch in range(epochs):
            nns = None
            if epoch > 0:
                nns = NearestNeighborsSampler.from_dataset(
                    map_nests=list(self._batch),
                    epoch=epoch - 1,
                )

            epoch_metrics = {}
            for map_nest in tqdm(self._batch):
                data_report = map_nest.cutout.get_instrumented_data()
                try:
                    args = arguments_from_data_report(
                        map_nest.cutout, data_report=data_report
                    )
                except:
                    args = random_arguments(map_nest.cutout)

                print(map_nest.sdfg.name)

                # Evolution
                tuner = EvolutionaryTuner(
                    map_nest,
                    arguments=args,
                    population=20 if epoch == 0 else 10,
                    generations=3 if epoch == 0 else 2,
                    epoch=epoch,
                    nns=nns,
                )
                tuner.tune()

                optimized_sdfg_path = (
                    map_nest.cache_folder
                    / "tuning"
                    / self._hostname
                    / "cpu"
                    / f"optimized_{epoch}.sdfg"
                )
                if not optimized_sdfg_path.is_file():
                    shutil.copy(
                        map_nest.cache_folder
                        / "tuning"
                        / self._hostname
                        / "cpu"
                        / f"optimized_{epoch-1}.sdfg",
                        optimized_sdfg_path,
                    )
                optimized_sdfg = dace.SDFG.from_file(optimized_sdfg_path)
                optimized_sdfg.build_folder = str(
                    optimized_sdfg_path.parent / f"dacecache_{epoch}"
                )

                # Measure actual runtime
                runtime_path = optimized_sdfg_path.parent / f"runtime_{epoch}.txt"
                if not runtime_path.is_file():
                    # Obtain consistent data
                    data_report = map_nest.cutout.get_instrumented_data()
                    try:
                        args = arguments_from_data_report(
                            map_nest.cutout, data_report=data_report
                        )
                    except:
                        args = random_arguments(optimized_sdfg)

                    runtime, _, _ = measure(optimized_sdfg, arguments=args)
                    runtime = runtime / 1000
                    with open(runtime_path, "w") as handle:
                        handle.write(str(runtime))
                else:
                    with open(runtime_path, "r") as handle:
                        runtime = float(handle.read())

                epoch_metrics[map_nest.hash] = runtime

                print("Optimized: ", runtime, map_nest.sdfg.build_folder)
                print("Reference: ", references.loc[map_nest.hash, "runtime"])

            epoch_metrics = pd.DataFrame.from_dict(
                epoch_metrics, orient="index", columns=["runtime"]
            )
            stats.append(epoch_metrics)
            print("Finished epoch with mean runtime: ", epoch_metrics.mean())

            print(stats[-1].mean())
            print(references["runtime"].mean())

        print(stats[0].mean())
        stats[0]["speedup"] = references["runtime"] / stats[0]["runtime"]
        print(stats[0]["speedup"].mean())

        print(stats[1].mean())
        stats[1]["speedup"] = references["runtime"] / stats[2]["runtime"]
        print(stats[1]["speedup"].mean())

        print(stats[2].mean())
        stats[2]["speedup"] = references["runtime"] / stats[2]["runtime"]
        print(stats[2]["speedup"].mean())

    def _references(self):
        references = {}
        for map_nest in tqdm(self._batch):
            print(map_nest.cache_folder.parent.parent.name)

            # Create data report to ensure consistent measurements
            data_report = map_nest.cutout.get_instrumented_data()
            if data_report is None:
                rand_args = random_arguments(map_nest.cutout)
                data_report = create_data_report(
                    map_nest.cutout, arguments=rand_args, transients=False
                )

            # Instrument
            try:
                arguments = arguments_from_data_report(
                    map_nest.cutout, data_report=data_report
                )
            except KeyError:
                arguments = random_arguments(map_nest.cutout)

            # analysis = Profiling(
            #     sdfg=map_nest.cutout,
            #     arguments=arguments,
            # )
            # try:
            #     res = analysis.analyze()
            #     print(res)
            # except:
            #     continue

            ref_path = map_nest.cache_folder / "tuning" / "reference.txt"
            if ref_path.is_file():
                with open(ref_path, "r") as handle:
                    runtime = float(handle.read())
                references[map_nest.hash] = runtime
            else:
                runtime, _, _ = measure(map_nest.cutout, arguments=arguments)
                runtime = runtime / 1000
                ref_path.parent.mkdir(parents=True, exist_ok=True)
                with open(ref_path, "w") as handle:
                    handle.write(str(runtime))

                references[map_nest.hash] = runtime

            print("Reference: ", references[map_nest.hash])

        return references

    @classmethod
    def from_dataset(cls, path: Path) -> CollectionGenerator:
        dataset = []
        subdirs = [Path(f.path) for f in os.scandir(path) if f.is_dir()]
        subdirs = sorted(subdirs)
        for subdir in tqdm(subdirs):
            sdfg = dace.SDFG.from_file(subdir / f"{subdir.stem}.sdfg")
            sdfg.build_folder = str(subdir / "dacecache")

            map_entry = None
            for node in sdfg.start_state.nodes():
                if (
                    isinstance(node, dace.nodes.MapEntry)
                    and sdfg.start_state.entry_node(node) is None
                ):
                    map_entry = node
                    break
            assert map_entry is not None
            loop_nest = MapNest.create(
                sdfg,
                sdfg.start_state,
                map_entry=map_entry,
                build_folder=sdfg.build_folder,
            )

            dataset.append(loop_nest)

        return CollectionGenerator(batch=dataset)
