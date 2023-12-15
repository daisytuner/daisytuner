# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import ast
import copy
import math
import json
import time
import random
import platform
import networkx as nx
import numpy as np

from deap import creator, base, tools, algorithms

from functools import partial
from tqdm import tqdm
from typing import Dict, List

from daisytuner.analysis.similarity import MapNest
from daisytuner.profiling.measure import measure

from daisytuner.tuning.schedule_space.schedule_space import ScheduleSpace, _arrays
from daisytuner.tuning.tiramisu.tiramisu_tuner import TiramisuTuner
from daisytuner.tuning.evolutionary.nearest_neighbors_sampler import (
    NearestNeighborsSampler,
)


class EvolutionaryTuner:
    def __init__(
        self,
        map_nest: MapNest,
        arguments: Dict,
        population: int = 100,
        generations: int = 10,
        epoch: int = 0,
        nns: NearestNeighborsSampler = None,
    ) -> None:
        self._hostname = platform.node()
        self._arguments = arguments
        self._map_nest = map_nest
        self._num_population = population
        self._num_generations = generations
        self._epoch = epoch
        self._nns = nns

        # Set up cache
        self._cache_path = (
            self._map_nest.cache_folder / "tuning" / self._hostname / "cpu"
        )
        self._cache_path.mkdir(exist_ok=True, parents=True)

        # Base SDFG and preprocess
        self._base_sdfg = copy.deepcopy(self._map_nest.cutout)
        self._base_sdfg_path = self._cache_path / "base.sdfg"
        if not self._base_sdfg_path.is_file():
            self._base_sdfg.save(self._base_sdfg_path)

        # Set up search space
        self._best_candidate = None
        self._best_candidate_desc = None
        self._schedule_space = ScheduleSpace(self._base_sdfg)
        outermost_map = next(nx.topological_sort(self._map_nest.tree))
        dims = len(outermost_map.map.params)
        in_arrays, out_arrays = _arrays(self._map_nest.cutout)

        # Set up objective
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # Set up functions
        self._toolbox = base.Toolbox()
        self._toolbox.register("evaluate", self._evaluate)
        self._toolbox.register("select", tools.selBest, k=population)
        self._toolbox.register("mate", tools.cxTwoPoint)
        self._toolbox.register("population", self._seed)
        self._toolbox.register(
            "mutate",
            tools.mutUniformInt,
            low=0,
            up=[
                len(ScheduleSpace._TILE_SIZES) ** (min(dims, 6)),
                len(ScheduleSpace._TILE_SIZES) ** (min(dims, 6)),
                math.factorial(min(9, 3 * dims)),
                (2 + len(ScheduleSpace._OMP_CHUNK_SIZES)) ** (min(3 * dims, 6)),
                4,
                (dims ** len(in_arrays)) * (dims ** len(out_arrays)),
            ],
            indpb=0.3,
        )
        state_gen = [
            partial(random.randint, 0, min(high, 1e7))
            for high in [
                len(ScheduleSpace._TILE_SIZES) ** (min(dims, 6)),
                len(ScheduleSpace._TILE_SIZES) ** (min(dims, 6)),
                math.factorial(min(9, 3 * dims)),
                (2 + len(ScheduleSpace._OMP_CHUNK_SIZES)) ** (min(3 * dims, 6)),
                4,
                (dims ** len(in_arrays)) * (dims ** len(out_arrays)),
            ]
        ]
        self._toolbox.register(
            "individual",
            tools.initCycle,
            creator.Individual,
            state_gen,
            n=1,
        )

        self._toolbox.register(
            "individual_guess", self._init_individual, creator.Individual
        )

        self._best_candidate = None

    def _init_individual(self, icls, content):
        return icls(content)

    def _seed(self) -> List[List[int]]:
        descs = []
        if self._epoch == 0 or self._nns is None:
            # Tiramisu seed
            tiramisu_candidates = TiramisuTuner().tune(
                self._base_sdfg,
                self._base_sdfg_path,
                beam_size=self._num_population,
                max_depth=15,
            )
            descs.extend(tiramisu_candidates)
        elif self._nns is not None:
            nns_candidates = self._nns.sample(
                map_nest=self._map_nest, k=self._num_population
            )
            descs.extend(nns_candidates)

        print("Seeds: ", descs)

        # Create individuals from states
        population = []
        for desc in descs:
            state = self._schedule_space.find_state(desc=desc)
            if state is None:
                print("Failed: ", desc)
                continue

            population.append(self._toolbox.individual_guess(state[1]))
            if len(population) >= self._num_population:
                break

        if not population:
            population.append(self._toolbox.individual_guess([0, 0, 0, 0, 0, 0]))

        # Fill up with random individuals
        for _ in range(max(self._num_population - len(population), 0)):
            ind = random.sample(population, k=1)[0]
            noise = np.random.normal(1, 1.1, (len(ind),)).squeeze()
            noise[noise < 0] = 0
            ind = np.array(ind) + noise
            ind = [int(i) for i in ind]
            population.append(self._toolbox.individual_guess(ind))

        return population

    def _evaluate(self, individual):
        state = individual
        print("Evaluating: ", state)

        candidate = None
        if str(individual) not in self._cache:
            try:
                candidate, state, state_desc = next(
                    self._schedule_space.enumerate(state=individual).__iter__()
                )

                print(state_desc, state)
                runtime, process_time, _ = measure(
                    candidate,
                    arguments=self._arguments,
                    max_variance=0.3,
                    timeout=self._upper_bound_process_time * 1.5,
                )
            except:
                runtime, process_time = (math.inf, math.inf)
            self._cache[str(individual)] = (runtime, process_time)
        else:
            runtime, process_time = self._cache[str(individual)]

        print(runtime, self._upper_bound_runtime)

        if runtime < self._upper_bound_runtime:
            if candidate is None:
                candidate, state, state_desc = next(
                    self._schedule_space.enumerate(state=individual).__iter__()
                )

            self._upper_bound_runtime = runtime
            self._upper_bound_process_time = process_time
            self._best_candidate = candidate
            self._best_candidate_desc = state_desc
            self._best_candidate.save(
                self._cache_path / f"optimized_{self._epoch}.sdfg"
            )
            with open(self._cache_path / f"optimized_{self._epoch}.txt", "w") as handle:
                handle.write(self._best_candidate_desc)

        if (time.time() - self._cache_timer) > 10 * 60:
            with open(self._cache_path / f"cache_{self._cache_i}.json", "w") as handle:
                json.dump(self._cache, handle)

                self._cache_i = self._cache_i + 1
                self._cache_timer = time.time()

        return (runtime,)

    def tune(self):
        if (self._cache_path / f"optimized_{self._epoch}.sdfg").is_file():
            return

        self._cache = {}
        self._cache_i = 0
        caches = sorted(list(self._cache_path.glob("cache_*.json")), reverse=True)
        if caches:
            for last_cache_path in caches:
                try:
                    with open(last_cache_path, "r") as handle:
                        self._cache = json.load(handle)
                        self._cache_i = int(last_cache_path.stem.split("_")[-1]) + 1
                    break
                except:
                    continue

        init_state = [0, 0, 0, 1, 0, 0]
        if str(init_state) not in self._cache:
            candidate, state, desc = next(
                self._schedule_space.enumerate(state=init_state).__iter__()
            )
            base_runtime, base_process_time, _ = measure(
                self._base_sdfg, arguments=self._arguments, max_variance=0.3
            )
            self._cache[str(init_state)] = (base_runtime, base_process_time)

            self._best_candidate = self._base_sdfg
            self._best_candidate_desc = desc
            self._best_candidate.save(
                self._cache_path / f"optimized_{self._epoch}.sdfg"
            )
            with open(self._cache_path / f"optimized_{self._epoch}.txt", "w") as handle:
                handle.write(self._best_candidate_desc)

        self._upper_bound_runtime, self._upper_bound_process_time = self._cache[
            str(init_state)
        ]
        self._load_best_from_cache()
        print("Initial time ", self._upper_bound_runtime)

        # seed population
        population = self._toolbox.population()
        offspring = population

        self._cache_timer = time.time()
        for _ in tqdm(range(self._num_generations)):
            # fitness
            fits = self._toolbox.map(self._toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit

            # selection
            population = self._toolbox.select(offspring, k=len(population))

            # crossover and mutation
            offspring = algorithms.varAnd(
                population, self._toolbox, cxpb=0.5, mutpb=0.5
            )

            print("Finished Generation, new population generated:")
            print(offspring)

            # Dump cache
            if (time.time() - self._cache_timer) > 10 * 60:
                with open(
                    self._cache_path / f"cache_{self._cache_i}.json", "w"
                ) as handle:
                    json.dump(self._cache, handle)

                    self._cache_i = self._cache_i + 1
                    self._cache_timer = time.time()

    def _load_best_from_cache(self):
        state, (runtime, process_time) = min(
            list(self._cache.items()), key=lambda item: item[1][0]
        )
        if runtime == math.inf:
            return

        state = ast.literal_eval(state)
        candidate, state, state_desc = next(
            self._schedule_space.enumerate(state=state).__iter__()
        )

        self._best_candidate = candidate
        self._best_candidate_desc = state_desc
        self._upper_bound_runtime = runtime
        self._upper_bound_process_time = process_time
