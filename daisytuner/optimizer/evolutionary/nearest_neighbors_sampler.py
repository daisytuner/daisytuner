# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import platform

from typing import List, Dict

from daisytuner.analysis.similarity import MapNest
from daisytuner.model import DaisyNet
from daisytuner.analysis.embeddings.embedding_space import EmbeddingSpace


class NearestNeighborsSampler:
    def __init__(self, space: EmbeddingSpace, schedules: Dict) -> None:
        self._space = space
        self._schedules = schedules

    def sample(self, map_nest: MapNest, k: int) -> List[str]:
        nns = self._space.nearest_neighbors(
            query=map_nest, k=min(5 * k, len(self._space))
        )
        schedules = []
        for index, _ in nns.iterrows():
            if not index in self._schedules:
                continue

            schedules.append(self._schedules[index])

        return schedules

    @classmethod
    def from_dataset(cls, map_nests: MapNest, epoch: int):
        model = DaisyNet.create()
        space = EmbeddingSpace.from_dataset(map_nests, model=model)
        schedules = {}
        for map_nest in map_nests:
            tuning_path = map_nest.cache_folder / "tuning" / platform.node() / "cpu"
            optimized_desc = tuning_path / f"optimized_{epoch}.txt"
            if not optimized_desc.is_file():
                continue

            with open(optimized_desc, "r") as handle:
                schedule = handle.readline().strip()

            if len(schedule.split("#")) == 5:
                loc_storage = {"in": {}, "out": {}}
                schedule = schedule + f"#{loc_storage}"

            schedules[map_nest.hash] = schedule

        return NearestNeighborsSampler(space=space, schedules=schedules)
