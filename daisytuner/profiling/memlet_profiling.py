# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import copy
import dace
import platform
import pandas as pd

from typing import Dict

from daisytuner.analysis.similarity.map_nest import MapNest
from daisytuner.profiling.likwid_helpers import likwid_groups
from daisytuner.profiling.profiling import Profiling


class MemletProfiling:
    def __init__(
        self, loop_nest: MapNest, arguments: Dict = None, device: str = "cpu"
    ) -> None:
        assert platform.system() in [
            "Linux"
        ], f"MemletProfiling is not supported {platform.system()}"
        assert device in ["cpu", "gpu"]

        self._loop_nest = loop_nest
        self._arguments = arguments
        self._device = device
        self._hostname = platform.node()
        self._groups = likwid_groups(device)

        self._cache_path = loop_nest.cache_folder / "analysis" / "memlet_profiling"

        self._input_arrays = None
        self._kernels = None
        self._metrics = None

    def analyze(self) -> pd.DataFrame:
        sdfg = self._loop_nest.cutout
        state = sdfg.start_state
        self._input_arrays = set()
        for node in state.data_nodes():
            if state.in_degree(node) == 0 and isinstance(
                sdfg.arrays[node.data], dace.data.Array
            ):
                self._input_arrays.add(node.data)

        # Create kernels
        self._kernels = {}
        for array in self._input_arrays:
            self._kernels[array] = self._create_kernel(array)

        # Measure
        self._metrics = {}
        for array, kernel in self._kernels.items():
            loop_nest = MapNest.create(
                kernel, kernel.start_state, build_folder=self._cache_path / array
            )
            inst = Profiling(
                loop_nest=loop_nest.cutout,
                arguments=self._arguments,
                groups=self._groups,
                device=self._device,
                cache_path=loop_nest.cache_folder,
            )
            _ = inst.analyze()
            metrics = inst.performance_metrics()
            self._metrics[array] = metrics

        return self._metrics

    def _create_kernel(self, array: dace.data.Array):
        kernel = copy.deepcopy(self._loop_nest.cutout)

        # Remove memlet paths of all other arrays
        state = kernel.start_state
        for node in state.data_nodes():
            if state.in_degree(node) == 0 and node.data != array:
                for edge in state.out_edges(node):
                    state.remove_memlet_path(edge)

        syms = {**kernel.constants}
        for sym in kernel.free_symbols:
            syms[sym] = 1
        kernel.specialize(syms)

        kernel.validate()

        return kernel
