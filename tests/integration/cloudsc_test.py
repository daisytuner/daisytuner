# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import copy
import pytest
import numpy as np

from collections import Counter

from daisytuner.normalization import APrioriMapNestNormalization
from daisytuner.profiling.helpers import random_arguments

from pathlib import Path

EXPECTED_NORMALIZATION = {
    "_state_l1649_c1649": Counter((1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3)),
    "_state_l2241_c2241_5": Counter((1, 1, 1, 1, 1, 1, 1, 1)),
    "_state_l2248_c2248_4": Counter((1, 1, 1)),
    "_state_l2399_c2399_1": Counter((1, 1, 1, 1, 1, 1, 1)),
    "_state_l3856_c3856_8": Counter((1, 1, 1)),
    "single_state_body_12": Counter({1: 41, 2: 6, 3: 2}),
    "single_state_body_41": Counter({1: 1, 2: 4, 3: 2}),
    # "single_state_body_48": Counter({1: 1}),
    "single_state_body_60": Counter({1: 22}),
    "single_state_body_61": Counter((2, 2)),
    "single_state_body": Counter((1, 2)),
    # "state_14": Counter({1: 1}),
}


@pytest.mark.skip()
@pytest.mark.parametrize(
    "benchmark",
    [
        pytest.param("_state_l1649_c1649"),
        pytest.param("_state_l2241_c2241_5"),
        pytest.param("_state_l2248_c2248_4"),
        pytest.param("_state_l2399_c2399_1"),
        pytest.param("_state_l3856_c3856_8"),
        pytest.param("single_state_body_12"),
        pytest.param("single_state_body_41"),
        # pytest.param("single_state_body_48"),
        pytest.param("single_state_body_60"),
        pytest.param("single_state_body_61"),
        pytest.param("single_state_body"),
        # pytest.param("state_14"),
    ],
)
def test_cloudsc(benchmark):
    sdfg = dace.SDFG.from_file(Path(__file__).parent / "cloudsc" / f"{benchmark}.sdfg")

    # Avoid dead code elimination
    for dnode in sdfg.start_state.data_nodes():
        if sdfg.start_state.out_degree(dnode) == 0:
            sdfg.arrays[dnode.data].transient = False
        if sdfg.start_state.in_degree(dnode) == 0:
            sdfg.arrays[dnode.data].transient = False

    sdfg_original = copy.deepcopy(sdfg)
    sdfg_original.name = "original"

    # 1. Normalization
    results = {}
    pipeline = APrioriMapNestNormalization()
    pipeline.apply_pass(sdfg, results)

    maps = []
    for nsdfg in sdfg.all_sdfgs_recursive():
        for state in nsdfg.states():
            for node in state.nodes():
                if not isinstance(node, dace.nodes.MapEntry):
                    continue

                maps.append(len(node.map.params))
    assert Counter(maps) == EXPECTED_NORMALIZATION[benchmark]

    # 1.c Compile
    sdfg.compile()

    # 1.d Numerical evaluation
    sdfg_original.specialize({"NPROMA": 512, "KLEV": 128, "_for_it_24": 2})
    sdfg.specialize({"NPROMA": 512, "KLEV": 128, "_for_it_24": 2})

    args_original = random_arguments(sdfg_original)
    args_normalized = copy.deepcopy(args_original)
    args_tuned = copy.deepcopy(args_original)

    sdfg_original(**args_original)
    sdfg(**args_normalized)

    for array, values in args_original.items():
        assert np.allclose(values, args_normalized[array], equal_nan=True)
