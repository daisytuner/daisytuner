# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import copy
import numpy as np

from dace.transformation.dataflow import MapTiling, MapExpansion, MapInterchange
from daisytuner.transformations.map_untiling import MapUntiling
from daisytuner.profiling.helpers import random_arguments

N, M, K = [dace.symbol(k) for k in "NMK"]


@dace.program
def matmul(A: dace.float64[N, K], B: dace.float64[K, M], C: dace.float64[N, M]):
    for i, j, k in dace.map[0:N, 0:M, 0:K]:
        with dace.tasklet:
            a << A[i, k]
            b << B[k, j]
            c >> C(1, lambda a, b: a + b)[i, j]

            c = a * b


def test_matmul_basic_untiling():
    # Create an SDFG from the matrix multiplication function.
    sdfg = matmul.to_sdfg()
    sdfg.simplify()

    # Tile and untile maps.
    sdfg.apply_transformations(MapTiling, dict(tile_sizes=(16, 16)), validate=True)
    applied = sdfg.apply_transformations(MapUntiling, validate=True)

    # Assertions
    assert applied == 1
    assert len(sdfg.states()) == 1

    maps = [
        node
        for node in sdfg.start_state.nodes()
        if isinstance(node, dace.nodes.MapEntry)
    ]
    assert len(maps) == 1

    # Check equivalence
    sdfg.specialize({"N": 32, "M": 32, "K": 32})

    args = random_arguments(sdfg)
    args_untiled = copy.deepcopy(args)

    sdfg(**args_untiled)

    sdfg = matmul.to_sdfg()
    sdfg.simplify()
    sdfg.specialize({"N": 32, "M": 32, "K": 32})

    sdfg(**args)
    for array in args:
        assert array in args_untiled
        assert np.allclose(args[array], args_untiled[array], equal_nan=False)


def test_matmul_mixed_untiling():
    # Create an SDFG from the matrix multiplication function.
    sdfg = matmul.to_sdfg()
    sdfg.simplify()

    # Tile and untile maps.
    sdfg.apply_transformations(MapTiling, dict(tile_sizes=(16, 1, 1)), validate=True)
    applied = sdfg.apply_transformations(MapUntiling, validate=True)

    # Assertions
    assert applied == 1
    assert len(sdfg.states()) == 1

    maps = [
        node
        for node in sdfg.start_state.nodes()
        if isinstance(node, dace.nodes.MapEntry)
    ]
    assert len(maps) == 2


def test_matmul_nested_untiling():
    # Create an SDFG from the matrix multiplication function.
    sdfg = matmul.to_sdfg()
    sdfg.simplify()

    # Tile and untile maps.
    sdfg.apply_transformations(MapTiling, dict(tile_sizes=(16, 16, 16)), validate=True)
    sdfg.apply_transformations(MapExpansion, validate=True)

    applied = sdfg.apply_transformations_repeated(MapUntiling, validate=True)
    sdfg.apply_transformations(MapInterchange, validate=True)
    applied += sdfg.apply_transformations_repeated(MapUntiling, validate=True)
    sdfg.apply_transformations(MapInterchange, validate=True)
    sdfg.apply_transformations(MapInterchange, validate=True)
    applied += sdfg.apply_transformations_repeated(MapUntiling, validate=True)

    # Assertions
    assert applied == 3
    assert len(sdfg.states()) == 1

    maps = [
        node
        for node in sdfg.start_state.nodes()
        if isinstance(node, dace.nodes.MapEntry)
    ]
    assert len(maps) == 3

    # Check equivalence
    sdfg.specialize({"N": 32, "M": 32, "K": 32})

    args = random_arguments(sdfg)
    args_untiled = copy.deepcopy(args)

    sdfg(**args_untiled)

    sdfg = matmul.to_sdfg()
    sdfg.simplify()
    sdfg.specialize({"N": 32, "M": 32, "K": 32})

    sdfg(**args)
    for array in args:
        assert array in args_untiled
        assert np.allclose(args[array], args_untiled[array], equal_nan=False)


def test_matmul_unbound_untiling():
    # Create an SDFG from the matrix multiplication function.
    sdfg = matmul.to_sdfg()
    sdfg.simplify()

    # Tile and untile maps.
    sdfg.apply_transformations(MapTiling, dict(tile_sizes=(16, 1, 1)), validate=True)
    sdfg.nodes()[0].nodes()[0].map.range[0] = (
        sdfg.nodes()[0].nodes()[0].map.range[0][0],
        sdfg.nodes()[0].nodes()[0].map.range[0][1] + 1,
        sdfg.nodes()[0].nodes()[0].map.range[0][2],
    )

    applied = sdfg.apply_transformations_repeated(MapUntiling, validate=True)

    # Assertions
    assert applied == 0


def test_matmul_dependency_untiling():
    # Create an SDFG from the matrix multiplication function.
    sdfg = matmul.to_sdfg()
    sdfg.simplify()

    # Tile and untile maps.
    sdfg.apply_transformations(MapTiling, dict(tile_sizes=(16, 1, 1)), validate=True)
    tiled_param = dace.symbol(sdfg.nodes()[0].nodes()[-1].map.params[0])
    sdfg.nodes()[0].edges()[0].data.subset[0] = (tiled_param, tiled_param, 1)

    applied = sdfg.apply_transformations_repeated(MapUntiling, validate=True)

    # Assertions
    assert applied == 0
