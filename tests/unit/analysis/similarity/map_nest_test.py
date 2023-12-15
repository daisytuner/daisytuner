# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace

from collections import Counter

from daisytuner.analysis.similarity.map_nest import MapNest


def test_nodes():
    @dace.program
    def sdfg_test_nodes(A: dace.float32[32], B: dace.float32[32], C: dace.float32[32]):
        C = A + B

    sdfg = sdfg_test_nodes.to_sdfg()

    root = None
    for node in sdfg.start_state.nodes():
        if isinstance(node, dace.nodes.MapEntry):
            root = node
            break

    map_nest = MapNest(sdfg.start_state, root)

    assert map_nest.root == root
    assert len(map_nest.nodes()) == 6
    node_statistics = Counter([type(node).__name__ for node in map_nest.nodes()])
    assert node_statistics["AccessNode"] == 3
    assert node_statistics["MapEntry"] == 1
    assert node_statistics["MapExit"] == 1
    assert node_statistics["Tasklet"] == 1


def test_invalid_node():
    @dace.program
    def sdfg_test_nodes(A: dace.float32[32], B: dace.float32[32], C: dace.float32[32]):
        C = A + B

    sdfg = sdfg_test_nodes.to_sdfg()

    root = None
    for node in sdfg.start_state.nodes():
        if not isinstance(node, dace.nodes.MapEntry):
            root = node
            break

    try:
        _ = MapNest(sdfg.start_state, root)
        assert False
    except AssertionError:
        assert True


def test_as_cutout():
    M = dace.symbol("M")
    N = dace.symbol("N")

    @dace.program
    def sdfg_test_as_cutout(
        A: dace.float32[M, N], B: dace.float32[M, N], C: dace.float32[M, N]
    ):
        C = A + B

    sdfg = sdfg_test_as_cutout.to_sdfg()
    sdfg.specialize({"M": 32, "N": 16})

    root = None
    for node in sdfg.start_state.nodes():
        if isinstance(node, dace.nodes.MapEntry):
            root = node
            break

    map_nest = MapNest(sdfg.start_state, root)
    cutout = map_nest.as_cutout()
    assert "M" in cutout.symbols
    assert "N" in cutout.symbols
    assert cutout.constants["M"] == 32
    assert cutout.constants["N"] == 16
