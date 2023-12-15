# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import math
import numpy as np

from dace.sdfg.analysis.cutout import SDFGCutout

from daisytuner.profiling.measure import (
    measure,
    random_arguments,
    create_data_report,
    arguments_from_data_report,
)


@dace.program
def maps(A: dace.float64[10, 20], B: dace.float64[10, 20]):
    tmp = dace.define_local([10, 20], dtype=A.dtype)
    for i, j in dace.map[0:10, 0:20]:
        with dace.tasklet:
            a << A[i, j]
            b >> tmp[i, j]

            b = a * a

    for i, j in dace.map[0:10, 0:20]:
        with dace.tasklet:
            a << tmp[i, j]
            b >> B[i, j]

            b = a + 2


def test_random_arguments():
    sdfg = maps.to_sdfg()
    sdfg.simplify()

    arguments = random_arguments(sdfg)
    assert len(arguments) == 2

    assert "A" in arguments
    A = arguments["A"]
    assert A.shape == (10, 20)
    assert np.sum(A) != 0

    assert "B" in arguments
    B = arguments["B"]
    assert B.shape == (10, 20)
    assert np.sum(A) != 0


def test_create_data_report():
    sdfg = maps.to_sdfg()
    sdfg.simplify()

    sdfg.clear_data_reports()

    arguments = random_arguments(sdfg)
    dreport = create_data_report(sdfg, arguments)

    assert len(dreport.keys()) == 3


def test_arguments_from_data_report_cutouts():
    sdfg = maps.to_sdfg()
    sdfg.simplify()

    cutouts = []
    for state in sdfg.nodes():
        for node in state.nodes():
            if not isinstance(node, dace.nodes.MapEntry):
                continue

            subgraph = state.scope_subgraph(node)
            cutout = SDFGCutout.singlestate_cutout(
                state,
                *subgraph.nodes(),
                make_copy=False,
                make_side_effects_global=False,
                use_alibi_nodes=False
            )
            cutouts.append(cutout)

    sdfg.clear_data_reports()

    arguments = random_arguments(sdfg)
    dreport = create_data_report(sdfg, arguments)

    for cutout in cutouts:
        args = arguments_from_data_report(cutout, dreport)
        assert len(args) == 2

        first_map = False
        for node in cutout.start_state:
            if isinstance(node, dace.nodes.AccessNode) and node.data == "A":
                first_map = True
                break

        if first_map:
            assert "A" in args and "tmp" in args
        else:
            assert "tmp" in args and "B" in args


@dace.program
def matmul(A: dace.float64[10, 10], B: dace.float64[10, 10], C: dace.float64[10, 10]):
    C = A + B


def test_measure():
    measurements = 3

    sdfg = matmul.to_sdfg()
    sdfg.expand_library_nodes()
    sdfg.simplify()

    sdfg.instrument = dace.InstrumentationType.Timer

    A = np.random.rand(10, 10).astype(np.float64)
    B = np.random.rand(10, 10).astype(np.float64)
    C = np.zeros_like(B)

    arguments = {"A": A, "B": B, "C": C}
    target = A + B

    runtime, process_time, _ = measure(sdfg, arguments, measurements=measurements)
    assert runtime != math.inf and process_time != math.inf
    assert (runtime / 1000) < process_time

    report = sdfg.get_latest_report()
    durations = list(report.durations.values())[0]
    durations = list(durations.values())[0]
    durations = list(durations.values())[0]

    assert len(durations) == measurements
    assert np.median(np.array(durations)) == runtime


def test_measure_failure():
    sdfg = matmul.to_sdfg()
    sdfg.expand_library_nodes()
    sdfg.simplify()

    sdfg.instrument = dace.InstrumentationType.Timer

    A = np.random.rand(10, 10).astype(np.float64)
    B = np.random.rand(10, 10).astype(np.float64)
    C = np.zeros_like(B)

    arguments = {"A": A, "B": B}

    runtime, process_time, _ = measure(sdfg, arguments, measurements=1)
    assert runtime == math.inf
    assert process_time != math.inf
