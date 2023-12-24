# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import numpy as np

from daisytuner.analysis.similarity.map_nest_encoding import MapNestEncoding


def test_encode_ctype_void():
    features = MapNestEncoding.encode_ctype("void")
    assert np.all(
        features
        == np.array(
            [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
    )


def test_encode_ctype_float():
    features = MapNestEncoding.encode_ctype("float")
    assert np.all(
        features
        == np.array(
            [
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
    )


def test_encode_code_arithmetic_python():
    features = MapNestEncoding.encode_code(
        "b = 2 * a + 1", language=dace.Language.Python
    )
    assert np.all(
        features
        == np.array(
            [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
    )


def test_encode_code_trigonometric_python():
    features = MapNestEncoding.encode_code(
        "b = math.exp(math.pow(a, 2), 2)", language=dace.Language.Python
    )
    assert np.all(
        features
        == np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
    )


def test_encode_code_arithmetic_cpp():
    features = MapNestEncoding.encode_code("b = 2 * a + 1;", language=dace.Language.CPP)
    assert np.all(
        features
        == np.array(
            [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
    )


def test_encode_code_trigonometric_cpp():
    features = MapNestEncoding.encode_code(
        "b = exp(pow(a, 2), 2);", language=dace.Language.CPP
    )
    assert np.all(
        features
        == np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
    )


def test_encode_expression_constant():
    features = MapNestEncoding.encode_expression("2", {}, ["i0"])
    assert np.all(
        features
        == np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
            dtype=np.float32,
        )
    )


def test_encode_expression_symbol():
    features = MapNestEncoding.encode_expression("i0", {}, ["i0"])
    assert np.all(
        features
        == np.array(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
    )


def test_encode_expression_affine():
    features = MapNestEncoding.encode_expression(
        "2 * i1 + 3 * i0 + 2", {}, ["i0", "i1"]
    )
    assert np.all(
        features
        == np.array(
            [3.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
            dtype=np.float32,
        )
    )


def test_encode_expression_non_affine():
    features = MapNestEncoding.encode_expression("i1 * i0", {}, ["i0", "i1"])
    assert np.all(
        features
        == np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float32,
        )
    )


def test_encode_expression_symbolic():
    features = MapNestEncoding.encode_expression("M * i0", {"M": 32}, ["i0"])
    assert np.all(
        features
        == np.array(
            [32.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
    )


def test_encode_expression_missing_symbol():
    features = MapNestEncoding.encode_expression("M * i0", {}, ["i0"])
    assert np.all(
        features
        == np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float32,
        )
    )


def test_encode_diagonal_sum():
    N = dace.symbol("N")
    M = dace.symbol("M")

    @dace.program
    def sdfg_diagonal_sum(A: dace.float32[N, M], B: dace.float32[N]):
        for i in dace.map[0:N]:
            tmp = dace.define_local(i, dtype=dace.float32)
            for j in dace.map[0:i]:
                with dace.tasklet:
                    a << A[i, j]
                    b >> tmp(1, lambda a, b: a + b)[i]
                    b = a

            B[i] = tmp[i]

    sdfg = sdfg_diagonal_sum.to_sdfg()
    sdfg.simplify()

    encoding = MapNestEncoding(sdfg, symbol_values={"N": 32, "M": 64})
    assert encoding.can_be_encoded(sdfg, symbol_values={"N": 32, "M": 64})

    encoding.encode()

    second_map_entry = None
    tmp_access_node = None
    for node in sdfg.start_state.nodes():
        if isinstance(node, dace.nodes.AccessNode) and node.data == "tmp":
            tmp_access_node = node
        elif (
            isinstance(node, dace.nodes.MapEntry)
            and sdfg.start_state.entry_node(node) is not None
        ):
            second_map_entry = node

    sme_encoding = encoding._element_table[second_map_entry]
    sme_encoding = encoding._nodes[sme_encoding]
    sme_encoding = sme_encoding[
        MapNestEncoding.START_MAP_ENTRY : MapNestEncoding.START_MAP_ENTRY
        + MapNestEncoding.NUM_FEATURES_MAP_ENTRY
    ]
    assert np.all(
        sme_encoding
        == np.array(
            [
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                -1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
            ],
            dtype=np.float32,
        )
    )

    tan_encoding = encoding._element_table[tmp_access_node]
    tan_encoding = encoding._nodes[tan_encoding]
    tan_encoding = tan_encoding[
        MapNestEncoding.START_ACCESS_NODE : MapNestEncoding.START_ACCESS_NODE
        + MapNestEncoding.NUM_FEATURES_ACCESS_NODE
    ]
    assert np.all(
        tan_encoding
        == np.array(
            [
                3.0,
                1.0,
                1.0,
                0.0,
                0.0,
                4.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
    )


def test_encode_spmv():
    C = dace.symbol("C")
    nnz = dace.symbol("nnz")

    @dace.program
    def sdfg_spmv(
        A_row: dace.int32[C + 1],
        A_col: dace.int32[nnz],
        A_val: dace.float32[nnz],
        B: dace.float32[C],
        out: dace.float32[C],
    ):
        for i in dace.map[0:C]:
            for j in dace.map[A_row[i] : A_row[i + 1]]:
                with dace.tasklet:
                    w << A_val[j]
                    b << B[A_col[j]]
                    o >> out(0, lambda x, y: x + y)[i]
                    o = w * b

    sdfg = sdfg_spmv.to_sdfg()
    sdfg.simplify()

    encoding = MapNestEncoding(sdfg, symbol_values={"C": 32, "nnz": 10})
    assert encoding.can_be_encoded(sdfg, symbol_values={"C": 32, "nnz": 10})

    encoding.encode()
    assert len(encoding._nodes) == len(sdfg.start_state.nodes())
    assert len(encoding._edges_attr) == len(sdfg.start_state.edges())

    indirection_tasklet = None
    second_map_entry = None
    for node in sdfg.start_state.nodes():
        if (
            isinstance(node, dace.nodes.MapEntry)
            and sdfg.start_state.entry_node(node) is not None
        ):
            second_map_entry = node
        elif isinstance(node, dace.nodes.Tasklet) and node.label == "Indirection":
            indirection_tasklet = node

    sme_encoding = encoding._element_table[second_map_entry]
    sme_encoding = encoding._nodes[sme_encoding]
    sme_encoding = sme_encoding[
        MapNestEncoding.START_MAP_ENTRY : MapNestEncoding.START_MAP_ENTRY
        + MapNestEncoding.NUM_FEATURES_MAP_ENTRY
    ]
    assert np.all(
        sme_encoding
        == np.array(
            [
                2.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ],
            dtype=np.float32,
        )
    )

    it_encoding = encoding._element_table[indirection_tasklet]
    it_encoding = encoding._nodes[it_encoding]
    it_encoding = it_encoding[
        MapNestEncoding.START_TASKLET : MapNestEncoding.START_TASKLET
        + MapNestEncoding.NUM_FEATURES_TASKLET
    ]
    assert np.all(
        it_encoding
        == np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float32,
        )
    )
