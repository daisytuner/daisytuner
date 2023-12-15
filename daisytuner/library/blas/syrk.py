# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
from __future__ import annotations

import dace
import dace.library
import copy

from dace import properties
from dace.libraries.blas.environments import openblas, intel_mkl
from dace.libraries.blas.blas_helpers import to_blastype
from dace.transformation.transformation import ExpandTransformation


@dace.library.expansion
class ExpandSyrkOpenBLAS(ExpandTransformation):

    environments = [openblas.OpenBLAS]

    @staticmethod
    def expansion(node: Syrk, state: dace.SDFGState, sdfg: dace.SDFG):
        a_edge = list(state.in_edges_by_connector(node, connector="_a"))[0]
        adesc = sdfg.arrays[a_edge.data.data]
        cdesc = sdfg.arrays[state.out_edges(node)[0].data.data]

        dtype = adesc.dtype.base_type
        func = to_blastype(dtype.type).lower() + "syrk"
        alpha = f"{dtype.ctype}({node.alpha})"
        beta = f"{dtype.ctype}({node.beta})"

        # Deal with complex input constants
        if isinstance(node.alpha, complex):
            alpha = f"{dtype.ctype}({node.alpha.real}, {node.alpha.imag})"
        if isinstance(node.beta, complex):
            beta = f"{dtype.ctype}({node.beta.real}, {node.beta.imag})"

        uplo = "CblasUpper"
        if node.uplo in ["L", "l"]:
            uplo = "CblasLower"

        trans = "CblasNoTrans"
        k = adesc.shape[1]
        if node.trans in ["T", "t"]:
            trans = "CblasTrans"
            k = adesc.shape[0]
        elif node.trans in ["C", "c"]:
            trans = "CblasConjTrans"
            k = adesc.shape[0]

        # Adaptations for BLAS API
        opt = {
            "func": func,
            "uplo": uplo,
            "trans": trans,
            "n": cdesc.shape[0],
            "k": k,
            "alpha": alpha,
            "lda": adesc.shape[1],
            "beta": beta,
            "ldc": cdesc.shape[0],
        }

        code = ""
        if dtype in (dace.complex64, dace.complex128):
            code = f"""
            {dtype.ctype} alpha = {alpha};
            {dtype.ctype} beta = {beta};
            """
            opt["alpha"] = "&alpha"
            opt["beta"] = "&beta"

        code += "cblas_{func}(CblasRowMajor, {uplo}, {trans}, {n}, {k}, {alpha}, _a, {lda}, {beta}, _c, {ldc});".format_map(
            opt
        )

        tasklet = dace.sdfg.nodes.Tasklet(
            node.name,
            node.in_connectors,
            node.out_connectors,
            code,
            language=dace.dtypes.Language.CPP,
        )
        return tasklet


@dace.library.expansion
class ExpandSyrkMKL(ExpandTransformation):
    environments = [intel_mkl.IntelMKL]

    @staticmethod
    def expansion(*args, **kwargs):
        return ExpandSyrkOpenBLAS.expansion(*args, **kwargs)


@dace.library.node
class Syrk(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "MKL": ExpandSyrkMKL,
        "OpenBLAS": ExpandSyrkOpenBLAS,
    }
    default_implementation = None

    # Object fields
    uplo = properties.Property(
        dtype=str,
        desc="Specifies whether the upper or lower triangular part of the array c is used ('u', 'l)",
    )
    trans = properties.Property(dtype=str, desc="")
    alpha = properties.Property(allow_none=False, default=1, desc="")
    beta = properties.Property(allow_none=False, default=0, desc="")

    def __init__(self, name, location=None, uplo="U", trans="N", alpha=1, beta=0):
        super().__init__(
            name,
            location=location,
            inputs=({"_a", "_cin"} if beta != 0 else {"_a"}),
            outputs={"_c"},
        )
        self.uplo = uplo
        self.trans = trans
        self.alpha = alpha
        self.beta = beta
        self._cin = beta != 0

    def validate(self, sdfg, state):
        assert self.uplo in ["L", "l", "U", "u"]
        assert self.trans in ["N", "n", "T", "t", "C", "c"]

        in_edges = state.in_edges(self)
        assert len(in_edges) == 1 and not self._cin or len(in_edges) == 2 and self._cin

        out_edges = state.out_edges(self)
        assert len(out_edges) == 1

        size_a = None
        size_cin = None
        for _, _, _, dst_conn, memlet in state.in_edges(self):
            if dst_conn == "_a":
                subset = copy.deepcopy(memlet.subset)
                subset.squeeze()
                size_a = []
                for dim in subset.size():
                    if dace.symbolic.issymbolic(dim, sdfg.constants):
                        size_a.append(dim)
                    else:
                        size_a.append(dace.symbolic.evaluate(dim, sdfg.constants))
            if dst_conn == "_cin":
                subset = copy.deepcopy(memlet.subset)
                subset.squeeze()
                size_cin = []
                for dim in subset.size():
                    if dace.symbolic.issymbolic(dim, sdfg.constants):
                        size_cin.append(dim)
                    else:
                        size_cin.append(dace.symbolic.evaluate(dim, sdfg.constants))

        assert size_a is not None

        if self.trans in ["T", "t", "C", "c"]:
            size_a = list(size_a[::-1])

        out_memlet = out_edges[0].data
        out_subset = copy.deepcopy(out_memlet.subset)
        out_subset.squeeze()
        size_cout = []
        for dim in out_subset.size():
            if dace.symbolic.issymbolic(dim, sdfg.constants):
                size_cout.append(dim)
            else:
                size_cout.append(dace.symbolic.evaluate(dim, sdfg.constants))

        assert len(size_cout) == 2 and size_cout[0] == size_cout[1]
        assert size_cin is None or size_cin == size_cout
        assert len(size_a) == 2 and size_a[0] == size_cout[0]
