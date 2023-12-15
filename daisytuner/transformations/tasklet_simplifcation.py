# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import re

from dace.sdfg import SDFG
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.state import SDFGState
from dace.transformation import transformation
from dace.properties import make_properties


@make_properties
class TaskletSimplification(transformation.SingleStateTransformation):
    tasklet = transformation.PatternNode(nodes.Tasklet)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.tasklet)]

    def can_be_applied(
        self, state: dace.SDFGState, expr_index: int, sdfg: dace.SDFG, permissive=False
    ):
        if self.tasklet.language != dace.Language.CPP:
            return False

        cpp_code = self.tasklet.code.as_string
        if cpp_code.count("=") != 1:
            return False
        lhs, rhs = cpp_code.split("=")
        rhs = rhs.replace(";", "")

        for in_conn in self.tasklet.in_connectors:
            p1 = r"\s\(%s\)" % in_conn
            p2 = r"\(\(%s\)" % in_conn

            if re.findall(p1, rhs) or re.findall(p2, rhs):
                return True

        rhs = rhs.strip()
        if rhs.startswith("(") and rhs.endswith(")"):
            ps = 1
            for s in rhs[1:-1]:
                if s == "(":
                    ps += 1
                elif s == ")":
                    ps -= 1

                if ps == 0:
                    return False

            return True

        return False

    def apply(self, state: SDFGState, sdfg: SDFG):
        cpp_code = self.tasklet.code.as_string
        lhs, rhs = cpp_code.split("=")
        lhs = lhs.strip()
        rhs = rhs.replace(";", "")

        for in_conn in self.tasklet.in_connectors:
            p1 = r"\s\(%s\)" % in_conn
            rhs = re.sub(p1, f" {in_conn}", rhs)
            p2 = r"\(\(%s\)" % in_conn
            rhs = re.sub(p2, f"({in_conn}", rhs)

        rhs = rhs.strip()
        if rhs.startswith("(") and rhs.endswith(")"):
            ps = 1
            for s in rhs[1:-1]:
                if s == "(":
                    ps += 1
                elif s == ")":
                    ps -= 1

                if ps == 0:
                    break

            if ps == 1:
                rhs = rhs[1:-1]

        stripped_code = lhs + " = " + rhs + ";"
        self.tasklet.code.code = stripped_code
