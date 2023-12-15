import islpy as isl
import sympy as sp

from collections import defaultdict
from typing import List, Dict, Tuple, Set, Union

import dace.serialize

from dace import dtypes, symbolic
from dace.sdfg import nodes
from dace.subsets import Range, Subset
from dace.codegen import control_flow as cflow
from dace.frontend.python.astutils import negate_expr
from dace.symbolic import pystr_to_symbolic

from daisytuner.analysis.polyhedral.sympy2isl import (
    extract_end_cond,
    extract_step_cond,
    SympyToPwAff,
)
from daisytuner.analysis.polyhedral.scop import Scop
from daisytuner.pipelines.loopification import Loopification


class ScopAnalysis:
    """
    Performs scop analysis for SDFGs.
    """

    def __init__(self, sdfg: dace.SDFG):
        self._duplicate_name_cnt = defaultdict(int)
        self._name_table = dict()
        self.tasklets = dict()
        self.ctx = isl.Context()

        self.sdfg = sdfg
        self.symbols = self.sdfg.free_symbols.copy()
        self.constants = {}
        for k, v in sdfg.constants.items():
            self.constants[k] = pystr_to_symbolic(str(v))

    def _traverse(
        self,
        node: cflow.ControlFlow,
        constraints: List[dtypes.typeclass],
        loop_ranges: List[Tuple[str, Range]],
        replace_vars: Dict[dtypes.typeclass, dtypes.typeclass],
    ) -> Scop:
        if isinstance(node, cflow.GeneralBlock):
            # visit child and make sequence
            seq_poly = Scop()
            for child in node.children:
                next_poly = self._traverse(
                    node=child,
                    constraints=constraints.copy(),
                    loop_ranges=loop_ranges.copy(),
                    replace_vars=replace_vars.copy(),
                )
                seq_poly.sequence(next_poly)
            return seq_poly
        elif isinstance(node, cflow.SingleState):
            state = node.state
            for orig_sym, transformed_sym in replace_vars.items():
                state.replace(str(orig_sym), str(transformed_sym))
            code_nodes = [n for n in state.nodes() if isinstance(n, nodes.CodeNode)]
            assert all(
                (isinstance(node, nodes.Tasklet) for node in code_nodes)
            ), "Scop analysis currently does not support nested SDFGs"

            loop_vars = [v for v, r in loop_ranges]

            node_poly = Scop()
            for cn in code_nodes:
                stmt_name = self._get_stmt_name(state.label, cn.label)
                cn.label = stmt_name

                stmt_poly = Scop()
                stmt_domain = _create_constrained_set(
                    ctx=self.ctx,
                    params=self.symbols,
                    constants=self.constants,
                    constr_ranges=loop_ranges,
                    constraints=constraints,
                    set_name=stmt_name,
                )
                stmt_poly.domain = isl.UnionSet.from_set(stmt_domain)
                stmt_poly.schedule = isl.Schedule.from_domain(stmt_domain)
                stmt_poly.write = isl.UnionMap.empty(stmt_domain.get_space())
                stmt_poly.read = isl.UnionMap.empty(stmt_domain.get_space())

                out_data = dict()
                read_set, write_set = state.read_and_write_sets()
                for e in state.out_edges(cn):
                    if e.data.wcr:
                        continue
                    access_name = e.data.data
                    # TODO
                    if isinstance(e.data.subset, dace.subsets.Indices):
                        access_sympy = [
                            pystr_to_symbolic(str(a)) for a in e.data.subset.indices
                        ]
                    else:
                        access_sympy = [
                            pystr_to_symbolic(str(b))
                            for (b, _, _) in e.data.subset.ranges
                        ]
                    if access_name in write_set:
                        write = _create_access_map(
                            ctx=self.ctx,
                            params=self.symbols,
                            variables=loop_vars,
                            constants=self.constants,
                            accesses=access_sympy,
                            stmt_name=stmt_name,
                            access_name=access_name,
                        )
                        stmt_poly.write = stmt_poly.write.union(write)
                        out_data[e.src_conn] = e.data

                in_data = dict()
                for e in state.in_edges(cn):
                    if e.data.wcr:
                        continue
                    access_name = e.data.data
                    # TODO
                    if isinstance(e.data.subset, dace.subsets.Indices):
                        access_sympy = [
                            pystr_to_symbolic(str(a)) for a in e.data.subset.indices
                        ]
                    else:
                        access_sympy = [
                            pystr_to_symbolic(str(b))
                            for (b, _, _) in e.data.subset.ranges
                        ]
                    if access_name in read_set:
                        read = _create_access_map(
                            ctx=self.ctx,
                            params=self.symbols,
                            variables=loop_vars,
                            constants=self.constants,
                            accesses=access_sympy,
                            stmt_name=stmt_name,
                            access_name=access_name,
                        )
                        stmt_poly.read = stmt_poly.read.union(read)
                        in_data[e.dst_conn] = e.data

                self.tasklets[stmt_name] = (cn, loop_vars, in_data, out_data)
                node_poly.sequence(stmt_poly)

            return node_poly
        elif isinstance(node, cflow.ForScope):
            itersym = pystr_to_symbolic(node.itervar)
            start = pystr_to_symbolic(node.init)
            step_cond = pystr_to_symbolic(node.update)
            step = extract_step_cond(step_cond, itersym)
            end_cond = pystr_to_symbolic(node.condition.as_string)
            end = extract_end_cond(end_cond, itersym)
            # Normalize the loop: start <= end and step > 0
            for orig_sym, transformed_sym in replace_vars.items():
                start = start.replace(orig_sym, transformed_sym)
                end = end.replace(orig_sym, transformed_sym)
            if step < 0:
                step = -step
                start, end = end, start
                shift = start
                start -= shift
                end -= shift
                itersym_transformed = (end - itersym) + shift
                replace_vars[itersym] = itersym_transformed
            loop_ranges.append((node.itervar, Range([(start, end, step)])))
            loop_poly = self._traverse(
                node=node.body,
                constraints=constraints,
                loop_ranges=loop_ranges,
                replace_vars=replace_vars,
            )
            depth = len(loop_ranges)
            loop_poly.schedule = _add_loop_dim(loop_poly.schedule, depth)
            return loop_poly
        elif isinstance(node, cflow.IfScope):
            if_condition = pystr_to_symbolic(node.condition.as_string)
            for orig_sym, transformed_sym in replace_vars.items():
                if_condition = if_condition.replace(orig_sym, transformed_sym)
            if_constraints = constraints.copy()
            if_constraints.append(if_condition)
            if_poly = self._traverse(
                node=node.body,
                constraints=if_constraints,
                loop_ranges=loop_ranges.copy(),
                replace_vars=replace_vars.copy(),
            )
            if node.orelse:
                else_condition = negate_expr(if_condition)
                else_constraints = constraints.copy()
                else_constraints.append(else_condition)
                else_poly = self._traverse(
                    node=node.orelse,
                    constraints=else_constraints,
                    loop_ranges=loop_ranges.copy(),
                    replace_vars=replace_vars.copy(),
                )
                if_poly.union(else_poly)
            return if_poly
        elif isinstance(node, cflow.IfElseChain):
            raise NotImplementedError
        elif isinstance(node, cflow.WhileScope):
            raise NotImplementedError
        elif isinstance(node, cflow.DoWhileScope):
            raise NotImplementedError
        elif isinstance(node, cflow.SwitchCaseScope):
            raise NotImplementedError

    def _get_stmt_name(self, *args):
        name = "_".join(args)
        dup_cnt = self._duplicate_name_cnt[name]
        self._duplicate_name_cnt[name] += 1
        if dup_cnt:
            name += "_{}".format(dup_cnt)
        return name

    @staticmethod
    def create(sdfg: dace.SDFG):
        results = {}
        pipeline = Loopification()
        pipeline.apply_pass(sdfg, results)

        analysis = ScopAnalysis(sdfg)
        cft = cflow.structured_control_flow_tree(sdfg, lambda _: "")
        return analysis._traverse(cft, [], [], {}), analysis


def _add_loop_dim(schedule: isl.Schedule, n: int) -> isl.Schedule:
    sched_dom = schedule.get_domain()
    loop_dim = isl.UnionPwMultiAff.empty(sched_dom.get_space())
    dom_list = [
        sched_dom.get_set_list().get_at(i)
        for i in range(0, sched_dom.get_set_list().n_set())
    ]

    # map from all domains to its nth parameter. e.g. {A[i,j] -> [j]} for n=2
    for dom in dom_list:
        dim = dom.dim(isl.dim_type.set)
        pma = isl.PwMultiAff.project_out_map(
            dom.get_space(), isl.dim_type.set, n, dim - n
        )
        if n > 1:
            pma = pma.drop_dims(isl.dim_type.out, 0, n - 1)
        loop_dim = loop_dim.add_pw_multi_aff(pma)
    loop_sched = isl.MultiUnionPwAff.from_union_pw_multi_aff(loop_dim)
    schedule = schedule.insert_partial_schedule(loop_sched)
    return schedule


def _create_access_map(
    ctx: isl.Context,
    params: Set[str],
    variables: List[Union[str, symbolic.symbol]],
    constants: Dict[str, sp.Number],
    accesses: List[Union[str, symbolic.symbol]],
    stmt_name: str = None,
    access_name: str = None,
) -> isl.Map:
    space = isl.Space.create_from_names(ctx=ctx, set=variables, params=params)
    if stmt_name:
        space = space.set_tuple_name(isl.dim_type.set, stmt_name)
    sympy_to_pwaff = SympyToPwAff(space, constants)
    empty = isl.Map.empty(space)
    empty = empty.add_dims(isl.dim_type.in_, len(accesses)).reverse()
    mpa = isl.MultiPwAff.from_pw_multi_aff(isl.PwMultiAff.from_map(empty))
    for i, idx_expr in enumerate(accesses):
        idx_pwaff = sympy_to_pwaff.visit(idx_expr)
        mpa = mpa.set_pw_aff(i, idx_pwaff)
    access_map = isl.Map.from_multi_pw_aff(mpa)
    if access_name:
        access_map = access_map.set_tuple_name(isl.dim_type.set, access_name)
    return access_map


def _create_constrained_set(
    ctx: isl.Context,
    params: List[str],
    constants: Dict[str, sp.Number],
    constr_ranges: List[Tuple[str, Range]],
    constraints: List[dtypes.typeclass] = None,
    set_name: str = None,
    extra_var: List[str] = None,
) -> isl.Set:
    if not constraints:
        constraints = []
    loop_vars = [v for v, r in constr_ranges]
    params = [p for p in params if p not in loop_vars]
    if extra_var:
        for var in extra_var:
            if var not in loop_vars:
                loop_vars.append(var)
    space = isl.Space.create_from_names(ctx=ctx, set=loop_vars, params=params)
    pw_affs = isl.affs_from_space(space)
    domain = isl.Set.universe(space)
    for p, rng in constr_ranges:
        if isinstance(rng, Subset):
            start, end, step = rng.ranges[0]
        elif isinstance(rng, tuple) and len(rng) == 3:
            start, end, step = rng
        else:
            raise NotImplementedError
        lb = SympyToPwAff(space, constants=constants).visit(start)
        ub = SympyToPwAff(space, constants=constants).visit(end)
        assert isinstance(step, int) or (step.is_Integer and step.is_Integer)
        # start <= loop_dim <= end
        loop_condition = lb.le_set(pw_affs[p]) & pw_affs[p].le_set(ub)
        if step > 1 or step < -1:
            # create the stride condition
            init = lb.copy()
            dim = init.dim(isl.dim_type.in_)
            # add a temporary dimension for the stride
            init = isl.PwAff.add_dims(init, isl.dim_type.in_, 1)
            stride_space = init.get_domain_space()
            loop_dim = isl.affs_from_space(stride_space)[p]
            ls = isl.LocalSpace.from_space(stride_space)
            stride_dim = isl.PwAff.var_on_domain(ls, isl.dim_type.set, dim)
            scaled_stride_dim = stride_dim.scale_val(isl.Val(int(step)))
            # loop_dim = start + step * stride_dim
            stride = loop_dim.eq_set(init + scaled_stride_dim)
            # stride_dim >= 0
            stride = stride.lower_bound_val(isl.dim_type.set, dim, 0)
            # (start + loop_dim) mod step = 0
            stride = stride.project_out(isl.dim_type.set, dim, 1)
            loop_condition = loop_condition & stride
        domain = domain & loop_condition
    # add extra constrains (e.g. from if/else) to domain
    for cond_expr in constraints:
        condition = SympyToPwAff(space, constants=constants).visit(cond_expr)
        domain = domain & condition
    if set_name:
        domain = domain.set_tuple_name(set_name)
    domain = domain.coalesce()
    return domain
