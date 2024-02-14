import copy
import islpy as isl
import sympy as sp

from typing import Set

from dace import SDFG
from dace import sdfg as sd, SDFG, Memlet, subsets, dtypes, nodes
from dace.subsets import Range
from dace.symbolic import pystr_to_symbolic
from dace.frontend.python.astutils import negate_expr

from daisytuner.analysis.polyhedral.scop import Scop
from daisytuner.analysis.polyhedral.scop_analysis import ScopAnalysis
from daisytuner.analysis.polyhedral.scop_analysis import _create_constrained_set
from daisytuner.analysis.polyhedral.sympy2isl import (
    extract_end_cond,
    to_sympy,
    sympy_to_pystr,
)


class ASTWalker:
    def __init__(
        self,
        sdfg: SDFG,
        scop: Scop,
        analysis: ScopAnalysis,
        input_arrays,
        output_arrays,
    ):
        self.sdfg = sdfg
        self.scop = scop
        self.analysis = analysis
        self.inputs = {}
        self.outputs = {}
        self.input_arrays = input_arrays
        self.output_arrays = output_arrays

    @staticmethod
    def parse(
        name: str,
        ast: isl.AstNode,
        scop: Scop,
        analysis: ScopAnalysis,
        input_arrays: Set[dtypes.typeclass],
        output_arrays: Set[dtypes.typeclass],
    ):
        sdfg = SDFG(name=name)

        # Adding symbols
        for sym, stype in analysis.sdfg.symbols.items():
            sdfg.add_symbol(sym, stype)
        for array in input_arrays:
            sdfg.add_datadesc(array, copy.deepcopy(analysis.sdfg.arrays[array]))
        for array in output_arrays:
            sdfg.add_datadesc(array, copy.deepcopy(analysis.sdfg.arrays[array]))

        sdfg.specialize(analysis.sdfg.constants)

        init = sdfg.add_state("init", is_start_state=True)
        final = sdfg.add_state("end_state", is_start_state=False)

        pv = ASTWalker(
            sdfg=sdfg,
            scop=scop,
            analysis=analysis,
            input_arrays=input_arrays,
            output_arrays=output_arrays,
        )
        first, last = pv._visit(ast, [], [])

        sdfg.add_edge(init, first, sd.InterstateEdge())
        sdfg.add_edge(last, final, sd.InterstateEdge())
        return sdfg

    def _visit(self, ast_node, loop_ranges, constraints):
        """
        Visit a AST node.
        """
        if ast_node.get_type() == isl.ast_node_type.block:
            first, last = self._visit_block(ast_node, loop_ranges, constraints)
        elif ast_node.get_type() == isl.ast_node_type.for_:
            first, last = self._visit_for(ast_node, loop_ranges, constraints)
        elif ast_node.get_type() == isl.ast_node_type.if_:
            first, last = self._visit_if(ast_node, loop_ranges, constraints)
        elif ast_node.get_type() == isl.ast_node_type.user:
            first, last = self._visit_user(ast_node, loop_ranges, constraints)
        else:
            raise NotImplementedError
        return first, last

    def _visit_block(self, ast_node, loop_ranges, constraints):
        """
        Visit a AST block node.
        """
        node_list = ast_node.block_get_children()
        n_children = node_list.n_ast_node()
        states = []
        for child_node in [node_list.get_at(i) for i in range(n_children)]:
            ret_val = self._visit(child_node, loop_ranges.copy(), constraints)
            s1, s2 = ret_val

            states.append((s1, s2))
        for (_, s1), (s2, _) in zip(states[:-1], states[1:]):
            self.sdfg.add_edge(s1, s2, sd.InterstateEdge())
        return states[0][0], states[-1][1]

    def _visit_for(self, ast_node, loop_ranges, constraints):
        """
        Visit a AST for node.
        """

        iter_sympy = to_sympy(ast_node.for_get_iterator())
        iterator_var = sympy_to_pystr(iter_sympy)

        init_sympy = to_sympy(ast_node.for_get_init())
        init_str = sympy_to_pystr(init_sympy)

        cond_sympy = to_sympy(ast_node.for_get_cond())
        end_sympy = extract_end_cond(cond_sympy, iter_sympy)
        condition_str = sympy_to_pystr(cond_sympy)

        step_sym = to_sympy(ast_node.for_get_inc())
        incr_str = sympy_to_pystr(sp.Add(iter_sympy, step_sym))

        loop_rng = subsets.Range([(init_sympy, end_sympy, step_sym)])
        loop_ranges.append((iterator_var, loop_rng))

        stmt_domain = _create_constrained_set(
            ctx=self.analysis.ctx,
            params=self.analysis.symbols,
            constants=self.analysis.constants,
            constr_ranges=loop_ranges,
            constraints=constraints,
        )
        stmt_domain = stmt_domain.coalesce()

        while stmt_domain.n_dim() > 1:
            stmt_domain = stmt_domain.move_dims(
                isl.dim_type.param, 0, isl.dim_type.set, 0, 1
            )

        is_parallel = False
        if True:
            build = ast_node.get_annotation().user.build
            part_schedule = build.get_schedule()
            deps = self.scop.dependency_analysis()
            is_parallel = ASTBuilder.is_parallel(part_schedule, deps)
            # is_parallel = ast_node.get_annotation().user.is_parallel
        if is_parallel:
            state = self.sdfg.add_state("MapState")
            subset = loop_rng

            map_nodes = nodes.Map(label="map", params=[iterator_var], ndrange=subset)

            entry = nodes.MapEntry(map_nodes)
            exit = nodes.MapExit(map_nodes)
            state.add_nodes_from([entry, exit])

            # create a new SDFG for the map body
            body_sdfg = SDFG("{}_body".format(entry.label))

            # add all arrays of SDFG to the body-SDFG
            # all transient become False
            for arr_label, arr in self.sdfg.arrays.items():
                arr_copy = copy.deepcopy(arr)
                arr_copy.transient = False
                body_sdfg.add_datadesc(arr_label, arr_copy)

            body_sdfg.symbols.update(self.sdfg.symbols)

            # walk and add the states to the body_sdfg
            pv = ASTWalker(
                sdfg=body_sdfg,
                scop=self.scop,
                analysis=self.analysis,
                input_arrays=self.input_arrays,
                output_arrays=self.output_arrays,
            )

            _, _ = pv._visit(ast_node.for_get_body(), loop_ranges.copy(), constraints)

            body_inputs = {c: m for c, m in pv.inputs.items()}
            body_outputs = {c: m for c, m in pv.outputs.items()}

            for arr in {
                a
                for a in body_sdfg.arrays
                if a not in body_inputs and a not in body_outputs
            }:
                body_sdfg.remove_data(arr)

            body = state.add_nested_sdfg(
                body_sdfg, self.sdfg, body_inputs.keys(), body_outputs.keys()
            )

            for arr_name, in_mem in body_inputs.items():
                if arr_name not in body.in_connectors:
                    continue
                read_node = state.add_read(arr_name)
                arr = body_sdfg.arrays[arr_name]
                subset = Range.from_array(arr)

                memlet = Memlet(data=arr_name, subset=subset)

                state.add_memlet_path(
                    read_node,
                    entry,
                    body,
                    memlet=memlet,
                    dst_conn=arr_name,
                    propagate=True,
                )
            if len(body.in_connectors) == 0:
                state.add_edge(entry, None, body, None, Memlet())

            for arr_name, out_mem in body_outputs.items():
                if arr_name not in body.out_connectors:
                    continue
                write_node = state.add_write(arr_name)
                arr = body_sdfg.arrays[arr_name]
                subset = Range.from_array(arr)

                memlet = Memlet(data=arr_name, subset=subset)

                state.add_memlet_path(
                    body,
                    exit,
                    write_node,
                    memlet=memlet,
                    src_conn=arr_name,
                    dst_conn=None,
                    propagate=True,
                )
            if len(body.out_connectors) == 0:
                state.add_edge(body, None, exit, None, Memlet())

            self.inputs.update(body_inputs)
            self.outputs.update(body_outputs)
            return state, state
        else:
            body_begin, body_end = self._visit(
                ast_node.for_get_body(), loop_ranges.copy(), constraints
            )

            if iterator_var not in self.sdfg.symbols:
                self.sdfg.add_symbol(iterator_var, dtypes.int64)

            if body_begin == body_end:
                body_end = None

            loop_result = self.sdfg.add_loop(
                before_state=None,
                loop_state=body_begin,
                loop_end_state=body_end,
                after_state=None,
                loop_var=iterator_var,
                initialize_expr=init_str,
                condition_expr=condition_str,
                increment_expr=incr_str,
            )
            before_state, guard, after_state = loop_result
            return before_state, after_state

    def _visit_if(self, ast_node, loop_ranges, constraints):
        """
        Visit an AST if node.
        """
        # Add a guard state
        if_guard = self.sdfg.add_state("if_guard")
        end_if_state = self.sdfg.add_state("end_if")

        # Generate conditions
        if_cond_sym = to_sympy(ast_node.if_get_cond())
        if_cond_str = sympy_to_pystr(if_cond_sym)
        else_cond_sym = negate_expr(if_cond_sym)
        else_cond_str = sympy_to_pystr(else_cond_sym)

        then_node = ast_node.if_get_then_node()
        if_constraints = constraints.copy()
        if_constraints.append(if_cond_sym)
        first_if_state, last_if_state = self._visit(
            then_node, loop_ranges.copy(), if_constraints
        )

        # Connect the states
        self.sdfg.add_edge(if_guard, first_if_state, sd.InterstateEdge(if_cond_str))
        self.sdfg.add_edge(last_if_state, end_if_state, sd.InterstateEdge())

        if ast_node.if_has_else_node():
            else_node = ast_node.if_get_else_node()
            else_constraints = constraints.copy()
            else_constraints.append(else_cond_sym)
            first_else_state, last_else_state = self._visit(
                else_node, loop_ranges.copy(), else_constraints
            )

            # Connect the states
            self.sdfg.add_edge(
                if_guard, first_else_state, sd.InterstateEdge(else_cond_str)
            )
            self.sdfg.add_edge(last_else_state, end_if_state, sd.InterstateEdge())
        else:
            self.sdfg.add_edge(if_guard, end_if_state, sd.InterstateEdge(else_cond_str))
        return if_guard, end_if_state

    def _visit_user(self, ast_node, loop_ranges, constraints):
        """
        Visit an AST user node.
        """
        ast_expr = ast_node.user_get_expr()
        if ast_expr.get_op_type() == isl.ast_expr_op_type.call:
            stmt_name = ast_expr.get_op_arg(0).to_C_str()
            state = self.sdfg.add_state("state")
            (tasklet, iter_vars, in_data, out_data) = self.analysis.tasklets[stmt_name]
            tasklet = copy.deepcopy(tasklet)

            repl_dict = {}
            for i, var in enumerate(iter_vars):
                old_sym = pystr_to_symbolic(var)
                new_var = ast_expr.get_op_arg(i + 1).to_C_str()
                new_sym = pystr_to_symbolic(new_var)
                repl_dict[old_sym] = new_sym

            state.add_node(tasklet)
            for conn, out_mem in out_data.items():
                arr_name = out_mem.data
                new_subset = copy.deepcopy(out_mem.subset)
                new_subset.replace(repl_dict)
                write_node = state.add_write(arr_name)
                memlet = Memlet(data=arr_name, subset=new_subset)
                state.add_edge(tasklet, conn, write_node, None, memlet)
                self.outputs[arr_name] = copy.deepcopy(memlet)
            for conn, in_mem in in_data.items():
                arr_name = in_mem.data
                new_subset = copy.deepcopy(in_mem.subset)
                new_subset.replace(repl_dict)

                read_node = state.add_read(arr_name)
                memlet = Memlet(data=arr_name, subset=new_subset)
                self.inputs[arr_name] = copy.deepcopy(memlet)
                state.add_edge(read_node, None, tasklet, conn, memlet)

            return state, state
        return None, None, None


class ASTBuilder:
    """
    Helper Class to generate a synthetic AST from the Scop.
    Functionality for tiling and detecting parallelism in synthetic AST.
    """

    deps = [None]

    class UserInfo:
        def __init__(self):
            # Loops is parallel
            self.is_parallel = False
            self.build = None
            self.schedule = None
            self.domain = None

    @staticmethod
    def at_each_domain(node: isl.AstNode, build: isl.AstBuild):
        """
        Annotated each node in the AST with the domain and partial schedule
        """
        info = ASTBuilder.UserInfo()
        id = isl.Id.alloc(ctx=isl.AstBuild.get_ctx(build), name="", user=info)
        info.build = isl.AstBuild.copy(build)
        info.schedule = build.get_schedule()
        info.domain = info.schedule.domain()
        node.set_annotation(id)
        return node

    @staticmethod
    def before_each_for(build: isl.AstBuild):
        """
        Detection of parallel loops.
        This function is called for each for in depth-first pre-order.
        """

        # A (partial) schedule for the domains elements for which part of
        # the AST still needs to be generated in the current build.
        # The domain elements are mapped to those iterations of the loops
        # enclosing the current point of the AST generation inside which
        # the domain elements are executed.
        part_sched = build.get_schedule()
        info = ASTBuilder.UserInfo()

        # Test for parallelism
        info.is_parallel = ASTBuilder.is_parallel(part_sched, ASTBuilder.deps[0])
        info.build = isl.AstBuild.copy(build)
        info.schedule = part_sched
        info.domain = part_sched.domain()

        return isl.Id.alloc(ctx=build.get_ctx(), name="", user=info)

    @staticmethod
    def is_parallel(part_sched: isl.UnionMap, stmt_deps: isl.UnionMap) -> bool:
        """
        Check if the current scheduling dimension is parallel by verifying that
        the loop does not carry any dependencies.

        :param part_sched: A partial schedule
        :param stmt_deps: The dependencies between the statements
        :return True if current the scheduling dimension is parallel, else False
        """

        # translate the dependencies into time-space, by applying part_sched
        time_deps = stmt_deps.apply_range(part_sched).apply_domain(part_sched)

        # the loop is parallel, if there are no dependencies in time-space
        if time_deps.is_empty():
            return True

        time_deps = isl.Map.from_union_map(time_deps)
        time_deps = time_deps.flatten_domain().flatten_range()

        curr_dim = time_deps.dim(isl.dim_type.set) - 1
        # set all dimension in the time-space equal, except the current one:
        # if the distance in all outer dimensions is zero, then it
        # has to be zero in the current dimension as well to be parallel
        for i in range(curr_dim):
            time_deps = time_deps.equate(isl.dim_type.in_, i, isl.dim_type.out, i)

        # computes a delta set containing the differences between image
        # elements and corresponding domain elements in the time_deps.
        time_deltas = time_deps.deltas()

        # the loop is parallel, if there are no deltas in the time-space
        if time_deltas.is_empty():
            return True

        # The loop is parallel, if the distance is zero in the current dimension
        delta = time_deltas.plain_get_val_if_fixed(isl.dim_type.set, curr_dim)
        return delta.is_zero()

    @staticmethod
    def get_annotation_build(ctx: isl.Context, deps: isl.UnionMap) -> isl.AstBuild:
        """
        helper function that return an isl.AstBuild
        """
        build = isl.AstBuild.alloc(ctx)
        # callback at_each_domain will be called for each domain AST node
        build, _ = build.set_at_each_domain(ASTBuilder.at_each_domain)
        ASTBuilder.deps = [deps]
        # callback before_each_for be called in depth-first pre-order
        build, _ = build.set_before_each_for(ASTBuilder.before_each_for)
        return build

    @staticmethod
    def get_ast_from_schedule(
        deps: isl.UnionMap, schedule: isl.Schedule, tile_size: int = 0
    ) -> isl.AstNode:
        """
        Compute a synthetic AST from the dependencies and a schedule tree
        :param deps: The dependencies to use
        :param schedule: The schedule to use
        :param tile_size: If tile_size>1 perform tiling on synthetic AST
        :return: The root of the generated synthetic AST
        """

        ctx = schedule.get_ctx()
        build = ASTBuilder.get_annotation_build(ctx, deps)

        def ast_build_options(node: isl.AstNode) -> isl.AstNode:
            """
            Sets the building options for this node in the AST
            """
            if node.get_type() != isl.schedule_node_type.band:
                return node
            # options: separate | atomic | unroll | default
            option = isl.UnionSet.read_from_str(node.get_ctx(), "{default[x]}")
            node = node.band_set_ast_build_options(option)
            return node

        def tile_band(node: isl.AstNode) -> isl.AstNode:
            """
            Tile this node of the AST if possible
            """
            # return if not tileable
            if node.get_type() != isl.schedule_node_type.band:
                return node
            elif node.n_children() != 1:
                return node
            elif not node.band_get_permutable():
                return node
            elif node.band_n_member() <= 1:
                return node
            n_dim = node.band_get_space().dim(isl.dim_type.set)
            if n_dim <= 1:
                return node

            tile_multi_size = isl.MultiVal.zero(node.band_get_space())
            for i in range(n_dim):
                tile_multi_size = tile_multi_size.set_val(i, isl.Val(tile_size))

            # tile the current band with tile_size
            node = node.band_tile(tile_multi_size)
            # mark all dimensions in the band node to be "atomic"
            for i in range(node.band_n_member()):
                atomic_type = isl.ast_loop_type.atomic
                node = node.band_member_set_ast_loop_type(i, atomic_type)
            return node

        root = schedule.get_root()
        root = root.map_descendant_bottom_up(ast_build_options)
        if tile_size >= 1:
            root = root.map_descendant_bottom_up(tile_band)
        schedule = root.get_schedule()
        root = build.node_from_schedule(schedule)
        return root

    @staticmethod
    def get_ast_from_schedule_map(deps, schedule_map):
        """
        :param deps: :class:`UnionMap`
        :param schedule_map: :class:`UnionMap`
        :return: :class:`AstBuild`
        """

        ctx = schedule_map.get_ctx()
        ctx.set_ast_build_detect_min_max(True)  # True
        # options that control how AST is created from the individual schedule
        # dimensions are stored in build
        build = ASTBuilder.get_annotation_build(ctx, deps)
        # generates an AST
        root = build.node_from_schedule_map(schedule_map)
        return root
