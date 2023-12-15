import dace
import islpy as isl

from typing import Dict

from daisytuner.analysis.polyhedral.scop_analysis import ScopAnalysis
from daisytuner.analysis.polyhedral.codegen import ASTWalker, ASTBuilder
from daisytuner.passes.map_expanded_form import MapExpandedForm
from daisytuner.tuning.cutout_tuner import CutoutTuner


class PlutoTuner(CutoutTuner):
    def __init__(self, tile_size: int = 16) -> None:
        self._tile_size = tile_size

    def can_be_tuned(self, cutout: dace.SDFG) -> bool:
        if not MapExpandedForm.is_expanded_form(cutout):
            return False
        return True

    def tune(
        self, cutout: dace.SDFG, arguments: Dict = None, parallelize: bool = True
    ) -> dace.SDFG:
        input_arrays = set()
        output_arrays = set()
        for dnode in cutout.start_state.data_nodes():
            if cutout.start_state.out_degree(dnode) == 0:
                output_arrays.add(dnode.data)
            elif cutout.start_state.in_degree(dnode) == 0:
                input_arrays.add(dnode.data)

        scop, analysis = ScopAnalysis.create(cutout)
        dependencies = scop.dependency_analysis()

        params_constraints = isl.Set.empty(scop.domain.get_space())

        # The generated schedule respects all validity dependencies. That is,
        # all dependence distances over these dependencies in the scheduled
        # space are lexicographically positive. Mapping domain elements i to
        # domain elements that should schedule after i
        validity = dependencies.copy()

        # coincidence dependencies: mapping domain elements i to domain elements
        # that should be scheduled together with I, if possible
        coincidence = dependencies.copy()

        # minimize the dependence distances over proximity dependencies
        # mapping domain elements i to domain elements that should be
        # scheduled either before I or as early as possible after i
        proximity = dependencies.copy()

        ctx = scop.domain.get_ctx()
        ctx.set_schedule_algorithm(isl.schedule_algorithm.ISL)

        # Scheduler Options:
        ctx.set_schedule_serialize_sccs(False)
        ctx.set_schedule_maximize_band_depth(True)
        ctx.set_schedule_outer_coincidence(True)
        ctx.set_schedule_maximize_coincidence(True)
        ctx.set_schedule_whole_component(False)
        ctx.set_tile_scale_tile_loops(False)

        # AST Build Options:
        ctx.set_ast_build_atomic_upper_bound(True)
        ctx.set_ast_build_detect_min_max(True)
        ctx.set_ast_build_prefer_pdiv(True)

        sc = isl.ScheduleConstraints.on_domain(scop.domain.copy())

        # constraints on parameters to hold during construction of the schedule
        sc.set_context(params_constraints)

        try:
            # Try to simplify validity, coincidence and proximity
            # gist: simplify domain or range with respect to known constraints
            validity = validity.gist_domain(scop.domain)
            validity = validity.gist_range(scop.domain)
            validity = validity.coalesce()
            sc = sc.set_validity(validity)
            coincidence = coincidence.gist_domain(scop.domain)
            coincidence = coincidence.gist_range(scop.domain)
            coincidence = coincidence.coalesce()
            sc = sc.set_coincidence(coincidence)
            proximity = proximity.gist_domain(scop.domain)
            proximity = proximity.gist_range(scop.domain)
            proximity = proximity.coalesce()
            sc = sc.set_proximity(proximity)
            new_schedule = sc.compute_schedule()
        except:
            # simplification failed: continue without simplification
            sc = sc.set_validity(dependencies.copy())
            sc = sc.set_coincidence(dependencies.copy())
            sc = sc.set_proximity(dependencies.copy())
            new_schedule = sc.compute_schedule()

        new_ast = ASTBuilder.get_ast_from_schedule(
            dependencies, new_schedule, tile_size=self._tile_size
        )

        # Build new SDFG
        new_sdfg = ASTWalker.parse(
            name=cutout.name + "_pluto",
            ast=new_ast,
            scop=scop,
            analysis=analysis,
            input_arrays=input_arrays,
            output_arrays=output_arrays,
        )

        dace.propagate_memlets_sdfg(new_sdfg)
        new_sdfg.simplify()

        return new_sdfg
