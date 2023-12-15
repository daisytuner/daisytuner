import dace


def test_single_dimension():
    N = dace.symbol("N")

    @dace.program
    def sdfg_single_dimension(A: dace.float32[N], B: dace.float32[N]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> B[i]

                b = a

    sdfg = sdfg_single_dimension.to_sdfg()

    input_arrays = set()
    output_arrays = set()
    for dnode in sdfg.start_state.data_nodes():
        if sdfg.start_state.out_degree(dnode) == 0:
            output_arrays.add(dnode.data)
        elif sdfg.start_state.in_degree(dnode) == 0:
            input_arrays.add(dnode.data)

    from daisytuner.analysis.polyhedral import ScopAnalysis
    from daisytuner.analysis.polyhedral.codegen import ASTBuilder, ASTWalker

    # Scop analysis
    scop, analysis = ScopAnalysis.create(sdfg)
    deps = scop.dependency_analysis()

    import islpy as isl

    ctx = scop.domain.get_ctx()
    ctx.set_schedule_algorithm(isl.schedule_algorithm.ISL)

    # Compute a basic schedule
    ctx.set_schedule_serialize_sccs(False)
    ctx.set_schedule_maximize_band_depth(True)
    ctx.set_schedule_outer_coincidence(True)
    ctx.set_schedule_maximize_coincidence(True)
    ctx.set_schedule_whole_component(False)
    ctx.set_tile_scale_tile_loops(False)
    ctx.set_ast_build_atomic_upper_bound(True)
    ctx.set_ast_build_detect_min_max(True)
    ctx.set_ast_build_prefer_pdiv(True)

    sc = isl.ScheduleConstraints.on_domain(scop.domain.copy())
    sc.set_context(isl.Set.empty(scop.domain.get_space()))

    sc = sc.set_validity(deps.copy())
    sc = sc.set_coincidence(deps.copy())
    sc = sc.set_proximity(deps.copy())

    schedule = sc.compute_schedule()

    # Generate AST
    ast = ASTBuilder.get_ast_from_schedule(deps.copy(), schedule)

    # Genrate SDFG
    generated_sdfg = ASTWalker.parse(
        name="test",
        ast=ast,
        scop=scop,
        analysis=analysis,
        input_arrays=input_arrays,
        output_arrays=output_arrays,
    )

    # Validate
    generated_sdfg.validate()


# def test_multi_dimension():
#     N = dace.symbol("N")
#     M = dace.symbol("M")

#     @dace.program
#     def sdfg_multi_dimension(A: dace.float32[N, M], B: dace.float32[M, N]):
#         for i, j in dace.map[0:N, 0:M]:
#             with dace.tasklet:
#                 a << A[i, j]
#                 b >> B[j, i]

#                 b = a

#     sdfg = sdfg_multi_dimension.to_sdfg()

#     input_arrays = set()
#     output_arrays = set()
#     for dnode in sdfg.start_state.data_nodes():
#         if sdfg.start_state.out_degree(dnode) == 0:
#             output_arrays.add(dnode.data)
#         elif sdfg.start_state.in_degree(dnode) == 0:
#             input_arrays.add(dnode.data)

#     from daisytuner.analysis.polyhedral import ScopAnalysis
#     from daisytuner.analysis.polyhedral.codegen import ASTBuilder, ASTWalker

#     # Scop analysis
#     scop, analysis = ScopAnalysis.create(sdfg)
#     deps = scop.dependency_analysis()

#     ctx = scop.domain.get_ctx()
#     ctx.set_schedule_algorithm(isl.schedule_algorithm.ISL)

#     # Compute a basic schedule
#     ctx.set_schedule_serialize_sccs(False)
#     ctx.set_schedule_maximize_band_depth(True)
#     ctx.set_schedule_outer_coincidence(True)
#     ctx.set_schedule_maximize_coincidence(True)
#     ctx.set_schedule_whole_component(False)
#     ctx.set_tile_scale_tile_loops(False)
#     ctx.set_ast_build_atomic_upper_bound(True)
#     ctx.set_ast_build_detect_min_max(True)
#     ctx.set_ast_build_prefer_pdiv(True)

#     sc = isl.ScheduleConstraints.on_domain(scop.domain.copy())
#     sc.set_context(isl.Set.empty(scop.domain.get_space()))

#     sc = sc.set_validity(deps.copy())
#     sc = sc.set_coincidence(deps.copy())
#     sc = sc.set_proximity(deps.copy())

#     schedule = sc.compute_schedule()

#     # Generate AST
#     ast = ASTBuilder.get_ast_from_schedule(deps.copy(), schedule)

#     # Genrate SDFG
#     generated_sdfg = ASTWalker.parse(
#         name="test",
#         ast=ast,
#         scop=scop,
#         analysis=analysis,
#         input_arrays=input_arrays,
#         output_arrays=output_arrays,
#     )

#     # Validate
#     generated_sdfg.validate()
