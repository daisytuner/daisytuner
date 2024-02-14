# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import copy
import dace

from typing import Dict, Any, Union, Set, Type

from dace.transformation import pass_pipeline as ppl
from dace.sdfg.state import StateSubgraphView, MultiConnectorEdge

from daisytuner.passes.data_dependent_symbol_analysis import DataDependentSymbolAnalysis


class IndirectionPropagation(ppl.Pass):
    """ """

    CATEGORY: str = "Normalization"

    def __init__(self) -> None:
        super().__init__()

    def depends_on(self) -> Set[Type[ppl.Pass]]:
        return {DataDependentSymbolAnalysis}

    def modifies(self) -> ppl.Modifies:
        return (
            ppl.Modifies.AccessNodes
            | ppl.Modifies.Tasklets
            | ppl.Modifies.Scopes
            | ppl.Modifies.Memlets
        )

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.InterstateEdges

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Dict[str, Any]) -> int:
        nodes_updated = set()
        analysis: Dict[
            Union[dace.SDFG, dace.SDFGState], Dict[str, dace.Memlet]
        ] = pipeline_results[DataDependentSymbolAnalysis.__name__][sdfg]
        for node, mapping in analysis.items():
            if isinstance(node, dace.SDFG):
                nested_analysis = {
                    DataDependentSymbolAnalysis.__name__: {node: mapping}
                }
                pipeline = IndirectionPropagation()
                nested_results = pipeline.apply_pass(node, nested_analysis)
                if nested_results is not None:
                    nodes_updated.add(node)
            else:
                # Replace symbols by memlets
                for n in node.nodes():
                    if isinstance(n, dace.nodes.MapEntry):
                        if not n.free_symbols:
                            continue

                        used_symbols = n.free_symbols.intersection(mapping.keys())
                        if not used_symbols:
                            continue

                        IndirectionPropagation.indirect_map_ranges(
                            sdfg, node, n, mapping
                        )
                        nodes_updated.add(n)
                    elif isinstance(n, dace.nodes.Tasklet):
                        indirect_in_edges = []
                        for e in node.in_edges(n):
                            if e.data.volume != 1:
                                continue
                            if e.data.wcr is not None:
                                continue

                            used_symbols = e.data.free_symbols.intersection(
                                mapping.keys()
                            )
                            if used_symbols:
                                indirect_in_edges.append(e)

                        for e in indirect_in_edges:
                            IndirectionPropagation.indirect_read(
                                sdfg, node, n, e, mapping
                            )
                            nodes_updated.add(n)

        if nodes_updated:
            return nodes_updated
        else:
            return None

    @staticmethod
    def indirect_map_ranges(
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        map_entry: dace.nodes.MapEntry,
        symbol_to_memlet: Dict[str, dace.Memlet],
    ) -> None:
        curr = map_entry
        entry_nodes = []
        while state.entry_node(curr) is not None:
            entry_nodes.append(state.entry_node(curr))
            curr = state.entry_node(curr)
        entry_nodes = list(reversed(entry_nodes))

        used_symbols = map_entry.free_symbols.intersection(symbol_to_memlet.keys())
        new_symbols = {}
        for sym in used_symbols:
            # Add new symbol to avoid clashes
            new_sym = sdfg.find_new_symbol(sym)
            new_symbols[sym] = new_sym

            # Add in-connector
            map_entry.add_in_connector(new_sym)

            # Connect via memlet
            memlet = copy.deepcopy(symbol_to_memlet[sym])
            sym_node = state.add_access(memlet.data)
            state.add_memlet_path(
                sym_node,
                *entry_nodes,
                map_entry,
                src_conn=None,
                dst_conn=new_sym,
                memlet=memlet,
            )

        map_exit = state.exit_node(map_entry)
        subgraph_nodes = set(state.all_nodes_between(map_entry, map_exit))
        subgraph_nodes.add(map_entry)
        subgraph_nodes.add(map_exit)

        for edge in state.in_edges(map_entry):
            subgraph_nodes.add(edge.src)
        for edge in state.out_edges(map_exit):
            subgraph_nodes.add(edge.dst)

        subgraph = StateSubgraphView(state, subgraph_nodes)
        subgraph.replace_dict(new_symbols)

    @staticmethod
    def indirect_read(
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        tasklet: dace.nodes.Tasklet,
        edge: MultiConnectorEdge,
        symbol_to_memlet: Dict[str, dace.Memlet],
    ) -> None:
        # Create indirection tasklet
        used_symbols = edge.data.free_symbols.intersection(symbol_to_memlet.keys())
        mapping = {sym: "_" + sym for sym in used_symbols}
        inputs = {"_" + sym for sym in used_symbols}
        inputs.add("_arr")

        new_subset = [
            (b.subs(mapping), e.subs(mapping), s.subs(mapping))
            for (b, e, s) in edge.data.subset
        ]
        new_subset = dace.subsets.Range(new_subset)
        indirection_tasklet = state.add_tasklet(
            "Indirection",
            inputs=inputs,
            outputs=set(["_out"]),
            code=f"""_out = _arr[{new_subset}]""",
        )

        curr = tasklet
        entry_nodes = []
        while state.entry_node(curr) is not None:
            entry_nodes.append(state.entry_node(curr))
            curr = state.entry_node(curr)

        entry_nodes = list(reversed(entry_nodes))

        for sym in used_symbols:
            memlet = copy.deepcopy(symbol_to_memlet[sym])
            sym_node = state.add_access(memlet.data)
            state.add_memlet_path(
                sym_node,
                *entry_nodes,
                indirection_tasklet,
                memlet=memlet,
                src_conn=None,
                dst_conn=mapping[sym],
            )

        state.add_edge(
            edge.src,
            edge.src_conn,
            indirection_tasklet,
            "_arr",
            dace.Memlet.from_array(edge.data.data, sdfg.arrays[edge.data.data]),
        )

        state.remove_edge(edge)

        scalar, scalar_desc = sdfg.add_scalar(
            "_tmp",
            sdfg.arrays[edge.data.data].dtype,
            transient=True,
            find_new_name=True,
        )
        state.add_edge(
            indirection_tasklet,
            "_out",
            tasklet,
            edge.dst_conn,
            dace.Memlet.from_array(scalar, scalar_desc),
        )
