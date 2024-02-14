# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import copy

from dace.sdfg import SDFG
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.state import SDFGState
from dace.transformation import transformation
from dace.properties import make_properties
from dace.sdfg.propagation import propagate_memlets_state


@make_properties
class InlineMap(transformation.SingleStateTransformation):
    map_entry = transformation.PatternNode(nodes.MapEntry)
    nested_sdfg = transformation.PatternNode(nodes.NestedSDFG)
    map_exit = transformation.PatternNode(nodes.MapExit)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry, cls.nested_sdfg, cls.map_exit)]

    def annotates_memlets(self) -> bool:
        return True

    def can_be_applied(
        self, state: dace.SDFGState, expr_index: int, sdfg: dace.SDFG, permissive=False
    ):
        if state.entry_node(self.nested_sdfg) != self.map_entry:
            return False
        if len(self.nested_sdfg.sdfg.states()) < 2:
            return False

        # Border arrays: Local arrays which must be extended by map.
        border_arrays = set()
        for array, desc in self.nested_sdfg.sdfg.arrays.items():
            if not desc.transient:
                continue

            occurence = 0
            for nstate in self.nested_sdfg.sdfg.states():
                for dnode in nstate.data_nodes():
                    if dnode.data == array:
                        occurence += 1
                        break

            if occurence > 1 or self.nested_sdfg.sdfg.has_cycles():
                border_arrays.add(array)

        # Control-flow is map-invariant
        for nedge in self.nested_sdfg.sdfg.edges():
            for param in self.map_entry.map.params:
                for sym in nedge.data.free_symbols:
                    if param == str(sym):
                        return False

        # Control-flow is invariant to border arrays
        for nedge in self.nested_sdfg.sdfg.edges():
            for array in border_arrays:
                if array in nedge.data.free_symbols:
                    return False

        # Border arrays are map-invariant
        for array in border_arrays:
            for param in self.map_entry.map.params:
                for sym in self.nested_sdfg.sdfg.arrays[array].free_symbols:
                    if str(sym) == param:
                        return False

        # Inputs / outputs are map-invariant
        for edge in state.in_edges(self.nested_sdfg):
            if edge.data is None:
                continue

            for (b, _, s) in edge.data.subset.ranges:
                for sym in b.free_symbols:
                    for param in self.map_entry.map.params:
                        if str(sym) == param:
                            return False

                if type(s) != int:
                    for sym in s.free_symbols:
                        for param in self.map_entry.map.params:
                            if str(sym) == param:
                                return False

        for edge in state.out_edges(self.nested_sdfg):
            if edge.data is None:
                continue

            for (b, _, s) in edge.data.subset.ranges:
                for sym in b.free_symbols:
                    for param in self.map_entry.map.params:
                        if str(sym) == param:
                            return False

                if type(s) != int:
                    for sym in s.free_symbols:
                        for param in self.map_entry.map.params:
                            if str(sym) == param:
                                return False

        return True

    def apply(self, state: SDFGState, sdfg: SDFG):
        nsdfg_node = self.nested_sdfg
        nsdfg = nsdfg_node.sdfg
        map_entry = self.map_entry
        map_exit = self.map_exit
        map_sizes = [(e + 1 - b) // s for (b, e, s) in map_entry.map.range]

        # Prepare border arrays

        # Collect
        border_arrays = set()
        for array, desc in nsdfg.arrays.items():
            if not desc.transient:
                continue

            occurence = 0
            for nstate in nsdfg.states():
                for dnode in nstate.data_nodes():
                    if dnode.data == array:
                        occurence += 1
                        break

            if occurence > 1 or nsdfg.has_cycles():
                border_arrays.add(array)

        # Enlarge by map size
        for array in border_arrays:
            desc = nsdfg.arrays[array]
            # Convert scalar to array first
            if isinstance(desc, dace.data.Scalar):
                desc = dace.data.Array(
                    desc.dtype,
                    desc.shape,
                    desc.transient,
                    desc.allow_conflicts,
                    desc.storage,
                    desc.location,
                    desc.strides,
                    desc.offset,
                    False,
                    desc.lifetime,
                    0,
                    desc.debuginfo,
                    desc.total_size,
                    desc.start_offset,
                )
            for sz in reversed(map_sizes):
                desc.strides = [desc.total_size] + list(desc.strides)
                desc.total_size = desc.total_size * sz

            desc.shape = map_sizes + list(desc.shape)
            desc.offset = [0] * len(map_sizes) + list(desc.offset)
            nsdfg.arrays[array] = desc

        # Additional indices
        additional_memlet_dims = []
        for i in range(len(map_entry.map.params)):
            dim = dace.symbolic.pystr_to_symbolic(map_entry.map.params[i])
            b, e, s = map_entry.map.range[i]

            start = (dim - b) // s
            additional_memlet_dims.append((start, start, 1))

        for nstate in nsdfg.states():
            for edge in nstate.edges():
                if edge.data.data is None or not edge.data.data in border_arrays:
                    continue

                new_ranges = (
                    copy.deepcopy(additional_memlet_dims) + edge.data.subset.ranges
                )
                edge.data.subset = dace.subsets.Range(new_ranges)

        # Distributed map
        # Create new nested SDFG
        distributed_sdfg = dace.SDFG(nsdfg.label)

        # Add symbols
        free_syms = nsdfg.free_symbols.difference(map_entry.map.params)
        for s in free_syms:
            stype = sdfg.symbols[s] if s in sdfg.symbols else dace.int64
            distributed_sdfg.add_symbol(s, stype)

        # Add data desc
        for array, desc in nsdfg.arrays.items():
            desc = copy.deepcopy(desc)
            distributed_sdfg.add_datadesc(array, desc)

        # Add node and connect
        distributed_sdfg_node = state.add_nested_sdfg(
            distributed_sdfg,
            parent=sdfg,
            inputs=set(nsdfg_node.in_connectors.keys()),
            outputs=set(nsdfg_node.out_connectors.keys()),
        )
        for inedge in state.in_edges(nsdfg_node):
            matching_edge = next(
                state.in_edges_by_connector(
                    map_entry, connector="IN_" + inedge.src_conn[4:]
                ).__iter__()
            )
            state.add_edge(
                matching_edge.src,
                matching_edge.src_conn,
                distributed_sdfg_node,
                inedge.dst_conn,
                copy.deepcopy(matching_edge.data),
            )

        for oedge in state.out_edges(nsdfg_node):
            matching_edge = next(
                state.out_edges_by_connector(
                    map_exit, connector="OUT_" + oedge.dst_conn[3:]
                ).__iter__()
            )
            state.add_edge(
                distributed_sdfg_node,
                oedge.src_conn,
                matching_edge.dst,
                matching_edge.dst_conn,
                copy.deepcopy(matching_edge.data),
            )

        # Add "mapped" states
        state_mapping = {}
        for nstate in nsdfg.states():
            # New state
            distributed_state = distributed_sdfg.add_state(nstate.label)
            state_mapping[nstate] = distributed_state

            if not nstate.nodes():
                continue

            # Create nsdfg with only current state
            temp = dace.SDFG("sdfg_" + nstate.label)
            for s, v in nsdfg.symbols.items():
                temp.add_symbol(s, v)

            temp_state = temp.add_state(nstate.label, is_start_state=True)
            node_mapping = {}
            for node in nstate.nodes():
                new_node = copy.deepcopy(node)
                temp_state.add_node(new_node)
                node_mapping[node] = new_node
            for edge in nstate.edges():
                temp_state.add_edge(
                    node_mapping[edge.src],
                    edge.src_conn,
                    node_mapping[edge.dst],
                    edge.dst_conn,
                    copy.deepcopy(edge.data),
                )

            # Add outer map
            state_map_entry = copy.deepcopy(map_entry)
            state_map_entry.in_connectors = {}
            state_map_entry.out_connectors = {}
            distributed_state.add_node(state_map_entry)

            state_map_exit = copy.deepcopy(map_exit)
            state_map_exit.in_connectors = {}
            state_map_exit.out_connectors = {}
            distributed_state.add_node(state_map_exit)

            state_inputs = set()
            state_outputs = set()
            for dnode in nstate.data_nodes():
                if not (
                    dnode.data in nsdfg_node.in_connectors
                    or dnode.data in nsdfg_node.out_connectors
                    or dnode.data in border_arrays
                ):
                    continue

                if nstate.in_degree(dnode) == 0:
                    if "IN_" + dnode.data in state_map_entry.in_connectors:
                        continue

                    access_node = distributed_state.add_access(dnode.data)
                    state_map_entry.add_in_connector("IN_" + dnode.data)
                    distributed_state.add_edge(
                        access_node,
                        None,
                        state_map_entry,
                        "IN_" + dnode.data,
                        dace.Memlet.from_array(
                            dnode.data, distributed_sdfg.arrays[dnode.data]
                        ),
                    )
                    state_inputs.add(dnode.data)
                elif nstate.out_degree(dnode) == 0:
                    if "OUT_" + dnode.data in state_map_exit.out_connectors:
                        continue

                    access_node = distributed_state.add_access(dnode.data)
                    state_map_exit.add_out_connector("OUT_" + dnode.data)
                    distributed_state.add_edge(
                        state_map_exit,
                        "OUT_" + dnode.data,
                        access_node,
                        None,
                        dace.Memlet.from_array(
                            dnode.data, distributed_sdfg.arrays[dnode.data]
                        ),
                    )
                    state_outputs.add(dnode.data)
                elif not nsdfg.arrays[dnode.data].transient:
                    if "IN_" + dnode.data not in state_map_entry.in_connectors:
                        continue

                    access_node = distributed_state.add_access(dnode.data)
                    state_map_entry.add_in_connector("IN_" + dnode.data)
                    distributed_state.add_edge(
                        access_node,
                        None,
                        state_map_entry,
                        "IN_" + dnode.data,
                        dace.Memlet.from_array(
                            dnode.data, distributed_sdfg.arrays[dnode.data]
                        ),
                    )
                    state_inputs.add(dnode.data)

                    if "OUT_" + dnode.data not in state_map_exit.out_connectors:
                        continue

                    access_node = distributed_state.add_access(dnode.data)
                    state_map_exit.add_out_connector("OUT_" + dnode.data)
                    distributed_state.add_edge(
                        state_map_exit,
                        "OUT_" + dnode.data,
                        access_node,
                        None,
                        dace.Memlet.from_array(
                            dnode.data, distributed_sdfg.arrays[dnode.data]
                        ),
                    )
                    state_outputs.add(dnode.data)

            for array, desc in nsdfg.arrays.items():
                if array not in set([dnode.data for dnode in nstate.data_nodes()]):
                    continue

                desc = copy.deepcopy(desc)
                if array in state_inputs or array in state_outputs:
                    desc.transient = False
                temp.add_datadesc(array, desc)

            temp_node = distributed_state.add_nested_sdfg(
                temp,
                parent=distributed_sdfg,
                inputs=state_inputs,
                outputs=state_outputs,
            )

            if not state_inputs:
                distributed_state.add_edge(
                    state_map_entry,
                    None,
                    temp_node,
                    None,
                    dace.Memlet(),
                )
            else:
                for data in state_inputs:
                    if "OUT_" + data not in state_map_entry.out_connectors:
                        state_map_entry.add_out_connector("OUT_" + data)

                    distributed_state.add_edge(
                        state_map_entry,
                        "OUT_" + data,
                        temp_node,
                        data,
                        dace.Memlet.from_array(data, distributed_sdfg.arrays[data]),
                    )

            for data in state_outputs:
                if "IN_" + data not in state_map_exit.in_connectors:
                    state_map_exit.add_in_connector("IN_" + data)

                distributed_state.add_edge(
                    temp_node,
                    data,
                    state_map_exit,
                    "IN_" + data,
                    dace.Memlet.from_array(data, distributed_sdfg.arrays[data]),
                )

            propagate_memlets_state(distributed_sdfg, distributed_state)

        for edge in nsdfg.edges():
            distributed_sdfg.add_edge(
                state_mapping[edge.src],
                state_mapping[edge.dst],
                copy.deepcopy(edge.data),
            )

        state.remove_node(nsdfg_node)
        state.remove_node(map_exit)
        state.remove_node(map_entry)

        sdfg.reset_sdfg_list()
        propagate_memlets_state(sdfg, state)
