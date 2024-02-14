import islpy as isl


class Scop:
    def __init__(self):
        self.domain = None
        self.read = None
        self.write = None
        self.schedule = None

    def _add_other(self, other):
        if self.read and other.read:
            self.read = self.read.union(other.read)
        elif not self.read:
            self.read = other.read
        if self.write and other.write:
            self.write = self.write.union(other.write)
        elif not self.write:
            self.write = other.write
        if self.domain and other.domain:
            self.domain = self.domain.union(other.domain)
        elif not self.domain:
            self.domain = other.domain

    def union(self, other):
        if self.schedule and other.schedule:
            # combines two schedules in an arbitrary order
            self.schedule = self.schedule.set(other.schedule)
        elif not self.schedule:
            self.schedule = other.schedule
        self._add_other(other)

    def sequence(self, other):
        if self.schedule and other.schedule:
            # combines two schedules in the given order (self before other)
            self.schedule = self.schedule.sequence(other.schedule)
        elif not self.schedule:
            self.schedule = other.schedule
        self._add_other(other)

    def dependency_analysis(self) -> isl.UnionMap:
        if self.write:
            write = self.write.intersect_domain(self.domain)
        else:
            write = isl.UnionMap.empty(self.domain.get_space())
        if self.read:
            read = self.read.intersect_domain(self.domain)
        else:
            read = isl.UnionMap.empty(self.domain.get_space())
        init_schedule = self.schedule.get_map().intersect_domain(self.domain)
        empty = isl.UnionMap.empty(self.domain.get_space())

        # ISL Dataflow Analysis
        # For details see: "Presburger Formulas and polyhedral Compilation"

        # value-based exact dependencies without transitive dependencies:
        # a read statement depends on the last statement that performed a
        # write to the same data element

        # RAW dependencies from the last write to a read
        (_, raw, _, _, _) = read.compute_flow(write, empty, init_schedule)
        # WAW dependencies from the last write to a write
        (_, waw, _, _, _) = write.compute_flow(write, empty, init_schedule)

        # WAR dependencies from the last read to a write
        flow_info = isl.UnionAccessInfo.from_sink(write)
        flow_info = flow_info.set_may_source(read)
        flow_info = flow_info.set_kill(write)
        flow_info = flow_info.set_schedule_map(init_schedule)
        flow = flow_info.compute_flow()
        war = flow.get_may_dependence()

        # coalescing: replace pairs of disjunct sets by single disjunct sets
        # without changing its meaning
        raw = raw.coalesce()
        war = war.coalesce()
        waw = waw.coalesce()

        dependencies = waw.union(war).union(raw)
        dependencies = dependencies.coalesce()

        # simplify the relation representation by detecting implicit equalities
        dependencies = dependencies.detect_equalities()

        return dependencies
