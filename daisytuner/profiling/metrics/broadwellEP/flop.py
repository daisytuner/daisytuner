# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import numpy as np
import dace
import platform


from daisytuner.profiling.metrics.metric import Metric


class FLOP(Metric):
    def __init__(
        self, sdfg: dace.SDFG, hostname: str = platform.node(), cache=None
    ) -> None:
        super().__init__(
            sdfg,
            [
                "FP_ARITH_INST_RETIRED_SCALAR_DOUBLE",
                "FP_ARITH_INST_RETIRED_128B_PACKED_DOUBLE",
                "FP_ARITH_INST_RETIRED_256B_PACKED_DOUBLE",
                "FP_ARITH_INST_RETIRED_SCALAR_SINGLE",
                "FP_ARITH_INST_RETIRED_128B_PACKED_SINGLE",
                "FP_ARITH_INST_RETIRED_256B_PACKED_SINGLE",
            ],
            dace.DeviceType.CPU,
            hostname,
            "broadwellEP",
            cache=cache,
        )

    def compute(self) -> float:
        counters = self.values()

        volume = 0.0
        for state in self._sdfg.states():
            volume += sum(
                [
                    measurements[0]
                    for thread_id, measurements in counters[
                        "FP_ARITH_INST_RETIRED_SCALAR_DOUBLE"
                    ][state].items()
                ]
            )
            volume += (
                sum(
                    [
                        measurements[0]
                        for thread_id, measurements in counters[
                            "FP_ARITH_INST_RETIRED_128B_PACKED_DOUBLE"
                        ][state].items()
                    ]
                )
                * 2.0
            )
            volume += (
                sum(
                    [
                        measurements[0]
                        for thread_id, measurements in counters[
                            "FP_ARITH_INST_RETIRED_256B_PACKED_DOUBLE"
                        ][state].items()
                    ]
                )
                * 4.0
            )
            volume += sum(
                [
                    measurements[0]
                    for thread_id, measurements in counters[
                        "FP_ARITH_INST_RETIRED_SCALAR_SINGLE"
                    ][state].items()
                ]
            )
            volume += (
                sum(
                    [
                        measurements[0]
                        for thread_id, measurements in counters[
                            "FP_ARITH_INST_RETIRED_128B_PACKED_SINGLE"
                        ][state].items()
                    ]
                )
                * 4.0
            )
            volume += (
                sum(
                    [
                        measurements[0]
                        for thread_id, measurements in counters[
                            "FP_ARITH_INST_RETIRED_256B_PACKED_SINGLE"
                        ][state].items()
                    ]
                )
                * 8.0
            )

        metric = 1e-6 * volume
        return metric

    def compute_per_thread(self) -> np.ndarray:
        counters = self.values()

        volume = []
        for state in self._sdfg.states():
            volume.append(
                np.array(
                    [
                        measurements[0]
                        for thread_id, measurements in counters[
                            "FP_ARITH_INST_RETIRED_SCALAR_DOUBLE"
                        ][state].items()
                    ]
                )
            )
            volume.append(
                (
                    np.array(
                        [
                            measurements[0]
                            for thread_id, measurements in counters[
                                "FP_ARITH_INST_RETIRED_128B_PACKED_DOUBLE"
                            ][state].items()
                        ]
                    )
                    * 2.0
                )
            )
            volume.append(
                (
                    np.array(
                        [
                            measurements[0]
                            for thread_id, measurements in counters[
                                "FP_ARITH_INST_RETIRED_256B_PACKED_DOUBLE"
                            ][state].items()
                        ]
                    )
                    * 4.0
                )
            )
            volume.append(
                np.array(
                    [
                        measurements[0]
                        for thread_id, measurements in counters[
                            "FP_ARITH_INST_RETIRED_SCALAR_SINGLE"
                        ][state].items()
                    ]
                )
            )
            volume.append(
                (
                    np.array(
                        [
                            measurements[0]
                            for thread_id, measurements in counters[
                                "FP_ARITH_INST_RETIRED_128B_PACKED_SINGLE"
                            ][state].items()
                        ]
                    )
                    * 4.0
                )
            )
            volume.append(
                (
                    np.array(
                        [
                            measurements[0]
                            for thread_id, measurements in counters[
                                "FP_ARITH_INST_RETIRED_256B_PACKED_SINGLE"
                            ][state].items()
                        ]
                    )
                    * 8.0
                )
            )

        metric = 1e-6 * np.vstack(volume).sum(axis=0, keepdims=False)
        return metric
