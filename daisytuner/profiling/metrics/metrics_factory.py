# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
from typing import List


class MetricsFactory:
    @classmethod
    def create(cls, arch: str, groups: List[str]):
        if arch == "broadwellEP":
            from daisytuner.profiling.metrics.broadwellEP_metrics import (
                BroadwellEPMetrics,
            )

            return BroadwellEPMetrics(groups=groups)
        elif arch == "haswellEP":
            from daisytuner.profiling.metrics.haswellEP_metrics import (
                HaswellEPMetrics,
            )

            return HaswellEPMetrics(groups=groups)
        elif arch == "skylake":
            from daisytuner.profiling.metrics.skylakeX_metrics import (
                SkylakeMetrics,
            )

            return SkylakeMetrics(groups=groups)
        elif arch == "skylakeX":
            from daisytuner.profiling.metrics.skylakeX_metrics import (
                SkylakeXMetrics,
            )

            return SkylakeXMetrics(groups=groups)
        elif arch == "zen":
            from daisytuner.profiling.metrics.zen_metrics import (
                ZenMetrics,
            )

            return ZenMetrics(groups=groups)
        elif arch == "zen2":
            from daisytuner.profiling.metrics.zen2_metrics import (
                Zen2Metrics,
            )

            return Zen2Metrics(groups=groups)
        elif arch == "zen3":
            from daisytuner.profiling.metrics.zen3_metrics import (
                Zen3Metrics,
            )

            return Zen3Metrics(groups=groups)
        elif arch == "nvidia_cc_ge_7":
            from daisytuner.profiling.metrics.nvidia_cc_ge_7_metrics import (
                NVIDIACCGE7Metrics,
            )

            return NVIDIACCGE7Metrics(groups=groups)
        else:
            assert False
