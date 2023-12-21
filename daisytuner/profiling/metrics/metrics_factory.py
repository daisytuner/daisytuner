# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import re
import dace
import platform
import importlib

from typing import Dict

from daisytuner.profiling.metrics.metric import Metric
from daisytuner.profiling.likwid_helpers import cpu_codename


class MetricsFactory:
    @staticmethod
    def create(
        metric: str,
        sdfg: dace.SDFG,
        hostname: str = platform.node(),
        codename: str = cpu_codename(),
        cache: Dict = None,
    ) -> Metric:
        if sum(1 for c in metric if c.islower()) > 0:
            metric_module_name = re.sub(r"(?<!^)(?=[A-Z])", "_", metric).lower()
        else:
            metric_module_name = metric.lower()

        metric_module = importlib.import_module(
            f"daisytuner.profiling.metrics.{codename}.{metric_module_name}"
        )
        metric_class = getattr(metric_module, metric)
        return metric_class(sdfg=sdfg, hostname=hostname, cache=cache)
