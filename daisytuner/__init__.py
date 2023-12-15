# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
from daisytuner import analysis
from daisytuner import device_mapping
from daisytuner import library
from daisytuner import passes
from daisytuner import pipelines
from daisytuner import profiling
from daisytuner import transformations
from daisytuner import tuning

import dace

dace.libraries.blas.default_implementation = "MKL"
