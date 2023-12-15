# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
import dace
import math
import time
import numpy as np

import traceback
import multiprocessing as mp

ctx = mp.get_context("spawn")

import random
import warnings
import numpy as np

from typing import Dict, List, Set, Tuple, Union, Sequence
from dace import dtypes as ddtypes
from dace.data import Data, Scalar, make_array_from_descriptor, Array
from dace.sdfg import SDFG

from dace import SDFG, DataInstrumentationType
from dace import config, nodes, symbolic
from dace.codegen.instrumentation.data.data_report import InstrumentedDataReport
from dace.libraries.standard.memory import aligned_ndarray

from typing import Dict, Tuple

from dace.codegen.compiled_sdfg import CompiledSDFG, ReloadableDLL
from dace import SDFG, config, InstrumentationType


def measure(
    sdfg: dace.SDFG,
    arguments: Dict,
    max_variance: float = 0.1,
    measurements: int = None,
    timeout: float = None,
) -> Tuple[float, float]:
    """
    A helper function to measure the median runtime of a SDFG over several measurements. The measurement is executed in a subprocess that can be killed after a specific timeout. This function will add default Timer instrumentation to the SDFG and return the full SDFG's runtime. The instrumentation report with the individual runtimes and the additional instrumentation is available afterwards as well.

    :param SDFG: the SDFG to be measured.
    :param arguments: the arguments provided to the SDFG.
    :param timeout: optional timeout to kill the measurement.
    :return: a tuple of median runtime, time of the whole measurement and the modified arguments (results). The second time is useful to determine a tight timeout for a transformed SDFG.
    """
    with config.set_temporary("instrumentation", "report_each_invocation", value=False):
        with config.set_temporary("compiler", "allow_view_arguments", value=True):
            csdfg = sdfg.compile()

            proc = MeasureProcess(
                target=_measure,
                args=(
                    sdfg.to_json(),
                    sdfg.build_folder,
                    csdfg._lib._library_filename,
                    arguments,
                    max_variance,
                    measurements,
                ),
            )

            start = time.time()
            proc.start()
            proc.join(timeout)
            process_time = time.time() - start

            # Handle failure
            if proc.exitcode != 0:
                if proc.is_alive():
                    proc.kill()

                return math.inf, process_time, None

            if proc.exception:
                if proc.is_alive():
                    proc.kill()
                error, traceback = proc.exception
                print(error)
                print(traceback)

                return math.inf, process_time, None

            # Handle success
            if proc.is_alive():
                proc.kill()

            report = sdfg.get_latest_report()
            durations = list(report.durations.values())[0]
            durations = list(durations.values())[0]
            durations = list(durations.values())[0]
            durations = np.array(durations)

            # Median with 95% CI
            durations = np.sort(durations)
            median = np.median(durations)

            n = len(durations)
            lower_ci = int(math.floor((n - 1.96 * math.sqrt(n)) / 2))
            lower_ci = max(0, min(n - 1, lower_ci))

            upper_ci = 1 + int(math.ceil((n + 1.96 * math.sqrt(n)) / 2))
            upper_ci = max(0, min(n - 1, upper_ci))

            return (
                median,
                process_time,
                (durations[lower_ci], median, durations[upper_ci]),
            )


def _measure(
    sdfg_json: Dict,
    build_folder: str,
    filename: str,
    arguments: Dict,
    max_variance: float,
    measurements: int,
):
    sdfg = SDFG.from_json(sdfg_json)
    sdfg.build_folder = build_folder
    lib = ReloadableDLL(filename, sdfg.name)
    csdfg = CompiledSDFG(sdfg, lib, arguments.keys())

    with config.set_temporary("instrumentation", "report_each_invocation", value=False):
        with config.set_temporary("compiler", "allow_view_arguments", value=True):
            rel_var = 1e4
            runs = []
            while (measurements is None and rel_var >= max_variance) or (
                measurements is not None and len(runs) < measurements
            ):
                s = time.time()

                csdfg(**arguments)

                t = time.time() - s
                runs.append(t)

                if len(runs) >= 3:
                    arr = np.array(runs)
                    rel_var = np.var(arr) / np.mean(arr)

            csdfg.finalize()


class MeasureProcess(ctx.Process):
    def __init__(self, *args, **kwargs):
        ctx.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = ctx.Pipe()
        self._exception = None

    def run(self):
        try:
            ctx.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


def random_arguments(sdfg: SDFG) -> Dict:
    """
    Creates random inputs and empty output containers for the SDFG.

    :param SDFG: the SDFG.
    :return: a dict containing the arguments.
    """
    # Symbols
    symbols = {}
    for k, v in sdfg.constants.items():
        symbols[k] = int(v)

    if len(sdfg.free_symbols) > 0:
        warnings.warn(
            "Creating random arguments for symbolic SDFG. Symbol values will be sampled from pre-defined range."
        )

    for k in sdfg.free_symbols:
        symbols[k] = random.randint(1, 4)

    arguments = {**symbols}
    for state in sdfg.nodes():
        for dnode in state.data_nodes():
            if dnode.data in arguments:
                continue

            array = sdfg.arrays[dnode.data]
            if not array.transient:
                np_array = _random_container(array, symbols_map=symbols)
                arguments[dnode.data] = (
                    np_array
                    if not isinstance(np_array, np.ndarray)
                    else np.copy(np_array)
                )

    return arguments


def create_data_report(sdfg: SDFG, arguments: Dict) -> InstrumentedDataReport:
    """
    Creates a data instrumentation report for the given SDFG and arguments.

    :param SDFG: the SDFG.
    :param arguments: the arguments to use.
    :param transients: whether to instrument transient array.
    :return: the data report.
    """
    for state in sdfg.nodes():
        for node in state.nodes():
            if isinstance(node, nodes.AccessNode):
                if state.entry_node(node) is not None:
                    continue

                node.instrument = DataInstrumentationType.Save

    with config.set_temporary("compiler", "allow_view_arguments", value=True):
        csdfg = sdfg.compile()
        _ = csdfg(**arguments)

    # Disable data instrumentation again
    for state in sdfg.nodes():
        for node in state.nodes():
            if isinstance(node, nodes.AccessNode):
                if state.entry_node(node) is not None:
                    continue

                node.instrument = DataInstrumentationType.No_Instrumentation

    dreport = sdfg.get_instrumented_data()
    return dreport


def arguments_from_data_report(sdfg: SDFG, data_report: InstrumentedDataReport) -> Dict:
    """
    Creates the arguments for the SDFG from the data report.

    :param SDFG: the SDFG.
    :param data_report: the data report.
    :return: a dict containing the arguments.
    """
    arguments = {}
    for state in sdfg.nodes():
        for dnode in state.data_nodes():
            if dnode.data in arguments:
                continue

            array = sdfg.arrays[dnode.data]
            if state.in_degree(dnode) == 0 or state.out_degree(dnode) == 0:
                data = data_report[dnode.data]
                if isinstance(data, Sequence):
                    data = data.__iter__().__next__()

                if isinstance(array, Array):
                    arguments[dnode.data] = make_array_from_descriptor(
                        array, data, symbols=sdfg.constants
                    )
                else:
                    scalar = data.astype(array.dtype.as_numpy_dtype()).item()
                    arguments[dnode.data] = scalar

    return arguments


def _random_container(array: Data, symbols_map: Dict[str, int]) -> np.ndarray:
    shape = symbolic.evaluate(array.shape, symbols=symbols_map)
    newdata = _uniform_sampling(array, shape)
    if isinstance(array, Scalar):
        return newdata
    else:
        return _align_container(array, symbols_map, newdata)


def _empty_container(
    array: Data, symbols_map: Dict[str, int]
) -> Union[int, float, np.ndarray]:
    if isinstance(array, Scalar):
        npdt = array.dtype.as_numpy_dtype()
        if npdt in [np.float16, np.float32, np.float64]:
            return 0.0
        else:
            return 0
    else:
        shape = symbolic.evaluate(array.shape, symbols=symbols_map)
        empty_container = np.zeros(shape).astype(array.dtype.as_numpy_dtype())
        return _align_container(array, symbols_map, empty_container)


def _align_container(
    array: Data, symbols_map: Dict[str, int], container: np.ndarray
) -> np.ndarray:
    view: np.ndarray = make_array_from_descriptor(array, container, symbols_map)
    if isinstance(array, Array) and array.alignment:
        return aligned_ndarray(view, array.alignment)
    else:
        return view


def _uniform_sampling(array: Data, shape: Union[List, Tuple]):
    npdt = array.dtype.as_numpy_dtype()
    if npdt in [np.float16, np.float32, np.float64]:
        low = 0.0
        high = 1.0
        if isinstance(array, Scalar):
            return np.random.uniform(low=low, high=high)
        else:
            return np.random.uniform(low=low, high=high, size=shape).astype(npdt)
    elif npdt in [
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    ]:
        low = max(np.iinfo(npdt).min, np.iinfo(np.int16).min)
        high = min(np.iinfo(npdt).max, np.iinfo(np.int16).max)
        if isinstance(array, Scalar):
            return np.random.randint(low, high)
        else:
            return np.random.randint(low, high, size=shape).astype(npdt)
    elif array.dtype in [ddtypes.bool, ddtypes.bool_]:
        if isinstance(array, Scalar):
            return np.random.randint(low=0, high=2)
        else:
            return np.random.randint(low=0, high=2, size=shape).astype(npdt)
    else:
        raise TypeError()
