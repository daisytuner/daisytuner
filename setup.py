# Copyright 2022-2023 ETH Zurich and the Daisytuner authors.
#!/usr/bin/env python
import sys
import setuptools
import tempfile
import pybind11

from glob import glob
from setuptools import find_packages, Extension
from distutils.core import setup
from pybind11.setup_helpers import build_ext


def has_flag(compiler, flagname):
    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False

    return True


def cpp_flag(compiler):
    if has_flag(compiler, "-std=c++14"):
        return "-std=c++14"
    elif has_flag(compiler, "-std=c++11"):
        return "-std=c++11"
    raise RuntimeError("Unsupported compiler: at least C++11 support is needed")


class BuildExt(build_ext):

    c_opts = {
        "msvc": ["/EHsc"],
        "unix": [],
    }

    if sys.platform == "darwin":
        c_opts["unix"] += ["-stdlib=libc++", "-mmacosx-version-min=10.7"]

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == "unix":
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
        elif ct == "msvc":
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())

        # Add libraries
        opts.append("-lisl")

        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)


cpp_files = sorted(glob("backends/tiramisu-autoscheduler/src/*.cpp"))
ext_modules = [
    Extension(
        "daisy_tiramisu",
        [
            "backends/tiramisu-autoscheduler/python_bindings/PyDaisyTiramisu.cpp",
        ]
        + cpp_files,
        include_dirs=[
            pybind11.get_include(False),
        ]
        + [pybind11.get_include(True) for _ in cpp_files],
        language="c++",
        libraries=["isl"],
        optional=True,
    ),
]

with open("README.md", "r") as fp:
    long_description = fp.read()

setup(
    name="daisytuner",
    version="0.2.4",
    description="A cloud-connected compiler pass for performance optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="SPCL @ ETH Zurich",
    python_requires=">=3.8",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=[
        "requests>=2.11.0",
        "tqdm>=4.64.1",
        "tabulate>=0.9.0",
        "dace>=0.15.0",
        "numpy>=1.23.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.2.0",
        "plotly>=5.11.0",
        "seaborn>=0.2.12",
        "kaleido>=0.2.1",
        "opt_einsum>=3.3.0",
        "torch>=1.13.0",
        "torchvision",
        "torchaudio",
        "torchmetrics>=0.11.4",
        "pytorch-lightning>=1.9.4",
        "torch_geometric>=2.3",
        "jupyterlab>=4.0.3",
        "deap>=1.3.3",
        "pyvis>=0.3.2",
        "networkx>=3.1.0",
    ],
    extras_require={
        "dev": ["black==22.10.0", "pytest>=7.2.0", "pytest-cov>=4.1.0"],
        "polyhedral": ["islpy==2023.1.2"],
    },
    include_package_data=True,
    package_data={
        "daisytuner": [
            "data/MapNestModel_v4_cpu.ckpt",
            "data/MapNestModel_v4_gpu.ckpt",
            "data/nvidia_gpu_cc_ge_7/*.txt",
            "data/tiramisu/hier_LSTM_fusion_tree_tagLo_transfer_5bl.pkl",
            "data/tiramisu/scripts/main.py",
            "data/tiramisu/scripts/hier_lstm.py",
            "data/tiramisu/scripts/json_to_tensor.py",
        ],
    },
    classifiers=[
        "Topic :: Utilities",
    ],
    cmdclass={"build_ext": BuildExt},
    ext_modules=ext_modules,
)
