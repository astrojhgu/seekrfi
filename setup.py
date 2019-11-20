#!/usr/bin/env python
from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="pyseek",
    version="1.0",
    rust_extensions=[RustExtension("pyseek.native", binding=Binding.PyO3)],
    packages=["pyseek"],
    zip_safe=False,
)
