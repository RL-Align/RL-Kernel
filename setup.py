# SPDX-License-Identifier: Apache-2.0
"""Setuptools entry point; metadata lives in pyproject.toml."""

import sys
from pathlib import Path

from setuptools import setup

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_tools.extensions import get_cmdclass, get_extensions  # noqa: E402

setup(
    ext_modules=get_extensions(),
    cmdclass=get_cmdclass(),
    include_package_data=True,
    zip_safe=False,
)
