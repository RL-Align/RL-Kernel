# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Index selection modes shared by the reference and backend implementations."""

from importlib.metadata import PackageNotFoundError, version

try:
    _RL_KERNEL_RELEASE = f"RL-Kernel {version('rl-kernel')}"
except PackageNotFoundError:
    _RL_KERNEL_RELEASE = "this RL-Kernel source checkout"

_UNSUPPORTED_INDEX_MODE = (
    f"index_select requires unique_indices=True (distinct indices) in {_RL_KERNEL_RELEASE}. "
    "Support for unique_indices=False is planned for a future release."
)


def require_unique_indices(unique_indices: bool) -> None:
    """Require the caller's explicit guarantee that the indices are distinct."""
    # TODO: support unique_indices=False in a future release.
    if unique_indices is not True:
        raise NotImplementedError(_UNSUPPORTED_INDEX_MODE)
