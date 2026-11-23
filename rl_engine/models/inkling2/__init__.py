# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini model-specific numerical contracts."""

from .shared_weight_grad import accumulate_shared_weight_gradients

__all__ = ["accumulate_shared_weight_gradients"]
