# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Portable evidence API; runtime-specific adapters are loaded only for live replay."""

from .core import fingerprint
from .evidence import Capture
from .auto import AutoCapture, ModuleBoundary, auto_capture, prepare_auto
from .events import EventCapture, LogicalTensor
from .discovery import DiscoverCapture
from .shards import ShardSet, TensorShard, parallel_tensor
from .megatron import load_megatron_model, megatron_weight_shards
from .decode import DecodeCapture, paged_kv
from .routing import RoutedTensor, moe_dispatch
from .graph import GraphCapture
from .probes import FusedCapture
from .training import TrainingCapture
