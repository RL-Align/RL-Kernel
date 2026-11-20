# SPDX-License-Identifier: Apache-2.0
"""DSV4 model boundaries.

This package is not a complete validated DSV4 model. It contains router gate,
LoRA, recent128 cache, and indexer projection components, and the layer end
after the last decoder layer (``layer_end``: mHC head, final norm, lm_head and
logprob, on ROCm). Supported configurations and GPU evidence are documented
separately.
"""
