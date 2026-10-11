# SPDX-License-Identifier: Apache-2.0
"""Checkpoint streams shared by the evidence runners, validator and plots."""

# Checkpoint tensors exercised per stream: block-0 projections and final modulation.
STREAMS = (
    ("image", "transformer_blocks.0.img_mod.1"),
    ("text", "transformer_blocks.0.txt_mod.1"),
    ("final", "norm_out.linear"),
)
