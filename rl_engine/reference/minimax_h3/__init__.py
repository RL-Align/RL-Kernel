# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""MiniMax-H3 (RFC #420) PyTorch goldens.

Shape constants are pinned to ``MiniMaxAI/MiniMax-H3@42ed227`` (see
``rl_engine/validation/models/h3_manifest.json``). The ops accept other sizes so that
tiny synthetic shapes can be tested, but the H3 layout rules (three modality
rows per timestep, six modulation chunks) are fixed.
"""

from __future__ import annotations

H3_FREQ_DIM = 256
H3_MAX_PERIOD = 10000
H3_TIME_EMBED_HIDDEN_DIM = 5376
H3_TIME_EMBED_DIM = 2688
H3_HIDDEN_SIZE = 5376
# Modality tags of the packed sequence: 0 video, 1 text, 2 audio.
H3_MODALITY_NUM = 3
# shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
H3_ADALN_CHUNKS = 6
H3_ADALN_CHUNK_NAMES = (
    "shift_msa",
    "scale_msa",
    "gate_msa",
    "shift_mlp",
    "scale_mlp",
    "gate_mlp",
)
