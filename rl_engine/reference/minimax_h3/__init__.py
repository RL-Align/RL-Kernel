# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""MiniMax-H3 (RFC #420) PyTorch goldens.

Shape constants are pinned to ``MiniMaxAI/MiniMax-H3@42ed227`` (see
``rl_engine/validation/models/h3_manifest.json``). The ops accept other sizes so that
tiny synthetic shapes can be tested, but the H3 layout rules (three modality
rows per timestep, six modulation chunks) are fixed.
"""

from __future__ import annotations

from rl_engine.reference.minimax_h3.sigma_schedule import H3SigmaSchedule, NativeH3SigmaScheduleOp

__all__ = [
    "H3SigmaSchedule",
    "H3_ADALN_CHUNK_NAMES",
    "H3_ADALN_CHUNKS",
    "H3_FREQ_DIM",
    "H3_HIDDEN_SIZE",
    "H3_MAX_PERIOD",
    "H3_MODALITY_NUM",
    "H3_TIME_EMBED_DIM",
    "H3_TIME_EMBED_HIDDEN_DIM",
    "NativeH3SigmaScheduleOp",
]

H3_FREQ_DIM = 256
H3_MAX_PERIOD = 10000
H3_TIME_EMBED_HIDDEN_DIM = 5376
H3_TIME_EMBED_DIM = 2688
H3_HIDDEN_SIZE = 5376
# Modality tags of the packed sequence: 0 video, 1 text, 2 audio.
H3_MODALITY_NUM = 3
# shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
H3_ADALN_CHUNK_NAMES = (
    "shift_msa",
    "scale_msa",
    "gate_msa",
    "shift_mlp",
    "scale_mlp",
    "gate_mlp",
)
H3_ADALN_CHUNKS = len(H3_ADALN_CHUNK_NAMES)
