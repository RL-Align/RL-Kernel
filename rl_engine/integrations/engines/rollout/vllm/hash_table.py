# SPDX-License-Identifier: Apache-2.0
"""Keep original I64 hash Parameters during native checkpoint loading."""

from contextlib import contextmanager

import torch


def preserve_original_hash_dtype(moe):
    table = moe.gate.tid2eid
    if table is None:
        return
    if (
        not isinstance(table, torch.nn.Parameter)
        or table.requires_grad
        or table.dtype != torch.int32
    ):
        raise TypeError("expected the pinned frozen native I32 hash Parameter")
    # Keep the object referenced by the native router. The native custom op
    # dispatches both input_ids and tid2eid as I64 (topk_softplus_sqrt.cu).
    # This runs BEFORE the original checkpoint loader, never from audit payloads.
    table.data = table.data.to(torch.int64)


@contextmanager
def original_hash_table_loading():
    from vllm.models.deepseek_v4.amd.model import DeepseekV4MoE

    original = DeepseekV4MoE.__init__

    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        preserve_original_hash_dtype(self)

    DeepseekV4MoE.__init__ = initialize
    try:
        yield
    finally:
        DeepseekV4MoE.__init__ = original
