# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
import hashlib
import json
import struct

import pytest
import torch

from rl_engine.ops.rope.rope_table import (
    RoPETableIdentity,
    build_shared_rope_table,
    load_shared_rope_table,
    table_identity,
)


def artifact():
    # Independent serialized fixture: position zero is exactly cos=1, sin=0.
    c = torch.ones(1, 64, dtype=torch.bfloat16)
    s = torch.zeros_like(c)
    metadata = dict(
        algorithm="f2-torch-fp32-pow-recip-mul-cos-sin-bf16-v1",
        theta=500000.0,
        dimension=128,
        position_start=0,
        length=1,
        layout="cos-then-sin/bf16-le/row-major/Px64",
    )
    payload = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    payload += struct.pack("<64H", *([0x3F80] * 64)) + bytes(128)
    return c, s, RoPETableIdentity(1, hashlib.sha256(payload).hexdigest())


def test_load_identity_and_ownership():
    c, s, expected = artifact()
    a, b = load_shared_rope_table(c, s, expected=expected, device="cpu")
    assert table_identity(a, b) == expected
    c.zero_()
    assert torch.all(a == 1)
    assert table_identity(a, b) == expected


@pytest.mark.parametrize("bad", ["zero", "swap", "bit", "theta", "algorithm", "coverage"])
def test_reject_bad_artifact(bad):
    c, s, expected = artifact()
    if bad == "zero":
        c.zero_()
    elif bad == "swap":
        c, s = s, c
    elif bad == "bit":
        c.view(torch.int16)[0, 1] += 1
    elif bad == "theta":
        expected = replace(expected, theta=10000.0)
    elif bad == "algorithm":
        expected = replace(expected, algorithm="unknown")
    else:
        expected = replace(expected, length=2)
    with pytest.raises(ValueError):
        load_shared_rope_table(c, s, expected=expected, device="cpu")


def test_reject_unregistered_mixed_and_modified_pairs():
    c, s, expected = artifact()
    with pytest.raises(ValueError, match="unverified"):
        table_identity(c, s)
    a, b = load_shared_rope_table(c, s, expected=expected, device="cpu")
    x, y = load_shared_rope_table(c, s, expected=expected, device="cpu")
    with pytest.raises(ValueError, match="unverified"):
        table_identity(a, y)
    a.add_(1)
    with pytest.raises(ValueError, match="modified"):
        table_identity(a, b)
    assert table_identity(x, y) == expected


def test_cpu_cannot_be_canonical_producer():
    with pytest.raises(ValueError, match="NVIDIA CUDA"):
        build_shared_rope_table(1, device="cpu")


def test_load_inside_inference_mode_keeps_version_tracking():
    c, s, expected = artifact()
    with torch.inference_mode():
        a, b = load_shared_rope_table(c, s, expected=expected, device="cpu")
        assert table_identity(a, b) == expected
    assert not a.is_inference()
