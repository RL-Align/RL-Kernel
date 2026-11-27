# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4.1 Engram: n-gram hashes, table lookup and the gated update of the mHC streams.

Port of DeepSeek's reference (``inference/engram.py`` and ``model.py`` ``Engram``; vLLM's
``common/engram.py`` is the same arithmetic):

- Token ids map to a compressed vocabulary (tokens that normalize alike collapse,
  :func:`build_compressed_token_map`; its size must equal ``engram_compressed_vocab_size``).
- Each position is hashed with the ``max_ngram_size - 1`` tokens before it: the compressed
  ids times per-(layer, lookback) odd multipliers, XORed one lookback at a time; after
  lookback ``i`` the running value lands in the prime-sized bucket range of each of the
  ``n_heads`` heads. Look-back stops at the sequence start (and at dead tokens), padding
  with ``token_map[engram_pad_token_id]``.
- The hashes pick MXFP8 table rows (dequantized exactly to BF16), ``wkv`` (BF16, D12)
  projects them to four keys and a value, and the gate kernel (``dsv41-engram-v1``)
  updates the streams.

All of it is integer arithmetic, gathers and row-local kernels: batch-invariant. The
tables (196B FP8 values in the release) may live in host memory; the lookup gathers on the
table's device and copies the rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from rl_engine.kernels.ops.cuda.matmul.bf16_linear import Bf16Linear
from rl_engine.models.deepseek_v41.config import Dsv41Config

DEAD = -2  # a token that takes part in no n-gram (image spans); -1 is "before the start"


def build_compressed_token_map(tokenizer_json: str) -> tuple[list[int], int]:
    """``(lookup [vocab], compressed vocab size)`` from a ``tokenizer.json`` (DeepSeek's
    ``build_compressed_token_map`` on the raw Rust tokenizer)."""
    from tokenizers import Regex, Tokenizer, normalizers

    backend = Tokenizer.from_file(str(tokenizer_json))
    sentinel = ""
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    key_to_new: dict[str, int] = {}
    vocab = backend.get_vocab_size(with_added_tokens=True)
    lookup = [0] * vocab
    for token_id in range(vocab):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def hash_multipliers(layer_ids, max_ngram_size: int, compressed_vocab_size: int) -> Tensor:
    """int64 ``[layers, max_ngram_size]``: odd, bounded so ``id * multiplier`` fits int64."""
    bound = max(1, (np.iinfo(np.int64).max // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        gen = np.random.default_rng(10007 * layer_id)
        values = gen.integers(low=0, high=bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


class NgramHasher:
    """Hash ids ``int64 [T, engram layers, (max_ngram - 1) * n_heads]`` of each position."""

    def __init__(self, cfg: Dsv41Config, token_map: list[int] | Tensor, device="cuda") -> None:
        tm = torch.as_tensor(token_map, dtype=torch.long)
        size = int(tm.max()) + 1
        if size != cfg.engram_compressed_vocab_size:
            raise ValueError(
                f"compressed vocabulary has {size} ids; the config says "
                f"{cfg.engram_compressed_vocab_size} (every multiplier depends on it)"
            )
        primes = cfg.engram_primes()  # [layer][ngram - 2][head]
        flat = [[p for per in layer for p in per] for layer in primes]
        offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]
        self.max_ngram = cfg.engram_max_ngram_size
        self.token_map = tm.to(device)
        self.pad_id = int(tm[cfg.engram_pad_token_id])
        self.primes = torch.tensor(primes, dtype=torch.long, device=device)
        self.offsets = torch.tensor(np.array(offsets), dtype=torch.long, device=device)
        self.multipliers = hash_multipliers(cfg.engram_layer_ids, self.max_ngram, size).to(device)

    def __call__(self, window: Tensor) -> Tensor:
        """``window`` int64 ``[T, max_ngram]``: the token ids at ``pos, pos - 1, ...`` (``-1``
        before the sequence start, :data:`DEAD` for a dead token)."""
        ids = window.long()
        compressed = self.token_map[ids.clamp(min=0)]
        blocked = torch.cumsum((ids < 0).long(), dim=1) > 0
        tokens = torch.where(blocked, torch.full_like(compressed, self.pad_id), compressed)
        products = tokens[:, None, :] * self.multipliers[None]  # [T, L, S]
        rolling, hashes = products[..., 0], []
        for i in range(1, self.max_ngram):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[None, :, i - 1])
        return torch.cat(hashes, dim=-1) + self.offsets[None]


def ngram_windows(sequences: list[Tensor], positions: list[Tensor], max_ngram: int) -> Tensor:
    """int64 ``[T, max_ngram]`` for the tokens of a step: request ``r``'s tokens at
    ``positions[r]`` with their look-back into ``sequences[r]`` (its token ids so far)."""
    rows = []
    for seq, pos in zip(sequences, positions):
        seq = seq.long()
        back = pos.long()[:, None] - torch.arange(max_ngram, device=pos.device)[None]
        rows.append(torch.where(back >= 0, seq.to(pos.device)[back.clamp(min=0)], -1))
    return torch.cat(rows)


@dataclass
class EngramWeights:
    table: Tensor  # E4M3 [rows, head_dim], on the device or in host memory
    table_scale: Tensor  # E8M0 [rows, head_dim / 32]
    wkv: Bf16Linear  # [(hc + 1) H, cols * head_dim]
    q_weight: Tensor  # BF16 [hc, H]
    k_weight: Tensor  # BF16 [hc, H]
    hash_index: int  # this layer's position in engram_layer_ids


def lookup_rows(w: EngramWeights, hashes: Tensor, device) -> Tensor:
    """BF16 ``[T, cols * head_dim]``: the hashed table rows, dequantized exactly."""
    idx = hashes.to(w.table.device)
    q = w.table[idx].float()  # [T, cols, head_dim]
    s = w.table_scale[idx].float().repeat_interleave(32, dim=-1)
    return (q * s).to(torch.bfloat16).flatten(1).to(device)


def engram_forward(
    w: EngramWeights, streams: Tensor, hashes: Tensor, mask: Tensor | None, eps: float
) -> Tensor:
    """Streams BF16 ``[T, 4, H]`` after this layer's Engram (``hashes`` ``[T, L, cols]``)."""
    from rl_engine.kernels.ops.cuda.engram import engram_gate_cuda

    rows = lookup_rows(w, hashes[:, w.hash_index], streams.device)
    kv = w.wkv(rows).float()  # BF16 wkv output, as DeepSeek's Linear
    return engram_gate_cuda(streams, kv, w.q_weight, w.k_weight, mask, eps)[0]


def engram_forward_train(
    w: EngramWeights, streams: Tensor, hashes: Tensor, mask: Tensor | None, eps: float
) -> Tensor:
    """:func:`engram_forward` under autograd (same bytes): dX of the streams only (tables,
    ``wkv``, ``q`` and ``k`` are frozen)."""
    from rl_engine.kernels.ops.cuda.engram import engram_gate

    with torch.no_grad():
        rows = lookup_rows(w, hashes[:, w.hash_index], streams.device)
        kv = w.wkv(rows).float()
    return engram_gate(streams, kv, w.q_weight, w.k_weight, mask, eps)
