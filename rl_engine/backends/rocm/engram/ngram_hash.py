# SPDX-License-Identifier: Apache-2.0
"""Integer-only V4.1 Engram hashing with explicit sequence history."""

import numpy as np
import torch
import triton
import triton.language as tl
from sympy import nextprime


@triton.jit
def _map_ids(IDS, MAP, MASK, CACHE, L: tl.constexpr, CAP: tl.constexpr, START: tl.constexpr):
    b = tl.program_id(0)
    s = tl.program_id(1) * 256 + tl.arange(0, 256)
    ids = tl.load(IDS + b * L + s, s < L, other=0)
    mapped = tl.load(MAP + ids)
    alive = tl.load(MASK + b * L + s, s < L, other=0)
    tl.store(CACHE + b * CAP + START + s, tl.where(alive, mapped, -1), s < L)


@triton.jit
def _hash(
    CACHE,
    MULT,
    PRIMES,
    OFF,
    OUT,
    L: tl.constexpr,
    CAP: tl.constexpr,
    START: tl.constexpr,
    PAD: tl.constexpr,
    LAYERS: tl.constexpr,
    NGRAM: tl.constexpr,
    HEADS: tl.constexpr,
    BH: tl.constexpr,
):
    b, s, layer = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    heads = tl.arange(0, BH)
    pos = START + s
    blocked = tl.full((), False, tl.int1)
    rolling = tl.full((), 0, tl.int64)
    for shift in tl.static_range(NGRAM):
        source = tl.load(CACHE + b * CAP + pos - shift, pos >= shift, other=-1)
        blocked = blocked | (pos < shift) | (source == -1)
        token = tl.where(blocked, PAD, source)
        multiplier = tl.load(MULT + layer * NGRAM + shift)
        rolling = rolling ^ (token * multiplier)
        if shift > 0:
            col = (shift - 1) * HEADS + heads
            prime = tl.load(PRIMES + (layer * (NGRAM - 1) * HEADS) + col, heads < HEADS, other=1)
            offset = tl.load(OFF + (layer * (NGRAM - 1) * HEADS) + col, heads < HEADS, other=0)
            value = rolling % prime + offset
            address = (((b * L + s) * LAYERS + layer) * (NGRAM - 1) * HEADS) + col
            tl.store(OUT + address, value, heads < HEADS)


class EngramHash(torch.nn.Module):
    """Prepared compressed token map is mandatory and must match the checkpoint.

    The map must come from the released tokenizer normalization. Raw token IDs
    cannot be substituted. ``reset`` and contiguous start positions prevent
    reading stale or uninitialized history across independent model requests.
    """

    def __init__(self, config, token_map, *, max_batch_size, max_seq_len):
        super().__init__()
        t = config.text
        if (
            token_map.dtype != torch.int64
            or token_map.ndim != 1
            or token_map.numel() != t["vocab_size"]
        ):
            raise ValueError("expected a complete compressed int64 tokenizer map")
        if not torch.version.hip or token_map.device.type != "cuda":
            raise RuntimeError("Engram hashing requires AMD ROCm")
        if max_batch_size <= 0 or max_seq_len <= 0:
            raise ValueError("history capacity must be positive")
        if (
            int(token_map.min()) != 0
            or int(token_map.max()) + 1 != t["engram_compressed_vocab_size"]
        ):
            raise ValueError("compressed vocabulary differs from the checkpoint")
        # Official layout draws globally unique primes across layers/ngrams/heads.
        seen = set()
        primes = []
        for _ in t["engram_layer_ids"]:
            layer = []
            for _ in range(t["engram_max_ngram_size"] - 1):
                current = t["engram_vocab_size"] - 1
                for _ in range(t["engram_n_heads"]):
                    current = int(nextprime(current))
                    while current in seen:
                        current = int(nextprime(current))
                    seen.add(current)
                    layer.append(current)
            primes.append(layer)
        prime = np.array(primes, dtype=np.int64)
        if prime.sum(1).tolist() != t["engram_num_embeddings"]:
            raise ValueError("computed hash bucket ranges differ from checkpoint table sizes")
        offsets = np.cumsum(np.pad(prime[:, :-1], ((0, 0), (1, 0))), axis=1)
        bound = (np.iinfo(np.int64).max // t["engram_compressed_vocab_size"]) // 2
        multipliers = []
        for layer_id in t["engram_layer_ids"]:
            rng = np.random.default_rng(10007 * layer_id)
            multipliers.append(
                rng.integers(0, bound, size=t["engram_max_ngram_size"], dtype=np.int64) * 2 + 1
            )
        device = token_map.device
        self.register_buffer("token_map", token_map.contiguous())
        self.register_buffer("primes", torch.tensor(prime, device=device))
        self.register_buffer("offsets", torch.tensor(offsets, device=device))
        self.register_buffer("multipliers", torch.tensor(np.array(multipliers), device=device))
        self.register_buffer(
            "cache",
            torch.full((max_batch_size, max_seq_len), -1, device=device, dtype=torch.int64),
            persistent=False,
        )
        self.pad = int(token_map[t["engram_pad_token_id"]])
        self.ngram = t["engram_max_ngram_size"]
        self.heads = t["engram_n_heads"]
        self.layers = len(t["engram_layer_ids"])
        self._length = 0
        self._batch = None

    def reset(self):
        self.cache.fill_(-1)
        self._length = 0
        self._batch = None

    @torch.no_grad()
    def forward(self, input_ids, start_pos=0, token_mask=None):
        if (
            input_ids.ndim != 2
            or input_ids.dtype != torch.int64
            or input_ids.device != self.cache.device
        ):
            raise ValueError("expected GPU int64 input_ids[B,L]")
        batch, length = input_ids.shape
        if (
            batch == 0
            or length == 0
            or batch > self.cache.shape[0]
            or start_pos + length > self.cache.shape[1]
        ):
            raise ValueError("empty input or history capacity exceeded")
        if start_pos == 0:
            self.reset()
            self._batch = batch
        if type(start_pos) is not int or start_pos != self._length or batch != self._batch:
            raise ValueError("history requires contiguous positions and a fixed batch until reset")
        if bool(((input_ids < 0) | (input_ids >= self.token_map.numel())).any()):
            raise ValueError("token ID outside the released vocabulary")
        if token_mask is None:
            token_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if (
            token_mask.dtype != torch.bool
            or token_mask.shape != input_ids.shape
            or token_mask.device != input_ids.device
        ):
            raise ValueError("invalid token mask")
        output = torch.empty(
            (batch, length, self.layers, (self.ngram - 1) * self.heads),
            device=input_ids.device,
            dtype=torch.int64,
        )
        with torch.cuda.device(input_ids.device):
            _map_ids[(batch, triton.cdiv(length, 256))](
                input_ids.contiguous(),
                self.token_map,
                token_mask.contiguous(),
                self.cache,
                length,
                self.cache.shape[1],
                start_pos,
            )
            _hash[(batch, length, self.layers)](
                self.cache,
                self.multipliers,
                self.primes,
                self.offsets,
                output,
                length,
                self.cache.shape[1],
                start_pos,
                self.pad,
                self.layers,
                self.ngram,
                self.heads,
                triton.next_power_of_2(self.heads),
            )
        self._length = start_pos + length
        return output
