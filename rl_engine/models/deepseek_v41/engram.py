# SPDX-License-Identifier: Apache-2.0
"""V4.1 Engram assembly with explicit real-row and projection providers."""

import torch

from rl_engine.backends.rocm.engram.gated_residual import engram_gated_residual


class Engram(torch.nn.Module):
    def __init__(self, config, original_layer_id, embedding, projection, q_weight, k_weight):
        super().__init__()
        t = config.text
        if original_layer_id not in t["engram_layer_ids"]:
            raise ValueError("original layer has no Engram")
        self.layer_id = original_layer_id
        self.hc = t["hc_mult"]
        self.dim = t["hidden_size"]
        self.hash_columns = (t["engram_max_ngram_size"] - 1) * t["engram_n_heads"]
        self.eps = t["rms_norm_eps"]
        if q_weight.shape != (self.hc, self.dim) or k_weight.shape != q_weight.shape:
            raise ValueError("checkpoint q/k gains have incorrect shapes")
        self.embedding = embedding
        self.projection = projection
        self.q_weight = torch.nn.Parameter(q_weight)
        self.k_weight = torch.nn.Parameter(k_weight)

    def forward(self, x, hash_ids, token_mask=None):
        if x.shape[-2:] != (self.hc, self.dim) or hash_ids.shape != (
            *x.shape[:-2],
            self.hash_columns,
        ):
            raise ValueError("Engram residual/hash shapes do not match the original layer")
        projected = self.projection(self.embedding(hash_ids).flatten(-2))
        key, value = projected.split((self.hc * self.dim, self.dim), dim=-1)
        key = key.reshape(*x.shape[:-2], self.hc, self.dim)
        return engram_gated_residual(
            x, key, value, self.q_weight, self.k_weight, token_mask, eps=self.eps
        )


def build_compressed_token_map(backend_tokenizer):
    """Apply the released tokenizer's NFKC/NFD/accent/case/space normalization.

    Partial UTF-8 tokens retain their raw token spelling. A single space retains
    a dedicated normalized key. This affects the actual learned Engram row IDs.
    """
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalize = normalizers.Sequence(
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
    keys = {}
    ids = []
    for token_id in range(backend_tokenizer.get_vocab_size(with_added_tokens=True)):
        text = backend_tokenizer.decode([token_id], skip_special_tokens=False)
        key = (
            backend_tokenizer.id_to_token(token_id)
            if "\ufffd" in text
            else (normalize.normalize_str(text) or text)
        )
        if key not in keys:
            keys[key] = len(keys)
        ids.append(keys[key])
    return torch.tensor(ids, dtype=torch.int64), len(keys)
