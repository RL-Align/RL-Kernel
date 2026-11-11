# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Write a small random DeepSeek-V4 checkpoint in the real DeepSeek-V4-Flash format.

For end-to-end runs (verl: vLLM rollout + Megatron actor) on a 2-GPU allocation. Layer
``i`` of the toy model copies the tensor list, dtypes and shapes of layer ``i`` of
DeepSeek-V4-Flash, so the first layers keep their real types:

- layers 0-1 are sliding-window only;
- layer 2 is C4 with the indexer;
- layer 3 is C128;
- layers 0-2 use hash routing and layer 3 scored routing.

Hidden size, heads and the expert width are unchanged. Only the routed-expert count
shrinks and MTP is dropped.

The tensor list comes from the real checkpoint's safetensors headers. ``--fetch`` reads
them from the Hugging Face Hub by HTTP range requests (headers only, no weights) along
with config.json and the tokenizer files.

Values are random:

- norms are 1;
- other BF16 / FP32 weights are N(0, 0.02);
- FP8 linears are N(0, 1/sqrt(K)) quantized per 128x128 block with UE8M0 scales;
- routed experts are N(0, 1/sqrt(K)) quantized to MXFP4 by ``mx_quantize``: packed
  int8, low nibble = even column, E8M0 per 32;
- ``tid2eid`` draws 6 distinct experts per token id.

    python scripts/make_dsv4_toy_checkpoint.py --fetch --meta ~/dsv4-meta
    python scripts/make_dsv4_toy_checkpoint.py --meta ~/dsv4-meta --out ~/dsv4-toy \\
        [--layers 4 --experts 16 --seed 0]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import struct
import urllib.request

import torch
from safetensors.torch import save_file

REPO = "deepseek-ai/DeepSeek-V4-Flash"
META_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json")
DTYPES = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.float8_e8m0fnu,
    "I8": torch.int8,
    "I64": torch.int64,
}


def _url(name: str) -> str:
    return f"https://huggingface.co/{REPO}/resolve/main/{name}"


def _read_header(shard: str) -> dict:
    req = urllib.request.Request(_url(shard), headers={"Range": "bytes=0-7"})
    n = struct.unpack("<Q", urllib.request.urlopen(req).read())[0]
    req = urllib.request.Request(_url(shard), headers={"Range": f"bytes=8-{7 + n}"})
    header = json.loads(urllib.request.urlopen(req).read())
    header.pop("__metadata__", None)
    return header


def fetch(meta: pathlib.Path, layers: int) -> None:
    meta.mkdir(parents=True, exist_ok=True)
    for name in META_FILES + ("model.safetensors.index.json",):
        urllib.request.urlretrieve(_url(name), meta / name)
    weight_map = json.loads((meta / "model.safetensors.index.json").read_text())["weight_map"]
    keep = re.compile(rf"^(?!mtp\.)(?!layers\.(?!(?:{'|'.join(map(str, range(layers)))})\.))")
    shards = sorted({v for k, v in weight_map.items() if keep.match(k)})
    headers: dict = {}
    for shard in shards:
        headers.update({k: v for k, v in _read_header(shard).items() if keep.match(k)})
    (meta / "headers.json").write_text(json.dumps(headers))
    print(f"fetched {len(headers)} tensor headers from {len(shards)} shards into {meta}")


def _fp8_block(w: torch.Tensor, scale_shape) -> tuple[torch.Tensor, torch.Tensor]:
    n, k = w.shape
    rows, cols = scale_shape
    pad = torch.zeros(rows * 128, cols * 128, dtype=torch.float32)
    pad[:n, :k] = w.float()
    blocks = pad.view(rows, 128, cols, 128)
    amax = blocks.abs().amax(dim=(1, 3)).clamp(min=2.0**-126)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    q = (blocks / scale[:, None, :, None]).view(rows * 128, cols * 128)[:n, :k]
    return q.to(torch.float8_e4m3fn), scale.to(torch.float8_e8m0fnu)


def _mxfp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from rl_engine.moe.mx_format import mx_quantize

    q = mx_quantize(w.float(), "e2m1")
    return q.codes.view(torch.int8), q.scales.view(torch.float8_e8m0fnu)


def make(meta: pathlib.Path, out: pathlib.Path, layers: int, experts: int, seed: int) -> None:
    headers = json.loads((meta / "headers.json").read_text())
    g = torch.Generator().manual_seed(seed)
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((meta / "config.json").read_text())
    cfg["num_hidden_layers"] = layers
    cfg["n_routed_experts"] = experts
    cfg["num_nextn_predict_layers"] = 0
    cfg["compress_ratios"] = cfg["compress_ratios"][:layers]
    cfg["num_hash_layers"] = min(cfg.get("num_hash_layers", 0), layers)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    for name in META_FILES[1:]:
        shutil.copy(meta / name, out / name)

    def tensor(name: str, info: dict) -> torch.Tensor | None:
        shape = list(info["shape"])
        m = re.search(r"\.experts\.(\d+)\.", name)
        if m and int(m.group(1)) >= experts:
            return None
        if name.endswith(("ffn.gate.weight", "ffn.gate.bias")):
            shape[0] = experts
        dtype = DTYPES[info["dtype"]]
        if name.endswith("tid2eid"):
            return torch.rand(shape[0], experts, generator=g).argsort(dim=1)[:, : shape[1]].long()
        if name.endswith("norm.weight"):
            return torch.ones(shape, dtype=dtype)
        if name.endswith(("attn_sink", "gate.bias")):
            return torch.zeros(shape, dtype=dtype)
        return (torch.randn(shape, generator=g) * 0.02).to(dtype)

    weight_map: dict[str, str] = {}
    shard_tensors: dict[str, dict[str, torch.Tensor]] = {}
    for name, info in sorted(headers.items()):
        m = re.match(r"layers\.(\d+)\.", name)
        if m and int(m.group(1)) >= layers:
            continue
        shard = f"model-layer{m.group(1)}.safetensors" if m else "model-top.safetensors"
        if info["dtype"] in ("F8_E8M0",):
            continue  # written with its weight
        if info["dtype"] in ("F8_E4M3", "I8"):
            m_exp = re.search(r"\.experts\.(\d+)\.", name)
            if m_exp and int(m_exp.group(1)) >= experts:
                continue
            base = name[: -len(".weight")]
            scale_info = headers[base + ".scale"]
            n, k_stored = info["shape"]
            k = k_stored * 2 if info["dtype"] == "I8" else k_stored
            w = torch.randn(n, k, generator=g) / k**0.5
            if info["dtype"] == "I8":
                q, s = _mxfp4(w)
            else:
                q, s = _fp8_block(w, scale_info["shape"])
            assert list(q.shape) == info["shape"] and list(s.shape) == scale_info["shape"], name
            shard_tensors.setdefault(shard, {})[name] = q
            shard_tensors[shard][base + ".scale"] = s
            weight_map[name] = weight_map[base + ".scale"] = shard
            continue
        t = tensor(name, info)
        if t is None:
            continue
        shard_tensors.setdefault(shard, {})[name] = t.contiguous()
        weight_map[name] = shard

    total = 0
    for shard, tensors in shard_tensors.items():
        save_file(tensors, str(out / shard), metadata={"format": "pt"})
        total += sum(t.numel() * t.element_size() for t in tensors.values())
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=1)
    )
    print(
        f"wrote {len(weight_map)} tensors ({total / 2**30:.2f} GiB) "
        f"in {len(shard_tensors)} shards to {out}"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--meta", type=pathlib.Path, required=True, help="headers / config / tokenizer dir"
    )
    p.add_argument("--fetch", action="store_true", help="download the metadata into --meta")
    p.add_argument("--out", type=pathlib.Path)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--experts", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.fetch:
        fetch(args.meta.expanduser(), args.layers)
    if args.out is not None:
        make(args.meta.expanduser(), args.out.expanduser(), args.layers, args.experts, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
