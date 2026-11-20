# SPDX-License-Identifier: Apache-2.0
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from safetensors.torch import save_file
from tokenizers import Tokenizer

from rl_engine.models.deepseek_v41.config import MODEL_ID, OFFICIAL_REVISION
from rl_engine.models.deepseek_v41.engram import build_compressed_token_map


def main():
    parser = argparse.ArgumentParser(
        description="Prepare the exact released compressed Engram token map"
    )
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    url = f"https://huggingface.co/{MODEL_ID}/resolve/{OFFICIAL_REVISION}/tokenizer.json"
    with urllib.request.urlopen(url, timeout=60) as response:
        raw = response.read()
    path = args.output / "tokenizer.json"
    path.write_bytes(raw)
    mapping, size = build_compressed_token_map(Tokenizer.from_file(str(path)))
    if size != 99092 or mapping.numel() != 129280:
        raise ValueError(f"released tokenizer contract mismatch: {mapping.numel()} {size}")
    save_file({"token_map": mapping}, str(args.output / "token-map.safetensors"))
    evidence = {
        "revision": OFFICIAL_REVISION,
        "source": url,
        "tokenizer_sha256": hashlib.sha256(raw).hexdigest(),
        "token_count": mapping.numel(),
        "compressed_vocab_size": size,
        "map_sha256": hashlib.sha256(mapping.numpy().tobytes()).hexdigest(),
    }
    (args.output / "manifest.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
