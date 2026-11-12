# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Device acceptance for automatic discovery: no per-model boundary mappings."""

import argparse
import json
from pathlib import Path

import torch

from rl_engine.alignment.debug.discover_runtime import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model", type=Path, help="optional existing checkpoint instead of tiny fixtures"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tokens = [[11, 12, 13, 14], [21, 22, 23, 24]]
    (args.output / "tokens.json").write_text(json.dumps(tokens) + "\n")
    if args.model:
        checkpoints = {"checkpoint": args.model}
    else:
        from transformers import AutoModelForCausalLM, GPT2Config, LlamaConfig

        configs = {
            "gpt2": GPT2Config(
                vocab_size=256,
                n_positions=64,
                n_embd=128,
                n_layer=3,
                n_head=4,
                resid_pdrop=0,
                embd_pdrop=0,
                attn_pdrop=0,
                bos_token_id=1,
                eos_token_id=2,
            ),
            "llama": LlamaConfig(
                vocab_size=256,
                hidden_size=128,
                intermediate_size=256,
                num_hidden_layers=3,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=64,
                attention_dropout=0,
                bos_token_id=1,
                eos_token_id=2,
                tie_word_embeddings=False,
            ),
        }
        checkpoints = {}
        for name, config in configs.items():
            torch.manual_seed(442)
            model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16).eval()
            checkpoint = args.output / f"{name}-weights"
            model.save_pretrained(checkpoint)
            checkpoints[name] = checkpoint
            del model
    for name, checkpoint in checkpoints.items():
        report = run(checkpoint, tokens, args.output / name, fault_control=True)
        if report["automatic_mapping"]["status"] != "validated":
            raise RuntimeError(f"{name}: automatic correspondence was not validated")
    print(f"Automatic discovery acceptance completed: {args.output}")


if __name__ == "__main__":
    main()
