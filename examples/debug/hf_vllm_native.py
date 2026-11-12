# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Single-GPU, two-sample HF/vLLM native prefill acceptance without RL kernels.

Creates a tiny random Llama locally (no model download), checks actual runtime
weights independently, and observes corresponding native boundaries. This is a
train-side forward, not backward/optimizer acceptance or a production scheduler
replay. Run as a module from the repository root:

    VLLM_PLUGINS='' python -m examples.debug.hf_vllm_native --output /tmp/unique-run
"""

from __future__ import annotations

import argparse
import json
import os
from functools import partial
from pathlib import Path

import torch

from rl_engine.alignment.debug import (
    AutoCapture,
    LogicalTensor,
    ModuleBoundary,
    fingerprint,
    prepare_auto,
)
from rl_engine.alignment.debug.entry import analyze, print_report


PROMPTS = [[11, 12, 13, 14], [21, 22, 23, 24]]


class LlamaPrefillBinding:
    """A deliberately small semantic map, NOT a model/kernel implementation."""

    def __init__(self, model, side):
        self.model, self.side = model, side
        self.tokens = torch.tensor(PROMPTS, dtype=torch.long)
        self.coordinates = {
            token: (sample, position)
            for sample, prompt in enumerate(PROMPTS)
            for position, token in enumerate(prompt)
        }
        self.rows = []

    def logical(self, tensor):
        return LogicalTensor(tensor.reshape(len(self.rows), -1), self.rows)

    def root_input(self, module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        positions = (
            kwargs.get("position_ids") if self.side == "training" else kwargs.get("positions")
        )
        if ids is None or positions is None:
            raise ValueError("this prefill map requires actual token and position tensors")
        observed = ids.detach().cpu().reshape(-1).tolist()
        if len(observed) != len(self.coordinates) or set(observed) != set(self.coordinates):
            raise ValueError("scheduler changed the frozen batch; cannot use this prefill-only map")
        self.rows = [self.coordinates[token] for token in observed]
        actual_positions = positions.detach().cpu().reshape(-1).tolist()
        if actual_positions != [row[1] for row in self.rows]:
            raise ValueError("actual position IDs differ from the frozen case")
        return {
            "token_ids": self.logical(ids.to(torch.int64)),
            "position_ids": self.logical(positions.to(torch.int64)),
        }

    def root_output(self, module, args, kwargs, output):
        return self.logical(output.hidden_states[-1] if self.side == "training" else output)

    def module_input(self, module, args, kwargs):
        value = args[0] if args else kwargs.get("input_", kwargs.get("x"))
        return self.logical(value)

    def embedding_input(self, module, args, kwargs):
        value = args[0] if args else kwargs.get("input_ids", kwargs.get("input_"))
        return self.logical(value.to(torch.int64))

    def module_output(self, module, args, kwargs, output):
        return self.logical(output[0] if isinstance(output, tuple) else output)

    def boundaries(self):
        bindings = {
            "embedding": ModuleBoundary(
                "model.embed_tokens", self.embedding_input, self.module_output
            ),
        }
        for layer in range(2):
            for role, path in (
                ("attention_projection", "self_attn.o_proj"),
                ("ffn_down", "mlp.down_proj"),
            ):
                bindings[f"layer.{layer}.{role}"] = ModuleBoundary(
                    f"model.layers.{layer}.{path}", self.module_input, self.module_output, layer
                )
        return bindings

    def weights(self):
        if self.side == "training":
            return self.model.state_dict()
        # Runtime-native fused storage -> canonical HF logical parameters.
        # TP=1, no quantization and fixed dimensions are checked by this example.
        source = dict(self.model.named_parameters())
        result = {}
        for name, value in source.items():
            if ".qkv_proj.weight" in name:
                for suffix, piece in zip(
                    ("q_proj", "k_proj", "v_proj"), value.split(128, dim=0), strict=True
                ):
                    result[name.replace("qkv_proj", suffix)] = piece
            elif ".gate_up_proj.weight" in name:
                for suffix, piece in zip(
                    ("gate_proj", "up_proj"), value.split(256, dim=0), strict=True
                ):
                    result[name.replace("gate_up_proj", suffix)] = piece
            elif name in {"model.embed_tokens.weight", "lm_head.weight"}:
                result[name] = value[:256]  # exclude runtime vocabulary padding
            else:
                result[name] = value
        return result

    def identity(self):
        # Actual token/position values are also checked in root_input on each forward.
        return {
            "weights": "tiny-llama-frozen; independently checked observed_weights",
            "tokens": fingerprint(self.tokens),
            "positions": list(range(4)),
            "mask": [[1] * 4] * 2,
            "sampling": {"temperature": 0, "max_tokens": 1},
            "context": "two complete causal prefill sequences; no padding or prefix cache",
        }

    def observer(self, directory):
        return AutoCapture(
            directory,
            self.side,
            self.identity(),
            self.model,
            positions=range(4),
            bindings=self.boundaries(),
            root_inputs=self.root_input,
            root_outputs=self.root_output,
            weights=self.weights,
        )


def install_rollout(model, directory, inject):
    binding = LlamaPrefillBinding(model, "rollout")
    observer = binding.observer(Path(directory))
    handle = None
    if inject:

        def change_embedding(module, args, output):
            value = output.clone()
            value[0, 0] += 0.125
            return value

        handle = model.model.embed_tokens.register_forward_hook(change_embedding)
    observer.__enter__()
    # Keep the session and hook alive across separate worker RPCs.
    model._generic_debug_observer = observer
    model._generic_debug_injection = handle
    return {"installed": True}


def close_rollout(model):
    model._generic_debug_observer.__exit__(None, None, None)
    if model._generic_debug_injection is not None:
        model._generic_debug_injection.remove()
    del model._generic_debug_observer
    del model._generic_debug_injection
    return {"closed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("this integration acceptance requires a GPU")
    # Disable optional RL Kernel/vLLM plugins for this process and its workers.
    os.environ["VLLM_PLUGINS"] = ""
    # In-process single-GPU observation avoids enabling insecure RPC/pickle
    # serialization merely to install these trusted local Python callbacks.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from transformers import LlamaConfig, LlamaForCausalLM
    from vllm import LLM, SamplingParams

    torch.manual_seed(442)
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        tie_word_embeddings=False,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).to(dtype=torch.bfloat16).eval()
    model.save_pretrained(root / "checkpoint")
    model = model.cuda()
    binding = LlamaPrefillBinding(model, "training")
    tokens = binding.tokens.cuda()
    positions = torch.arange(4, device="cuda").expand(2, -1)
    for arm in ("native", "injected"):
        directory = root / arm
        prepare_auto(
            directory,
            model,
            positions=range(4),
            scope="HF train-side eager vs vLLM native TP=1 prefill; 2 samples, 4 tokens each",
            bindings=binding.boundaries(),
        )
        with torch.inference_mode(), binding.observer(directory):
            model(
                input_ids=tokens,
                position_ids=positions,
                attention_mask=torch.ones_like(tokens),
                use_cache=False,
                output_hidden_states=True,
            )
    model.cpu()
    del model, binding
    torch.cuda.empty_cache()
    engine = LLM(
        model=str(root / "checkpoint"),
        skip_tokenizer_init=True,
        tensor_parallel_size=1,
        dtype="bfloat16",
        enforce_eager=True,
        compilation_config=0,
        max_model_len=32,
        max_num_seqs=2,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        kv_cache_memory_bytes=256 * 1024 * 1024,
        gpu_memory_utilization=0.1,
    )
    summary = {}
    for arm in ("native", "injected"):
        directory = str(root / arm)
        engine.apply_model(partial(install_rollout, directory=directory, inject=arm == "injected"))
        engine.generate(
            [{"prompt_token_ids": prompt} for prompt in PROMPTS],
            SamplingParams(temperature=0, max_tokens=1, ignore_eos=True, detokenize=False),
            use_tqdm=False,
        )
        engine.apply_model(close_rollout)
        report = analyze(Path(directory))
        print_report(report, arm)
        if report["status"] == "not_comparable":
            raise AssertionError(f"{arm}: evidence gates failed: {report['errors']}")
        if arm == "injected":
            assert report["status"] == "diverged"
            assert report["first_divergence"]["boundary"] == "embedding.output"
            assert report["first_divergence"]["input_evidence"] == "equal"
        summary[arm] = {
            "status": report["status"],
            "first_divergence": report["first_divergence"],
            "events": len(report["boundaries"]),
        }
    summary["scope"] = "real native HF/vLLM prefill only; no decode/optimizer/TP>1 acceptance"
    summary["versions"] = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(),
    }
    (root / "acceptance.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Acceptance: {root / 'acceptance.json'}")
    # This example owns its in-process engine and distributed groups.
    engine.llm_engine.engine_core.shutdown()
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
