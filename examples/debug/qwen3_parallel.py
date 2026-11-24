# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Native Qwen3 Dense Megatron/TE versus vLLM parallel capture matrix.

Each unique runtime topology executes once. Completed, immutable lane snapshots
are then paired offline, avoiding a Cartesian product of expensive GPU jobs.
No Ray server, production process, kernel replacement or optimizer is involved.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from functools import partial
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch
import torch.distributed as dist

from rl_engine.alignment.debug import AutoCapture, LogicalTensor, ModuleBoundary, fingerprint
from rl_engine.alignment.debug.auto import prepare_auto
from rl_engine.alignment.debug.adapters import Qwen3Adapter
from rl_engine.alignment.debug.entry import analyze
from rl_engine.alignment.debug.qwen3_layout import layer_parameters

PROMPTS = [list(range(1001, 1017)), list(range(2001, 2017))]
PARAMETER_NAMES = {
    "attention_norm": "input_layernorm.weight",
    "mlp_norm": "post_attention_layernorm.weight",
    "q_norm": "self_attn.q_norm.weight",
    "k_norm": "self_attn.k_norm.weight",
    **{key: f"self_attn.{key}_proj.weight" for key in ("q", "k", "v", "o")},
    **{key: f"mlp.{key}_proj.weight" for key in ("gate", "up", "down")},
}


class Checkpoint:
    """Read the existing official checkpoint lazily, one parameter at a time."""

    def __init__(self, root):
        from safetensors import safe_open

        self.root = Path(root)
        self.config = json.loads((self.root / "config.json").read_text())
        if not Qwen3Adapter().matches(self.config, "rocm"):
            raise ValueError("use a supported, unmodified official Qwen3 Dense configuration")
        self.stack = ExitStack()
        index = self.root / "model.safetensors.index.json"
        if index.is_file():
            self.index = json.loads(index.read_text())["weight_map"]
        else:
            with safe_open(self.root / "model.safetensors", framework="pt") as handle:
                self.index = {name: "model.safetensors" for name in handle.keys()}
        self.handles = {
            name: self.stack.enter_context(safe_open(self.root / name, framework="pt"))
            for name in sorted(set(self.index.values()))
        }

    def __getitem__(self, name):
        return self.handles[self.index[name]].get_tensor(name)

    def close(self):
        self.stack.close()


def load_megatron(model, checkpoint, tp_rank, tp_size):
    """Inverse of the repository's existing grouped-QKV/gate-up storage mapping."""
    config = checkpoint.config

    def copy(target, name, axis=None):
        value = checkpoint[name]
        if axis is not None:
            value = value.chunk(tp_size, dim=axis)[tp_rank]
        if target.shape != value.shape:
            raise ValueError(
                f"{name}: runtime/checkpoint shape mismatch: {target.shape}/{value.shape}"
            )
        target.copy_(value)

    with torch.no_grad():
        copy(model.embedding.word_embeddings.weight, "model.embed_tokens.weight", 0)
        copy(model.output_layer.weight, "lm_head.weight", 0)
        copy(model.decoder.final_layernorm.weight, "model.norm.weight")
        for index, layer in enumerate(model.decoder.layers):
            prefix = f"model.layers.{index}."
            shards = layer_parameters(layer, "training", tp_rank, tp_size)
            for name in ("attention_norm", "mlp_norm", "q_norm", "k_norm", "o", "down"):
                copy(shards[name].value, prefix + PARAMETER_NAMES[name], shards[name].axis)
            head = config["head_dim"]
            groups = config["num_key_value_heads"] // tp_size
            qkv = [
                checkpoint[prefix + PARAMETER_NAMES[name]]
                .chunk(tp_size, dim=0)[tp_rank]
                .reshape(groups, -1, head, config["hidden_size"])
                for name in ("q", "k", "v")
            ]
            layer.self_attention.linear_qkv.weight.copy_(torch.cat(qkv, dim=1).flatten(0, 2))
            gate_up = [
                checkpoint[prefix + PARAMETER_NAMES[name]].chunk(tp_size, dim=0)[tp_rank]
                for name in ("gate", "up")
            ]
            layer.mlp.linear_fc1.weight.copy_(torch.cat(gate_up, dim=0))


class Qwen3Observation:
    """Only logical ownership, selectors and independent weight evidence."""

    def __init__(self, model, side, config, tp_rank, tp_size, tp_group):
        self.model, self.side, self.config = model, side, config
        self.tp_rank, self.tp_size, self.tp_group = tp_rank, tp_size, tp_group
        self.rows = []
        self.final_hidden = None

    @property
    def training(self):
        return self.side == "training"

    def logical(self, value, *, sharded=False):
        value = value.reshape(len(self.rows), -1)
        width = value.shape[-1]
        return LogicalTensor(
            value,
            self.rows,
            self.tp_rank * width if sharded else 0,
            width * self.tp_size if sharded else width,
        )

    def root_input(self, module, args, kwargs):
        tokens = kwargs["input_ids"]
        positions = kwargs["position_ids" if self.training else "positions"]
        if self.training:
            # Native Megatron activations are [sequence, batch, hidden].
            self.rows = [
                (sample, int(positions[sample, index]))
                for index in range(tokens.shape[1])
                for sample in range(tokens.shape[0])
            ]
            tokens, positions = tokens.T.contiguous(), positions.T.contiguous()
        else:
            coordinates = {
                token: (sample, position)
                for sample, prompt in enumerate(PROMPTS)
                for position, token in enumerate(prompt)
            }
            self.rows = [coordinates[token] for token in tokens.cpu().tolist()]
        if [position for _, position in self.rows] != positions.reshape(-1).cpu().tolist():
            raise ValueError("native position IDs do not match logical token coordinates")
        expected = [PROMPTS[sample][position] for sample, position in self.rows]
        if tokens.reshape(-1).cpu().tolist() != expected:
            raise ValueError("native runtime changed the frozen tokens")
        return {
            "tokens": self.logical(tokens.to(torch.int64)),
            "positions": self.logical(positions.to(torch.int64)),
        }

    def root_output(self, module, args, kwargs, output):
        # Native Megatron's root returns logits; vLLM's returns hidden states.
        # Compare the actual final-norm output observed below, not a recomputation.
        if self.final_hidden is None:
            raise ValueError("final hidden-state endpoint was not observed")
        return self.logical(self.final_hidden)

    def embedding_input(self, module, args, kwargs):
        value = (
            kwargs.get("input_ids") if self.training else (args[0] if args else kwargs["input_"])
        )
        return self.logical((value.T.contiguous() if self.training else value).to(torch.int64))

    def projection_input(self, module, args, kwargs):
        return self.logical(args[0], sharded=True)

    def output(self, module, args, kwargs, output):
        return self.logical(output[0] if isinstance(output, tuple) else output)

    def final_output(self, module, args, kwargs, output):
        self.final_hidden = output[0] if isinstance(output, tuple) else output
        return self.logical(self.final_hidden)

    def bindings(self):
        result = {
            "embedding": ModuleBoundary(
                "embedding" if self.training else "model.embed_tokens",
                self.embedding_input,
                self.output,
            ),
        }
        for layer in range(self.config["num_hidden_layers"]):
            prefix = f"decoder.layers.{layer}" if self.training else f"model.layers.{layer}"
            paths = (
                ("self_attention.linear_proj", "mlp.linear_fc2")
                if self.training
                else ("self_attn.o_proj", "mlp.down_proj")
            )
            for role, path in zip(("attention_projection", "ffn_down"), paths, strict=True):
                result[f"layer.{layer}.{role}"] = ModuleBoundary(
                    f"{prefix}.{path}",
                    self.projection_input,
                    self.output,
                    layer,
                )
        result["final_norm"] = ModuleBoundary(
            "decoder.final_layernorm" if self.training else "model.norm",
            outputs=self.final_output,
            observe_inputs=False,
        )
        return result

    def _gather(self, value, axis):
        if self.tp_size == 1 or axis is None:
            return value
        shards = [torch.empty_like(value) for _ in range(self.tp_size)]
        dist.all_gather(shards, value.contiguous(), group=self.tp_group)
        return torch.cat(shards, dim=axis)

    def weights(self):
        # Yield one reconstructed actual parameter at a time: never retain
        # another entire 8B model, and never substitute checkpoint hashes.
        model = self.model
        embed = model.embedding.word_embeddings if self.training else model.model.embed_tokens
        head = model.output_layer if self.training else model.lm_head
        norm = model.decoder.final_layernorm if self.training else model.model.norm
        vocab = self.config["vocab_size"]
        yield "model.embed_tokens.weight", self._gather(embed.weight, 0)[:vocab]
        yield "lm_head.weight", self._gather(head.weight, 0)[:vocab]
        yield "model.norm.weight", norm.weight
        layers = model.decoder.layers if self.training else model.model.layers
        for index, layer in enumerate(layers):
            for name, shard in layer_parameters(
                layer, self.side, self.tp_rank, self.tp_size
            ).items():
                yield (
                    f"model.layers.{index}.{PARAMETER_NAMES[name]}",
                    self._gather(shard.value, shard.axis),
                )

    def identity(self):
        return {
            "weights": "official Qwen3 Dense; independently reconstructed runtime parameters",
            "tokens": fingerprint(torch.tensor(PROMPTS, dtype=torch.int64)),
            "positions": list(range(len(PROMPTS[0]))),
            "mask": [[1] * len(prompt) for prompt in PROMPTS],
            "sampling": {"temperature": 0, "max_tokens": 1},
            "context": "two complete unpadded causal prefill sequences; no prefix cache",
        }

    def prepare(self, root, world):
        prepare_auto(
            root,
            self.model,
            positions=range(len(PROMPTS[0])),
            scope="Qwen3 Dense native Megatron/TE and vLLM eager prefill; final hidden endpoint",
            bindings=self.bindings(),
            ranks={
                side: list(range(world)) if side == self.side else [0]
                for side in ("training", "rollout")
            },
        )

    def observer(self, root, rank):
        return AutoCapture(
            root,
            self.side,
            self.identity(),
            self.model,
            rank=rank,
            positions=range(len(PROMPTS[0])),
            bindings=self.bindings(),
            root_inputs=self.root_input,
            root_outputs=self.root_output,
            weights=self.weights,
        )


def train(args):
    from megatron.core import parallel_state as ps
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.utils import get_batch_on_this_cp_rank

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    ps.initialize_model_parallel(tensor_model_parallel_size=args.tp, context_parallel_size=args.cp)
    model_parallel_cuda_manual_seed(442)
    checkpoint = Checkpoint(args.model)
    cfg = checkpoint.config
    config = TransformerConfig(
        num_layers=cfg["num_hidden_layers"],
        hidden_size=cfg["hidden_size"],
        ffn_hidden_size=cfg["intermediate_size"],
        num_attention_heads=cfg["num_attention_heads"],
        num_query_groups=cfg["num_key_value_heads"],
        kv_channels=cfg["head_dim"],
        tensor_model_parallel_size=args.tp,
        context_parallel_size=args.cp,
        params_dtype=torch.bfloat16,
        bf16=True,
        normalization="RMSNorm",
        qk_layernorm=True,
        layernorm_epsilon=cfg["rms_norm_eps"],
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        sequence_parallel=False,
        gradient_accumulation_fusion=False,
        bias_activation_fusion=False,
        bias_dropout_fusion=False,
        apply_rope_fusion=False,
        cp_comm_type="p2p",
    )
    model = (
        GPTModel(
            config,
            get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True),
            vocab_size=cfg["vocab_size"],
            max_sequence_length=32,
            position_embedding_type="rope",
            rotary_base=cfg["rope_theta"],
            share_embeddings_and_output_weights=cfg["tie_word_embeddings"],
            parallel_output=True,
        )
        .cuda()
        .eval()
    )
    rank = ps.get_tensor_model_parallel_rank()
    load_megatron(model, checkpoint, rank, args.tp)
    checkpoint.close()
    binding = Qwen3Observation(
        model, "training", cfg, rank, args.tp, ps.get_tensor_model_parallel_group()
    )
    if dist.get_rank() == 0:
        binding.prepare(args.output, dist.get_world_size())
    dist.barrier()
    batch = {
        "tokens": torch.tensor(PROMPTS, device="cuda"),
        "position_ids": torch.arange(len(PROMPTS[0]), device="cuda")
        .expand(len(PROMPTS), -1)
        .clone(),
        "attention_mask": None,
    }
    batch = get_batch_on_this_cp_rank(
        batch,
        is_hybrid_cp=False,
        cp_group=ps.get_context_parallel_group(),
    )
    token_slices = [None] * args.tp
    dist.all_gather_object(
        token_slices,
        (fingerprint(batch["tokens"]), fingerprint(batch["position_ids"])),
        group=ps.get_tensor_model_parallel_group(),
    )
    if len(set(token_slices)) != 1:
        raise ValueError("TP replicas must consume identical CP token/position slices")
    with torch.inference_mode(), binding.observer(args.output, dist.get_rank()):
        model(
            input_ids=batch["tokens"],
            position_ids=batch["position_ids"],
            attention_mask=None,
        )
    dist.barrier()
    if dist.get_rank() == 0:
        print(f"Completed native Megatron TP={args.tp} CP={args.cp}: {args.output}", flush=True)
    ps.destroy_model_parallel()
    dist.destroy_process_group()


def install_rollout(model, root, cfg, inject):
    from vllm.distributed import get_tp_group

    group = get_tp_group()
    binding = Qwen3Observation(
        model,
        "rollout",
        cfg,
        group.rank_in_group,
        group.world_size,
        group.device_group,
    )
    if group.rank_in_group == 0:
        binding.prepare(Path(root), group.world_size)
    dist.barrier(group=group.device_group)
    handle = None
    if inject:

        def change(module, args, output):
            result = output.clone()
            result[0, 0] += 0.125
            return result

        handle = model.model.embed_tokens.register_forward_hook(change)
    observer = binding.observer(Path(root), group.rank_in_group)
    observer.__enter__()
    model._generic_debug_state = observer, handle
    return True


def close_rollout(model):
    observer, handle = model._generic_debug_state
    observer.__exit__(None, None, None)
    if handle:
        handle.remove()
    del model._generic_debug_state
    return True


def rollout(args):
    os.environ["VLLM_PLUGINS"] = ""
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    cfg = json.loads((args.model / "config.json").read_text())
    engine = LLM(
        model=str(args.model),
        skip_tokenizer_init=True,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        enforce_eager=True,
        compilation_config=0,
        max_model_len=32,
        max_num_seqs=len(PROMPTS),
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        kv_cache_memory_bytes=256 * 1024 * 1024,
        gpu_memory_utilization=0.2,
    )
    for inject in (False, True):
        root = args.output if not inject else args.output.with_name(args.output.name + "-injected")
        engine.apply_model(partial(install_rollout, root=str(root), cfg=cfg, inject=inject))
        engine.generate(
            [{"prompt_token_ids": prompt} for prompt in PROMPTS],
            SamplingParams(temperature=0, max_tokens=1, ignore_eos=True, detokenize=False),
            use_tqdm=False,
        )
        engine.apply_model(close_rollout)
    engine.llm_engine.engine_core.shutdown()
    cleanup_dist_env_and_memory()


def pair_lanes(training: Path, rollout_root: Path, output: Path):
    """Pair immutable evidence cheaply; no weights or tensors are recomputed."""
    left = json.loads((training / "capture-manifest.json").read_text())
    right = json.loads((rollout_root / "capture-manifest.json").read_text())
    for key in ("schema_version", "boundaries", "coverage", "scope"):
        if left[key] != right[key]:
            raise ValueError(f"lane capture plans differ in {key}")
    manifest = {
        **left,
        "ranks": {
            "training": left["ranks"]["training"],
            "rollout": right["ranks"]["rollout"],
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "capture-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for side, source in (("training", training), ("rollout", rollout_root)):
        for rank in manifest["ranks"][side]:
            target = output / "events" / f"{side}-rank{rank}"
            target.mkdir(parents=True)
            for path in (source / "events" / target.name).iterdir():
                try:
                    os.link(path, target / path.name)
                except OSError:
                    shutil.copy2(path, target / path.name)
    return analyze(output)


def matrix_axes(args):
    """Validate the small declarative plan before any GPU process is started."""
    devices = [item.strip() for item in args.devices.split(",")]
    topologies = [tuple(map(int, item.split("x"))) for item in args.training.split(",")]
    rollout_tps = [int(item) for item in args.rollout_tps.split(",")]
    if any(not item.isdecimal() for item in devices) or len({int(item) for item in devices}) != len(
        devices
    ):
        raise ValueError("devices must be distinct nonnegative device IDs")
    devices = [str(int(item)) for item in devices]
    if any(len(item) != 2 or min(item) < 1 for item in topologies):
        raise ValueError("training topologies must be positive TPxCP pairs")
    if len(set(topologies)) != len(topologies) or len(set(rollout_tps)) != len(rollout_tps):
        raise ValueError("parallel matrix axes must not contain duplicates")
    if any(tp < 1 for tp in rollout_tps):
        raise ValueError("rollout TP sizes must be positive")
    if any(cp > 1 and len(PROMPTS[0]) % (2 * cp) for _, cp in topologies):
        raise ValueError("frozen sequence length must be divisible by 2*CP")
    return devices, topologies, rollout_tps


def matrix(args):
    devices, topologies, rollout_tps = matrix_axes(args)
    if any(tp * cp > len(devices) for tp, cp in topologies) or max(rollout_tps) > len(devices):
        raise ValueError("requested topology exceeds explicitly allocated devices")
    args.output.mkdir(parents=True, exist_ok=False)
    lanes = args.output / "lanes"
    lanes.mkdir()
    env = os.environ.copy()
    for key in ("ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        env.pop(key, None)
    env["OMP_NUM_THREADS"] = "1"
    env["VLLM_PLUGINS"] = ""
    module = "examples.debug.qwen3_parallel"
    jobs = [("training", tp, cp, f"train-tp{tp}-cp{cp}") for tp, cp in topologies] + [
        ("rollout", tp, 1, f"rollout-tp{tp}") for tp in rollout_tps
    ]
    for side, tp, cp, name in jobs:
        world = tp * cp
        command = [sys.executable]
        if side == "training":
            command += ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={world}"]
        command += [
            "-m",
            module,
            "train" if side == "training" else "rollout",
            "--model",
            str(args.model),
            "--output",
            str(lanes / name),
            "--tp",
            str(tp),
            "--cp",
            str(cp),
        ]
        job_env = {**env, "HIP_VISIBLE_DEVICES": ",".join(devices[:world])}
        print(f"[capture] {name}", flush=True)
        with (args.output / f"{name}.log").open("w") as log:
            subprocess.run(command, env=job_env, stdout=log, stderr=subprocess.STDOUT, check=True)
    compare_matrix(args)


def compare_matrix(args):
    """Rebuild a matrix from completed lanes without launching either framework."""
    lanes = args.output / "lanes"
    devices, topologies, rollout_tps = matrix_axes(args)
    results = []
    for tp, cp in topologies:
        for rollout_tp in rollout_tps:
            for injected in (False, True):
                name = f"train-tp{tp}-cp{cp}_rollout-tp{rollout_tp}" + (
                    "-injected" if injected else ""
                )
                report = pair_lanes(
                    lanes / f"train-tp{tp}-cp{cp}",
                    lanes / (f"rollout-tp{rollout_tp}" + ("-injected" if injected else "")),
                    args.output / name,
                )
                passed = report["status"] != "not_comparable"
                if injected:
                    first = report["first_divergence"] or {}
                    passed &= (
                        report["status"] == "diverged"
                        and first.get("boundary") == "embedding.output"
                        and first.get("input_evidence") == "equal"
                    )
                results.append(
                    {
                        "case": name,
                        "passed": passed,
                        "status": report["status"],
                        "errors": report["errors"],
                        "first_divergence": report["first_divergence"],
                        "events": len(report["boundaries"]),
                    }
                )
                print(f"[compare] {name}: {report['status']} passed={passed}", flush=True)
    summary = {
        "passed": all(row["passed"] for row in results),
        "results": results,
        "scope": "native Qwen3 Dense pre-update eager prefill; no optimizer or decode acceptance",
        "model": str(args.model),
        "devices": devices,
    }
    (args.output / "matrix.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not summary["passed"]:
        raise RuntimeError(f"parallel acceptance failed; inspect {args.output / 'matrix.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "rollout", "matrix", "compare"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--training", default="1x1,2x1,1x2")
    parser.add_argument("--rollout-tps", default="1,2")
    args = parser.parse_args()
    args.model, args.output = args.model.resolve(), args.output.resolve()
    {"train": train, "rollout": rollout, "matrix": matrix, "compare": compare_matrix}[args.mode](
        args
    )


if __name__ == "__main__":
    main()
