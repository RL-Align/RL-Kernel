# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Pinned tiny-GPT checkpoint conversion and isolated vLLM test client."""

from __future__ import annotations

import json
import os
import select
import subprocess
from pathlib import Path

import torch

from rl_engine.integrations.common.ppo import PPORollout


def gpt2_weights(weights):
    """Convert this harness's dense Megatron GPT (H=128, heads=4) to HF GPT-2.

    QKV rows in Megatron are grouped by attention head; HF groups by Q/K/V.
    HF Conv1D weights are transposed. This is deliberately not a generic model
    converter: all source tensors must be consumed and match the test schema.
    """
    result, used = {}, set()

    def copy(target, source, *, transpose=False, qkv=False):
        """Store CPU ``weights[source]`` as ``result[target]`` and mark ``source`` as used.

        ``qkv`` regroups rows from per-head to Q/K/V order; ``transpose`` gives Conv1D layout.
        """
        value = weights[source].detach().cpu()
        used.add(source)
        if qkv:
            value = value.reshape(4, 3, 32, *value.shape[1:]).transpose(0, 1).reshape(value.shape)
        result[target] = (value.t() if transpose else value).contiguous()

    copy("transformer.wte.weight", "embedding.word_embeddings.weight")
    copy("transformer.wpe.weight", "embedding.position_embeddings.weight")
    copy("lm_head.weight", "output_layer.weight")
    for suffix in ("weight", "bias"):
        copy(f"transformer.ln_f.{suffix}", f"decoder.final_layernorm.{suffix}")
    for i in range(2):
        for target, source in (
            ("ln_1", "input_layernorm"),
            ("ln_2", "pre_mlp_layernorm"),
            ("attn.c_attn", "self_attention.linear_qkv"),
            ("attn.c_proj", "self_attention.linear_proj"),
            ("mlp.c_fc", "mlp.linear_fc1"),
            ("mlp.c_proj", "mlp.linear_fc2"),
        ):
            for suffix in ("weight", "bias"):
                copy(
                    f"transformer.h.{i}.{target}.{suffix}",
                    f"decoder.layers.{i}.{source}.{suffix}",
                    transpose=suffix == "weight" and "ln_" not in target,
                    qkv=target == "attn.c_attn",
                )
    if used != set(weights):
        raise ValueError(f"unsupported Megatron checkpoint keys: {set(weights) - used}")
    assert result["transformer.wte.weight"].shape == (32, 128)
    return result


class VllmPPOClient:
    def __init__(
        self,
        python: str,
        directory: Path,
        device,
        *,
        checkpoint=None,
        tp=1,
        cache_logprob_atol=None,
    ):
        """Launch the vLLM sidecar on ``tp`` GPUs from ``device`` and wait until it is ready.

        With ``checkpoint``, two fixed text prompts are encoded with its tokenizer and the
        sidecar serves that checkpoint; otherwise it serves the tiny GPT-2 harness. torchrun
        variables are removed from the sidecar's environment.
        """
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.checkpoint, self.tp = checkpoint, tp
        if checkpoint:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
            self.prompts = [
                tokenizer.encode(text)
                for text in (
                    "Explain in a few sentences why water freezes at low temperatures and "
                    "melts when heated.",
                    "Describe the steps needed to add two fractions with different "
                    "denominators, using an example.",
                )
            ]
        else:
            self.prompts = [[1, 2, 3, 4] * 5, [2, 3, 4, 5] * 4 + [2, 3]]
        self.installed, self.sampled, self.timings = [], [], []
        self.version = -1
        env = dict(os.environ)
        for key in list(env):
            if key in (
                "RANK",
                "WORLD_SIZE",
                "GROUP_RANK",
                "ROLE_RANK",
                "ROLE_WORLD_SIZE",
            ) or key.startswith(("MASTER_", "LOCAL_", "TORCHELASTIC_")):
                env.pop(key)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        env["CUDA_VISIBLE_DEVICES"] = (
            ",".join(visible.split(",")[device.index : device.index + tp])
            if visible
            else ",".join(str(device.index + i) for i in range(tp))
        )
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
        env["VLLM_PLUGINS"] = ""
        # Correctness check: Triton dot otherwise permits TF32 for FP32 inputs.
        env["TRITON_F32_DEFAULT"] = "ieee"
        env["VLLM_FLOAT32_MATMUL_PRECISION"] = "highest"
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        read_fd, write_fd = os.pipe()
        self.responses = os.fdopen(read_fd)
        self.log = (self.directory / "runtime.log").open("w")
        self.process = subprocess.Popen(
            [
                python,
                "-m",
                "tests.distributed.tp.vllm_ppo_sidecar",
                "--directory",
                str(self.directory / "model"),
                "--response-fd",
                str(write_fd),
            ]
            + (
                [
                    "--checkpoint",
                    str(checkpoint),
                    "--tp",
                    str(tp),
                    "--cache-logprob-atol",
                    str(cache_logprob_atol),
                ]
                if checkpoint
                else []
            ),
            stdin=subprocess.PIPE,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            pass_fds=(write_fd,),
        )
        os.close(write_fd)
        try:
            self.runtime = self._read()
            assert self.runtime["ready"]
        except Exception:
            self.process.terminate()
            self.process.wait(timeout=30)
            raise

    def _read(self):
        """Return the sidecar's next JSON response, waiting at most 600 seconds.

        Raises ``TimeoutError`` on timeout and ``RuntimeError`` if the sidecar exited or
        replied with an error.
        """
        if not select.select([self.responses], [], [], 600)[0]:
            raise TimeoutError(f"vLLM response timed out; see {self.directory}/runtime.log")
        line = self.responses.readline()
        if not line:
            raise RuntimeError(f"vLLM exited; see {self.directory}/runtime.log")
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(response["error"])
        return response

    def _request(self, request):
        """Send ``request`` to the sidecar as one JSON line and return its response."""
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        return self._read()

    def install(self, weights, version):
        """Convert actor weights to HF names, have the sidecar install them, return ``True``.

        ``version`` must follow the last installed one. Qwen3 checkpoints convert with
        ``megatron_to_hf``; the tiny harness uses ``gpt2_weights``.
        """
        if version != self.version + 1:
            raise ValueError("vLLM install versions must be consecutive")
        filename = self.directory / f"weights_v{version}.pt"
        if self.checkpoint:
            from rl_engine.integrations.engines.train.megatron.qwen3_checkpoint import (
                megatron_to_hf,
            )

            config = json.loads((Path(self.checkpoint) / "config.json").read_text())
            converted = megatron_to_hf([weights], config)
        else:
            converted = gpt2_weights(weights)
        torch.save(
            {"actor_version": version, "weights": converted} if self.checkpoint else converted,
            filename,
        )
        response = self._request({"op": "install", "filename": str(filename), "version": version})
        assert response["installed"] == version
        self.version = version
        self.installed.append(version)
        self.timings.append({"op": "install", **response})
        return True

    def sample(self, actor_version, critic_version):
        """Sample the fixed prompts on the installed version and return a left-padded rollout.

        Behavior logps are vLLM's sampled-token logprobs; ``terminated`` marks stop finishes.
        """
        assert self.version == actor_version
        prompts = self.prompts
        response = self._request({"op": "sample", "version": actor_version, "prompts": prompts})
        assert response["version"] == actor_version
        self.sampled.append(actor_version)
        self.timings.append(
            {"op": "sample", **{k: v for k, v in response.items() if k != "sequences"}}
        )
        sequences = response["sequences"]
        assert len(sequences) == len(prompts)
        length = max(len(s["prompt"]) + len(s["tokens"]) for s in sequences)
        ids = torch.zeros((2, length), dtype=torch.int64, device=self.device)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        actions = torch.zeros((2, length - 1), dtype=torch.bool, device=self.device)
        logps = torch.zeros_like(actions, dtype=torch.float32)
        for i, sequence in enumerate(sequences):
            assert sequence["prompt"] == prompts[i]
            tokens, scores = sequence["tokens"], sequence["logprobs"]
            assert len(tokens) == len(scores) > 0
            combined = prompts[i] + tokens
            # Left padding keeps the prompt/action positions explicit.
            start = length - len(combined)
            ids[i, start:] = torch.tensor(combined, device=self.device)
            mask[i, start:] = True
            action_start = start + len(prompts[i]) - 1
            actions[i, action_start:] = True
            logps[i, action_start:] = torch.tensor(scores, device=self.device)
        return PPORollout(
            ids,
            mask,
            (mask.long().cumsum(1) - 1).clamp(min=0),
            actions,
            logps,
            torch.tensor([s["terminated"] for s in sequences], device=self.device),
            torch.tensor([13, 7], device=self.device),
            torch.tensor(actor_version, device=self.device),
            torch.tensor(critic_version, device=self.device),
            torch.tensor([1.0, 0.0, 1.0], device=self.device),
        )

    def close(self):
        """Stop the sidecar, require exit code 0 and write ``report.json`` with its timings."""
        self._request({"op": "close"})
        self.process.wait(timeout=30)
        assert self.process.returncode == 0
        self.responses.close()
        self.log.close()
        (self.directory / "report.json").write_text(
            json.dumps(
                {
                    "runtime": self.runtime,
                    "installed": self.installed,
                    "sampled": self.sampled,
                    "timings": self.timings,
                },
                indent=2,
            )
        )
