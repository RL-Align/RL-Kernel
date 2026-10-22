# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Measurements of the RFC #420 H3 integration rows (runtime modes, ablation overlay, Diffusers
adapter, training provider, LoRA gate, CPS transition): JSON report plus a PNG.

    python tools/validation/models/h3_runtime_report.py --out report.json --png figure.png \
        [--checkpoint <dir with transformer/>] [--verl-omni <checkout>]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from rl_engine.models.minimax_h3 import runtime as rt
from rl_engine.models.minimax_h3 import transition as tr


def rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-300)).item()


def lora_rows():
    out = []
    for n, k, t in ((512, 384, 70), (2048, 1024, 257), (5120, 5120, 33)):
        g = torch.Generator().manual_seed(n)
        w = (torch.randn(n, k, generator=g) / k**0.5).bfloat16().cuda()
        x = torch.randn(t, k, generator=g).bfloat16().cuda()
        grad = torch.randn(t, n, generator=g).bfloat16().cuda()
        gate = dict(rt.lora_gradient_gate(w, x, grad))
        # gradients against an FP64 autograd of the same function
        r, scale = 8, 2.0
        gen = torch.Generator().manual_seed(0)
        a0 = (torch.randn(r, k, generator=gen) / k**0.5).cuda()
        b0 = (torch.randn(n, r, generator=gen) * 0.1).cuda()
        a, b = a0.clone().requires_grad_(), b0.clone().requires_grad_()
        xi = x.clone().requires_grad_()
        rt.strict_lora_linear(xi, w, a, b, scale).backward(grad)
        ad, bd, xd = (v.detach().double().requires_grad_() for v in (a0, b0, x))
        y = xd @ w.double().t() + scale * (xd @ ad.t()) @ bd.t()
        (y * grad.double()).sum().backward()
        gate.update(
            shape=f"N{n} K{k} T{t}",
            grad_a_rel_err_vs_fp64=rel(a.grad, ad.grad),
            grad_b_rel_err_vs_fp64=rel(b.grad, bd.grad),
            grad_x_rel_err_vs_fp64=rel(xi.grad, xd.grad),
        )
        out.append(gate)
    return out


def scheduler_cps(x, v, sigma, sigma_prev, noise, level):
    s = math.sin(level * math.pi / 2)
    std = sigma_prev * s
    mean = (x - sigma * v) * (1 - sigma_prev) + (x + v * (1 - sigma)) * torch.sqrt(
        sigma_prev**2 - std**2
    )
    prev = mean + std * noise
    return prev, mean, (-((prev - mean) ** 2)).mean(dim=tuple(range(1, prev.ndim)))


def transition_rows(verl_omni):
    sched = None
    if verl_omni:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "flow_match_sde",
            Path(verl_omni) / "verl_omni/pipelines/schedulers/flow_match_sde.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        FlowMatchSDEDiscreteScheduler = module.FlowMatchSDEDiscreteScheduler

        sched = FlowMatchSDEDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
        sched.sigmas = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
        sched.timesteps = sched.sigmas[:-1] * 1000
        sched._step_index = 1
    rows = []
    for batch, shape in ((3, (257, 96)), (4, (4096, 96)), (2, (8192, 32))):
        g = torch.Generator().manual_seed(1)
        x = torch.randn(batch, *shape, generator=g).cuda()
        v = torch.randn(batch, *shape, generator=g).cuda()
        noise = torch.stack([tr.step_noise(7, 2, i, "video", shape, "cuda") for i in range(batch)])
        sg, sp = torch.tensor(0.75, device="cuda"), torch.tensor(0.5, device="cuda")
        out = tr.cps_transition(x, v, sg, sp, noise=noise)
        prev, mean, logp = scheduler_cps(x, v, sg, sp, noise, tr.NOISE_LEVEL)
        row = dict(
            shape=f"B{batch} x {shape[0]}x{shape[1]}",
            prev_sample_bitwise=bool(torch.equal(out.prev_sample, prev)),
            mean_bitwise=bool(torch.equal(out.prev_sample_mean, mean)),
            logp_max_rel_diff_vs_torch_mean=((out.log_prob - logp).abs() / logp.abs()).max().item(),
        )
        for i in range(batch):
            one = tr.cps_transition(x[i : i + 1], v[i : i + 1], sg, sp, noise=noise[i : i + 1])
            if not (
                torch.equal(one.prev_sample, out.prev_sample[i : i + 1])
                and torch.equal(one.log_prob, out.log_prob[i : i + 1])
            ):
                row["batch_invariant"] = False
                break
        else:
            row["batch_invariant"] = True
        scored = tr.cps_transition(x, v, sg, sp, prev_sample=out.prev_sample)
        row["scoring_reproduces_sampling_logprob"] = bool(
            torch.equal(scored.log_prob, out.log_prob)
        )
        if sched is not None:
            ref = sched.sample_previous_step(
                sample=x.cpu(),
                model_output=v.cpu(),
                prev_sample=out.prev_sample.cpu(),
                noise_level=tr.NOISE_LEVEL,
                sde_type=tr.SDE_TYPE,
            )
            row["verl_omni_mean_bitwise"] = bool(torch.equal(ref[2], out.prev_sample_mean.cpu()))
            row["verl_omni_logp_max_rel_diff"] = (
                ((ref[1] - out.log_prob.cpu()).abs() / ref[1].abs()).max().item()
            )
        rows.append(row)
    return rows


def model_rows(checkpoint):
    from diffusers import MiniMaxH3Transformer3DModel

    from tools.validation.models.h3_full_model import packed_inputs

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = MiniMaxH3Transformer3DModel.from_pretrained(
        str(Path(checkpoint) / "transformer"),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.eval().enable_gradient_checkpointing()
    for p in model.parameters():
        p.register_post_accumulate_grad_hook(lambda q: setattr(q, "grad", None))
    base = packed_inputs("cuda")
    gi = ("hidden_states", "audio_hidden_states", "encoder_hidden_states")

    def run(mode, spec=""):
        inputs = {
            k: (v.detach().clone().requires_grad_() if k in gi else v) for k, v in base.items()
        }
        provider = rt.H3TrainingProvider(model, mode, rt.AblationOverlay.parse(spec))
        with provider.step():
            video, audio = provider.forward(**inputs)
            (video.float().square().mean() + audio.float().square().mean()).backward()
        rep = provider.adapter.report()
        trans = provider.transition(
            inputs["hidden_states"].detach(),
            video.detach(),
            torch.tensor(0.75, device="cuda"),
            torch.tensor(0.5, device="cuda"),
            seed=3,
            step=1,
            sample_ids=[0],
            modality="video",
        )
        return dict(
            video=video.detach(),
            audio=audio.detach(),
            grad=inputs["hidden_states"].grad.detach(),
            report=rep,
            logp=trans.log_prob.detach(),
        )

    ref = run("off")
    variants = (
        ("off", "off", ""),
        ("strict", "strict", ""),
        ("strict (repeat)", "strict", ""),
        ("audit", "audit", ""),
        ("strict, node:attention native", "strict", "node:attention"),
        ("strict, boundary:token_refiner native", "strict", "boundary:token_refiner"),
    )
    runs, rows = {}, []
    for name, mode, spec in variants:
        r = run(mode, spec)
        runs[name] = r
        rows.append(
            dict(
                variant=name,
                label=r["report"]["label"],
                case_id=r["report"]["case_id"],
                bound=r["report"]["bound"],
                video_rel_diff_vs_provider=rel(r["video"], ref["video"]),
                audio_rel_diff_vs_provider=rel(r["audio"], ref["audio"]),
                input_grad_rel_diff_vs_provider=rel(r["grad"], ref["grad"]),
                logp_rel_diff_vs_provider=rel(r["logp"], ref["logp"]),
                boundary_bindings=r["report"]["boundary_bindings"],
            )
        )
    a, b = runs["strict"], runs["strict (repeat)"]
    repeat = all(torch.equal(a[k], b[k]) for k in ("video", "audio", "grad", "logp"))
    return rows, dict(strict_bitwise_repeatable=repeat)


def plot(report, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    lora = report["lora_gate"]
    xs = range(len(lora))
    w = 0.25
    for j, key in enumerate(
        ("grad_a_rel_err_vs_fp64", "grad_b_rel_err_vs_fp64", "grad_x_rel_err_vs_fp64")
    ):
        ax[0].bar(
            [i + (j - 1) * w for i in xs],
            [r[key] for r in lora],
            w,
            label=key.replace("_rel_err_vs_fp64", ""),
        )
    ax[0].axhline(5e-3, color="r", ls="--", lw=1, label="gate 5e-3")
    ax[0].set_xticks(list(xs), [r["shape"] for r in lora], fontsize=8)
    ax[0].set_yscale("log")
    ax[0].set_title("LoRA gradients vs FP64 (relative error)")
    ax[0].legend(fontsize=7)

    tra = report["transition"]
    ax[1].bar(range(len(tra)), [max(r["logp_max_rel_diff_vs_torch_mean"], 1e-12) for r in tra])
    ax[1].set_xticks(range(len(tra)), [r["shape"] for r in tra], fontsize=8)
    ax[1].set_yscale("log")
    ax[1].set_title("CPS log-prob vs torch.mean (max rel diff);\nprev_sample/mean bitwise equal")

    if "model" in report:
        rows = [r for r in report["model"] if r["variant"] != "off"]
        xs = range(len(rows))
        for j, key in enumerate(
            (
                "video_rel_diff_vs_provider",
                "audio_rel_diff_vs_provider",
                "input_grad_rel_diff_vs_provider",
            )
        ):
            ax[2].bar(
                [i + (j - 1) * w for i in xs],
                [max(r[key], 1e-12) for r in rows],
                w,
                label=key.replace("_rel_diff_vs_provider", ""),
            )
        ax[2].set_xticks(
            list(xs),
            [r["variant"].replace(", ", "\n").replace(" native", "\nnative") for r in rows],
            fontsize=6,
        )
        ax[2].set_yscale("log")
        ax[2].legend(fontsize=7)
        ax[2].set_title("33B model, 12-token request: difference to provider")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--png", type=Path)
    p.add_argument("--checkpoint")
    p.add_argument("--verl-omni")
    args = p.parse_args()
    modes = {
        m: vars(rt.h3_runtime_policy(m)) | {"mode": m} for m in ("strict", "audit", "auto", "off")
    }
    report = dict(
        device=torch.cuda.get_device_name(0),
        torch=torch.__version__,
        runtime_modes={k: {kk: str(vv) for kk, vv in v.items()} for k, v in modes.items()},
        lora_gate=lora_rows(),
        transition=transition_rows(args.verl_omni),
    )
    if args.checkpoint:
        report["model"], report["model_checks"] = model_rows(args.checkpoint)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str))
    if args.png:
        plot(report, args.png)
    print(
        json.dumps({k: v for k, v in report.items() if k != "model"}, indent=1, default=str)[:3000]
    )
    if "model" in report:
        for r in report["model"]:
            print(r)
        print(report["model_checks"])


if __name__ == "__main__":
    main()
