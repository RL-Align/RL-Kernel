# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Persistent training worker for colocated dual-engine.

Launched ONCE by torchrun. Stays alive across steps. Communicates with the
orchestrator via file-based signals:

  Signal files (in work_dir):
    "train_step_N.signal"  → orchestrator tells worker to train on step N
    "train_done_N.signal"  → worker tells orchestrator step N is done
    "shutdown.signal"      → orchestrator tells worker to exit

  Data files:
    "completions_stepN.json"  → rollout completions (written by orchestrator)
    "latest_weights/"         → LoRA adapter (written by worker)
    "train_metrics_stepN.json"→ loss + timing (written by worker)

Between training steps, the worker offloads model to CPU and releases GPU
memory so vLLM can use the GPUs for inference.

Not intended to be run directly — use colocated_dual_engine.py instead.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.distributed as dist


def wait_for_file(path, poll_interval=0.1, timeout=600):
    """Poll until a file appears, then return."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path):
            return True
        time.sleep(poll_interval)
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--max-steps", type=int, required=True)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--max-len", type=int, default=320)
    args = p.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    # === One-time initialization ===
    import deepspeed
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model.enable_input_require_grads()

    if rank == 0:
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Worker] Initialized: {n:,} LoRA params", flush=True)

    optimizer = torch.optim.AdamW(
        [pp for pp in model.parameters() if pp.requires_grad], lr=args.lr, weight_decay=0.01
    )
    ds_config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {"stage": 2},
        "bf16": {"enabled": True},
    }
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model, optimizer=optimizer, config=ds_config, dist_init_required=False
    )

    weights_dir = os.path.join(args.work_dir, "latest_weights")

    # Signal orchestrator that initialization is done
    if rank == 0:
        os.makedirs(weights_dir, exist_ok=True)
        with open(os.path.join(args.work_dir, "worker_ready.signal"), "w") as f:
            f.write("ready")
        print("[Worker] Ready, waiting for commands...", flush=True)
    dist.barrier()

    # === Pre-allocate pinned CPU buffer (created AFTER offload, no GPU impact) ===
    pinned_state = {}

    # === Offload to CPU initially (vLLM needs GPUs first) ===
    engine.module.to("cpu")
    torch.cuda.empty_cache()
    dist.barrier()
    time.sleep(3)

    # Now safe to create pinned buffers (model is on CPU, no GPU allocation)
    for name, param in engine.module.named_parameters():
        pinned_state[name] = param.data.clone().pin_memory()

    # === Persistent loop ===
    for step in range(args.max_steps):
        signal_path = os.path.join(args.work_dir, f"train_step_{step}.signal")
        shutdown_path = os.path.join(args.work_dir, "shutdown.signal")

        # Wait for train signal or shutdown
        while True:
            if os.path.exists(shutdown_path):
                if rank == 0:
                    print("[Worker] Shutdown signal received", flush=True)
                dist.barrier()
                dist.destroy_process_group()
                return
            if os.path.exists(signal_path):
                break
            time.sleep(0.1)

        # --- Move model to GPU (from pinned memory, fast) ---
        t_load = time.time()
        for name, param in engine.module.named_parameters():
            param.data = pinned_state[name].to(device, non_blocking=True)
        torch.cuda.synchronize()
        load_ms = (time.time() - t_load) * 1000

        # --- Load completions ---
        completions_path = os.path.join(args.work_dir, f"completions_step{step}.json")
        with open(completions_path) as f:
            completions = json.load(f)

        # --- Train ---
        t_train = time.time()
        full_texts = [c["prompt"] + c["completion"] for c in completions]
        enc = tokenizer(
            full_texts, truncation=True, max_length=args.max_len, padding=True, return_tensors="pt"
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        engine.train()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = engine(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        engine.backward(outputs.loss)
        engine.step()
        train_ms = (time.time() - t_train) * 1000
        loss_val = outputs.loss.item()

        # --- Save weights (rank 0) ---
        if rank == 0:
            engine.module.save_pretrained(weights_dir)
            tokenizer.save_pretrained(weights_dir)
            metrics_path = os.path.join(args.work_dir, f"train_metrics_step{step}.json")
            with open(metrics_path, "w") as f:
                json.dump(
                    {"loss": loss_val, "train_ms": round(train_ms, 1), "load_ms": round(load_ms, 1)},
                    f,
                )

        dist.barrier()

        # --- Offload to pinned CPU, free GPU memory ---
        t_offload = time.time()
        for name, param in engine.module.named_parameters():
            pinned_state[name].copy_(param.data, non_blocking=True)
        torch.cuda.synchronize()
        engine.module.to("cpu")
        # Restore pinned data as param.data for next load
        for name, param in engine.module.named_parameters():
            param.data = pinned_state[name]
        torch.cuda.empty_cache()
        offload_ms = (time.time() - t_offload) * 1000

        if rank == 0:
            print(
                f"[Worker] Step {step}: loss={loss_val:.4f} "
                f"load={load_ms:.0f}ms train={train_ms:.0f}ms offload={offload_ms:.0f}ms",
                flush=True,
            )
            # Signal done
            done_path = os.path.join(args.work_dir, f"train_done_{step}.signal")
            with open(done_path, "w") as f:
                f.write(f"done step={step} loss={loss_val}")

    # Normal completion
    if rank == 0:
        print("[Worker] All steps complete", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
