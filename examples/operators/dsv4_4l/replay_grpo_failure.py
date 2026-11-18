# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 -- mirror the formal entry's native import/visibility bootstrap.
"""Diagnostic only: replay recorded native PPO batches and verify every parameter SHA.

This does not run AgentLoop/reward and cannot qualify the full GRPO acceptance.
It reconstructs a failed policy for numerical localization, using real MCore
backward/optimizer and live vLLM workers, without fabricating weights or scores.
"""
import json
import os
import sys
from pathlib import Path

from verl.experimental.rl_kernel.runtime import prepare_worker

prepare_worker()
from verl.experimental.rl_kernel import trainer_engine  # noqa: F401

import ray
import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.experimental.rl_kernel.trainer import RLKActorWorker, RLKVerlTrainer
from verl.single_controller.base.decorator import Dispatch, register
from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.utils import Role
from verl.utils import tensordict_utils as tu
from verl.utils.device import auto_set_device


class ReplayActorWorker(RLKActorWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def replay_recorded_update(self, source, version):
        from rl_engine.integrations.engines.rollout.vllm.checkpoint_sync import digest

        source = Path(source)
        batch = torch.load(
            source / f"parity-step{version}-raw.pt", map_location="cpu", weights_only=True
        )
        recorded = json.loads((source / f"training-batch-step{version}.json").read_text())
        expected = json.loads(
            (source / f"actor-rank-{dist.get_rank()}-v{version}.json").read_text()
        )
        engine = self.actor.engine
        parameters = {
            n: p for m in engine._rlk_models for n, p in m.named_parameters() if p.requires_grad
        }
        before = {n: digest(p) for n, p in parameters.items()}
        if before != expected["parameter_before"]:
            raise RuntimeError("replayed policy does not equal the original pre-update bytes")
        full = []
        prompts = []
        responses = []
        masks = []
        old = []
        advantages = []
        response_masks = []
        for i, row in enumerate(batch["prompts"]):
            # TQ pads on the right with 0. Token 0 at the start is the real BOS.
            # These captured chat prompts end in a nonzero assistant prefix.
            lengths = batch.get("prompt_lengths")
            length = int(lengths[i]) if lengths is not None else int(torch.nonzero(row)[-1]) + 1
            prompt = row[:length].clone()
            response = batch["responses"][i][batch["response_mask"][i].bool()].clone()
            assert prompt[-1] != 0 and response.numel() > 0
            prompts.append(prompt)
            responses.append(response)
            full.append(torch.cat((prompt, response)))
            masks.append(torch.cat((torch.zeros_like(prompt), torch.ones_like(response))))
            response_masks.append(torch.ones_like(response))
            old.append(batch["old_log_probs"][i][batch["response_mask"][i].bool()].clone())
            advantages.append(
                torch.tensor(
                    recorded["summaries"]["advantages"]["values"][i][: len(response)],
                    dtype=torch.float32,
                )
            )

        def nested(rows):
            return torch.nested.as_nested_tensor(rows, layout=torch.jagged)

        data = TensorDict(
            dict(
                input_ids=nested(full),
                loss_mask=nested(masks),
                prompts=nested(prompts),
                responses=nested(responses),
                response_mask=nested(response_masks),
                old_log_probs=nested(old),
                advantages=nested(advantages),
                temperature=torch.ones(len(full)),
            ),
            batch_size=[len(full)],
        )
        tu.assign_non_tensor(
            data,
            global_token_num=[len(x) for x in full],
            global_batch_size=len(full),
            micro_batch_size_per_gpu=1,
            use_dynamic_bsz=False,
            use_remove_padding=False,
            use_fused_kernels=False,
            disable_auto_offload=True,
            update_lr_scheduler=True,
            calculate_entropy=False,
        )
        self.actor.train_batch(data)
        after = {n: digest(p) for n, p in parameters.items()}
        missing = [n for n in after if after[n] != expected["parameter_after"][n]]
        root = Path(self.config.actor.megatron.rlk_evidence_dir)
        (root / f"diagnostic-replay-rank{dist.get_rank()}-v{version}.json").write_text(
            json.dumps(
                dict(
                    diagnostic_only=True,
                    weight_version=engine.rlk_weight_version,
                    exact_original_parameters=not missing,
                    mismatched_parameters=missing,
                    original_source=str(source),
                    actual_parameter_hashes=after,
                ),
                indent=2,
            )
        )
        if missing:
            raise RuntimeError(
                f"replayed PPO update differs from original parameter bytes: {missing}"
            )
        return dict(version=version, exact_original_parameters=True)


class ReplayTrainer(RLKVerlTrainer):
    def _init_resource_pool_mgr(self):
        super()._init_resource_pool_mgr()
        self.role_worker_mapping[Role.Actor] = ray.remote(ReplayActorWorker)

    def on_init_end(self):
        self.checkpoint_manager.update_weights(0)


@ray.remote
class ReplayTaskRunner:
    def run(self, config):
        import transfer_queue as tq

        config.transfer_queue.enable = True
        OmegaConf.resolve(config)
        tq.init(config.transfer_queue)
        root = Path(config.rlk.sync_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / "DIAGNOSTIC_ONLY.json").write_text(
            json.dumps(
                dict(
                    scope=(
                        "Recorded native PPO replay for localization; "
                        "no AgentLoop/reward acceptance"
                    ),
                    source=os.environ["RLK_REPLAY_SOURCE"],
                )
            )
        )
        trainer = None
        try:
            trainer = ReplayTrainer(config)
            trainer.init()
            source = Path(os.environ["RLK_REPLAY_SOURCE"])
            for version in (1, 2, 3):
                trainer.actor_rollout_wg.replay_recorded_update(str(source), version)
                trainer.global_steps = version
                trainer.checkpoint_manager.update_weights(version)
            trainer._rlk_fixed_cases = json.loads((source / "fixed-sequences.json").read_text())
            from verl.experimental.rl_kernel.acceptance import audit_version

            audit_version(trainer)
        finally:
            if trainer is not None:
                trainer.close_dataloaders()
            tq.close()


def main():
    import verl

    config_path = Path(verl.__file__).parent / "experimental/rl_kernel/config"
    with initialize_config_dir(config_dir=str(config_path), version_base=None):
        config = compose(
            config_name="dsv4_grpo",
            overrides=[a for a in sys.argv[1:] if not a.startswith("hydra.")],
        )
    auto_set_device(config)
    # Ray must import these classes by module name, not cloudpickle __main__ locks.
    from examples.operators.dsv4_4l.replay_grpo_failure import ReplayTaskRunner as Runner

    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    run_ppo(config, task_runner_class=Runner)


if __name__ == "__main__":
    main()
