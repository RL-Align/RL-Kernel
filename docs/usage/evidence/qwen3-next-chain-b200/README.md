# Qwen3-Next full-model chain (B200)

Measured at commit `c67ec14` (`measured-commit.txt`) from clean clones, one
stage after another on nodes with four B200 GPUs, in one environment
(`environment_sha256` `118384371a88…` in every `run.json`), on the official
`Qwen/Qwen3-Next-80B-A3B-Instruct` revision
`9c7f2fbe84465e40164a94cc16cd30b6999b0cc7`. Each stage was launched with the
launcher, `python tools/validation/models/qwen3_next_run.py --stage <stage>
--previous <run of the stage before> --runs-root <runs> --checkpoint <checkpoint>
--cuda-home <cuda-13.0> --python <venv>/bin/python` (see
`docs/usage/qwen3-next-chain.md`), and the reports were made with:

```bash
python tools/validation/models/qwen3_next_chain_report.py --last <runs>/<...-train-...> --out chain-report.json
bash ci/scripts/run_qwen3_next_chain_gate.sh chain-report.json
python tools/validation/models/qwen3_next_publish_evidence.py --last <runs>/<...-train-...> \
    --chain-report chain-report.json --out docs/usage/evidence/qwen3-next-chain-b200
```

| Directory | Stage | Result |
| --- | --- | --- |
| `preflight/` | operator gates, device guards, TP4 NCCL probe, checkpoint, rollout smoke | passed |
| `layers/` | per-layer TP4 contracts on real weights | passed |
| `l2/` | vLLM rollout vs VIME recompute, 16 cases, raw bits | passed, 16/16 |
| `train1/`, `train2/` | one RL update each, L2 on the updated weights | passed, 16/16 each |
| `train/` | checkpoint reload, L2 on the reloaded weights, continuous and resumed controls | passed, 16/16; reload == update 2 |
| `chain-report.json` | chain-gate report (`qwen3-next-chain-v1`, compatible with `ws1-c10-c11-v5`) | passed, 64 cells, `first_drift: null` |

Commands use the current repository layout; at `c67ec14` these runners were
in `scripts/` and `ci/`. The stages ran the code of `c67ec14`; the copy in this directory was made by
the publisher of `3025497`, which only adds redaction rules (`redaction.json`
counts the placeholders) and changes no measured byte. `SHA256SUMS` covers
every file except this README.

`train` was preempted and restarted several times. The training journal lets a
restarted run continue from its last durable step, and one restarted run reused
the final reload of an interrupted run; the chain report rejected it
(`Tensor artifact escaped its current attempt`). Before the run published here,
the journal's reload, continuous and resumed entries were retired, so this run
repeated all three itself; `train/reload.json` lists the retirements under
`recovery` (`repeat_final_reload_and_controls_in_fresh_train_run`). Updates 1
and 2 are the ones `train1/` and `train2/` recorded in the journal.
