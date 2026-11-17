# Miles orchestration

RL-Kernel installs into Miles' Megatron workers through Miles' own plugin hook,
`custom_megatron_init_path`. Miles loads the dotted path in every worker after Megatron
initialization and before model construction, and merges the `config` mapping into the
callback arguments after validating it against the fields the callback declares.

## Launcher configuration

```yaml
custom_megatron_init_path:
  path: rl_engine.integrations.orchestrators.miles.install.custom_megatron_init
  config:
    rl_kernel_miles_enabled: "1"
    rl_kernel_attention_case: P/P
    rl_kernel_ffn_case: P/P
    rl_kernel_logp_case: R/P
    rl_kernel_readback_dir: /path/to/readback
```

All five fields must be present: Miles rejects a mapping that omits one, because the
callback declares exactly these fields. A field left as `null` falls back to its
`RL_KERNEL_*` environment variable, which is how a local run is debugged without a
launcher.

Do not depend on the driver's shell environment reaching Ray workers. The config mapping
travels in the orchestration arguments, so every worker receives the same values by
construction.

## Auditing a run

Each worker writes `<framework>-<target>-<pid>.json` into `rl_kernel_readback_dir` when it
installs. After the run:

```bash
python -m rl_engine.integrations.orchestrators.miles.readback /path/to/readback \
    --expect-workers <number of training workers>
```

The command exits 0 only when every expected worker left a readback file, the workers
agree on one plan, no fallback was recorded, and every RL-Kernel route selected by the
plan has an installed hook. Add `--require-execution` once training has run to also
demand an execution record per RL-Kernel route (an RL-Kernel backend, a positive call
count). Add `--json` for a machine-readable report.

The readback file is a per-process snapshot: `call_count` is written when a route first
executes and when its route fingerprint changes, plus once at process exit. A worker that
was killed before exiting therefore reports fewer calls than it performed; use the file to
establish which route ran, not how many times it ran in total.
