# Cross-Configuration Drift Report

The cross-configuration runner writes a sealed, append-only attempt directory.
The report tool reads that directory after `COMPLETE` has been published and
renders the recorded comparison rather than re-running a model. It shows actual
operator provenance, requested TP/CP and dtype axes, the fixed comparison
threshold, and the worst active selected-token delta.

Generate the offline desktop bundle from one completed attempt:

```bash
python -m rl_engine.validation.cross_config report \
  runs/<experiment>/cases/<case>/attempt-0001 \
  --output /tmp/qwen3-tp2-cp2.rlk-drift
```

The `.rlk-drift` file contains sanitized report JSON, a Chrome Trace Event JSON
trace, and a PNG preview. It excludes checkpoints, prompts, and raw score
tensors. Install the optional local viewer and open it without a browser:

```bash
python -m pip install "rl-engine[drift-viewer]"
rlk-drift-view /tmp/qwen3-tp2-cp2.rlk-drift
```

The viewer has an expandable track tree, horizontal zoom/scroll, selectable
events, and an event-details panel. The horizontal scale is explicitly marked
as a sample ordinal unless the source artifacts contain real timestamps.

Other output suffixes expose the same sealed evidence in a form suitable for a
specific workflow:

| Suffix | Artifact | Intended use |
| --- | --- | --- |
| `.rlk-drift` | Offline desktop bundle | Interactive post-training triage |
| `.png`, `.jpg` | Static summary | Pull request or issue attachment |
| `.json` | Chrome Trace Event | Perfetto or trace-viewer inspection |

An unsealed or malformed attempt is rejected. A failed identity gate stays
`not comparable`; the tool does not turn it into a numerical drift value.
