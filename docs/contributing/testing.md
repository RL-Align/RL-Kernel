# Testing

RL-Kernel uses focused tests for dispatch behavior and operator accuracy.

## Dispatch Tests

```bash
python -m pytest rl_engine/tests/test_dispatch.py -v
```

## Operator Accuracy

```bash
python tests/test_op_accuracy.py
```

## Teacher-Forced Logprob Checks

Consistency benchmarks must score a fixed token sequence. Rollout sampling
parameters such as temperature, top-p, or top-k choose the generated tokens, but
they must not be applied again when training or benchmark code computes selected
log-probs for that sequence.

Use `teacher_forced_logprobs_reference` for reference scoring:

```python
from rl_engine.testing import teacher_forced_logprobs_reference

logps = teacher_forced_logprobs_reference(logits, token_ids, mask=completion_mask)
```

Focused checks:

```bash
python -m pytest tests/test_reference_ops.py tests/test_training_contract.py -v
```

## Documentation Build

```bash
pip install -r requirements-docs.txt
mkdocs build --strict -f mkdocs.yaml
```

Run the documentation build whenever adding a new operator page or changing navigation.
