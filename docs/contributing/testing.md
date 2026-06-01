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

## Documentation Build

```bash
pip install -r requirements-docs.txt
mkdocs build --strict -f mkdocs.yaml
```

Run the documentation build whenever adding a new operator page or changing navigation.
