# Installation

RL-Kernel requires Python 3.10 or newer and PyTorch. CUDA builds require a working
CUDA toolchain; ROCm builds require a compatible ROCm environment.

## From Source

```bash
git clone https://github.com/Flink-ddd/RL-Kernel.git
cd RL-Kernel
pip install -e .
```

## Optional Backends

```bash
pip install -e ".[cuda]"
```

```bash
pip install -e ".[rocm]"
```

## Development Dependencies

```bash
pip install -e ".[dev]"
pip install -r requirements-docs.txt
```

## Documentation Preview

```bash
mkdocs serve
```

Then open the local URL printed by MkDocs.
