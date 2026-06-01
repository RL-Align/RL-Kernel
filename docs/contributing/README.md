# Developer Guide

This section collects general contribution material, design documents, and operator
development notes for RL-Kernel.

Before merging a new operator, include:

- The implementation and dispatch registration.
- A focused correctness test or documented validation path.
- A dedicated page under `docs/operators/`.
- Navigation updates in `docs/.nav.yml`.
- A passing documentation build with `mkdocs build --strict -f mkdocs.yaml`.

Useful pages:

- [Documentation Guide](documentation.md)
- [Testing](testing.md)
- [Runtime Dispatch](../design/runtime-dispatch.md)
