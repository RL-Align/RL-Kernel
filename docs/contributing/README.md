# Developer Guide

Start with the [Contributor Guide](contributor-guide.md) to choose where your PR's
implementation, configuration, tests, benchmarks, and documentation belong.
It explains the boundaries between models, operators, hardware backends, runtime
dispatch, distributed execution, and train/rollout integrations.

This section also contains architecture references, numerical contracts,
integration guides, validation procedures, and historical design documents.

Before merging a new operator, include:

- The implementation and dispatch registration.
- A focused correctness test or documented validation path.
- A dedicated page under `docs/operators/`.
- Navigation updates in `docs/.nav.yml`.
- A passing documentation build with `mkdocs build --strict -f mkdocs.yaml`.

Useful pages:

- [Contributor Guide](contributor-guide.md)
- [Repository Layout and Ownership](../architecture/repository-layout.md)
- [Documentation Guide](documentation.md)
- [Testing](testing.md)
- [Runtime Dispatch](../architecture/runtime-dispatch.md)
