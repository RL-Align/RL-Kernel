# Documentation Guide

RL-Kernel documentation is built with MkDocs Material and published as a static GitHub
Pages site.

## Local Build

```bash
pip install -r requirements-docs.txt
mkdocs build --strict -f mkdocs.yaml
```

## Local Preview

```bash
mkdocs serve
```

## Adding an Operator Page

1. Copy `docs/contributing/operator-doc-template.md` to `docs/operators/<operator-name>.md`.
2. Fill in the operator contract, backend list, examples, tests, and limitations.
3. Add the page to the `Operators` section in `docs/.nav.yml`.
4. Run `mkdocs build --strict -f mkdocs.yaml`.

## CI Requirement

Documentation changes must keep `mkdocs build --strict -f mkdocs.yaml` passing. The CI
docs job validates navigation, links, and Markdown extension configuration.
