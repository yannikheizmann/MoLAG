# Contributing to MoLAG

Contributions that improve correctness, reproducibility, documentation, or support for
additional tracker architectures are welcome. Please discuss substantial changes in an
issue before investing in an implementation.

## Development setup

MoLAG requires Python 3.11 or newer and uses
[uv](https://docs.astral.sh/uv/) for its locked development environment.

```bash
git clone https://github.com/yannikheizmann/MoLAG.git
cd MoLAG
uv sync --frozen --extra dev --extra docs
```

Create a branch in your fork and keep each change focused. Do not commit credentials,
generated experiment outputs, model checkpoints, local caches, or private development
instructions.

## Implementing changes

- Preserve the typed Pydantic configuration hierarchy and its precedence of defaults,
  YAML values, and CLI overrides.
- Retain interfaces and registry-based extension points for trackers and metrics.
- Implement independent affinity-loss terms through `AffinityLossComponentBase`.
- Add tests for new behaviour and regression tests for corrected behaviour.
- Keep model, loss, dataset, calibration, and metric semantics compatible with stored
  artefacts unless the change explicitly introduces a documented format revision.
- Use British English in prose and documentation.
- Write concise module and public API docstrings. Use imperative summaries for
  operations and noun-phrase summaries for classes.
- Avoid comments that restate the code or describe development history.

The [project documentation](https://yannikheizmann.github.io/MoLAG/) describes the
architecture and principal extension interfaces.

## Validation

Run the complete local checks before opening a pull request:

```bash
uv run --frozen ruff check .
uv run --frozen pytest -q --cov=molag --cov-branch --cov-fail-under=85
uv run --frozen mkdocs build --strict
uv build
```

Changes to numerical objectives, graph construction, dataset generation, calibration,
or evaluation require deterministic regression coverage. Report any expected metric or
artefact changes in the pull request.

## Pull requests

A pull request should:

- explain the problem and the chosen solution;
- identify affected commands, configurations, or artefacts;
- include appropriate tests and documentation;
- pass the continuous-integration checks; and
- avoid unrelated formatting or refactoring changes.

By contributing, you agree that your contribution is licensed under the repository's
[Apache License 2.0](LICENSE).
