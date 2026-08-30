# Reproduce the workflow

Run all commands from the repository root. The provided YAML files are complete,
version-controlled experiment specifications.

## 1. Generate held-out datasets

```bash
uv run --frozen generate_eval_dataset \
  --config experiments/molag_calibration_dataset.yaml

uv run --frozen generate_eval_dataset \
  --config experiments/molag_test_dataset.yaml
```

## 2. Finetune MoLAG

```bash
uv run --frozen finetune \
  --config experiments/molag_finetune.yaml
```

## 3. Calibrate and evaluate

```bash
uv run --frozen evaluate \
  --config experiments/molag_evaluate.yaml
```

The model is loaded once. The command first selects the affinity threshold on the
calibration set, then evaluates the frozen test set and stores raw predictions,
metrics, per-sample records, and provenance.

## Recompute metrics

```bash
uv run --frozen recompute \
  --config experiments/molag_evaluate.yaml \
  --evaluation_args metrics='["Affinity", "RealAffinity", "Partition"]'
```

Recomputation reads the saved prediction cache. It does not rerun model inference or
alter the original evaluation result.
