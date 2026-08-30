# Configuration

Every command parses the same typed root `Args` model. Configuration values are
resolved in this order:

1. Pydantic model defaults;
2. values in the YAML passed through `--config`;
3. explicit CLI values.

Nested CLI overrides use one grouped argument followed by `key=value` values:

```bash
uv run finetune \
  --config experiments/molag_finetune.yaml \
  --training_args output_dir=results/rerun learning_rate=0.0005
```

Unknown YAML keys are rejected with naming suggestions. The terminal reports which
values came from YAML and which were overridden by the CLI. Resolved training
configuration and the dataset profile are stored in the run directory.

The root groups are:

| Group | Responsibility |
| --- | --- |
| `dataset_args` | dataset profile and train/evaluation sizes |
| `model_args` | graph-network architecture |
| `loss_args` | modular affinity-loss configuration |
| `training_args` | Transformers training, logging, and Hub options |
| `evaluation_args` | datasets, metrics, calibration, and output paths |
| `eval_dataset_generation_args` | frozen evaluation-dataset generation |
