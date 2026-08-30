# MoLAG

MoLAG performs monocular LED assignment for scenes containing multiple coded rigid
trackers. It predicts a same-tracker affinity for every unordered pair of localised
LED points and returns connected components of the thresholded affinity graph as
candidate tracker groups.

This repository contains the model, synthetic-data generator, structured affinity
loss, finetuning workflow, held-out threshold calibration, evaluation, and calibrated
inference API accompanying *MoLAG: Monocular LED Assignment via a Graph Neural Network
for Multi-Tracker Surgical Navigation*.

## Installation

MoLAG requires Python 3.11 or newer. The locked development environment can be created
with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/yannikheizmann/MoLAG.git
cd MoLAG
uv sync --frozen
```

The default finetuning profile uses bfloat16 arithmetic and is intended for a
CUDA-capable system. Device and precision settings can be overridden through the CLI
or a copied experiment profile.

## Reproducing the workflow

The supplied profiles record the model, training, held-out dataset, calibration, and
evaluation settings. Run commands from the repository root.

Generate the frozen calibration and test datasets:

```bash
uv run --frozen generate_eval_dataset \
  --config experiments/molag_calibration_dataset.yaml

uv run --frozen generate_eval_dataset \
  --config experiments/molag_test_dataset.yaml
```

Finetune MoLAG:

```bash
uv run --frozen finetune \
  --config experiments/molag_finetune.yaml
```

Calibrate the affinity threshold and evaluate the resulting partitioning in one
workflow:

```bash
uv run --frozen evaluate \
  --config experiments/molag_evaluate.yaml
```

The evaluation command loads the model once, selects the threshold on the frozen
calibration set, and then evaluates the frozen test set. A run directory contains the
resolved configuration, model weights, calibration grid, and evaluation result:

```text
results/
├── config.yaml
├── model.safetensors
├── calibration_predictions.npz
├── calibration.json
├── predictions.npz
├── samples.csv
├── tracker_samples.csv
└── evaluation.json
```

Calibration records every configured metric at every threshold. The configured
`objective` selects the operating threshold; the default is strict scene-level
`partition_accuracy`, with ties resolved in favor of the higher threshold.

Raw predictions are stored independently of the metrics. New or changed metrics can
therefore be computed without rerunning the model or modifying the original result:

```bash
uv run --frozen recompute \
  --config experiments/molag_evaluate.yaml \
  --evaluation_args metrics='["Affinity", "RealAffinity", "Partition"]'
```

Recomputed artifacts are written to `results/recomputed` by default. Passing an
explicit `threshold` skips recalibration; otherwise the threshold is selected again
from `calibration_predictions.npz` using the configured metric and threshold grid.

### Configuration precedence

Every command parses the same typed root configuration. Values are resolved in this
order:

1. Pydantic model defaults;
2. values in the YAML passed through `--config`;
3. explicit CLI values.

Nested CLI overrides use grouped `key=value` arguments:

```bash
uv run finetune \
  --config experiments/molag_finetune.yaml \
  --training_args output_dir=results/rerun learning_rate=0.0005
```

Unknown YAML fields are rejected with naming suggestions, and the terminal reports
which values came from YAML and which came from the CLI.

## Calibrated inference

After evaluation has produced `calibration.json`, inference needs only the run
directory and localised two-dimensional point coordinates:

```python
import numpy as np

from molag.inference import MoLAGPredictor

predictor = MoLAGPredictor.from_run_directory(
    "results",
)

coordinates = np.array(
    [[742.0, 501.0], [811.0, 518.0], [776.0, 566.0]],
    dtype=np.float32,
)
result = predictor.predict(coordinates)

print(result.affinities)
print(result.groups)
```

Models uploaded with the training `push_to_hub` options can be loaded with the
same interface:

```python
predictor = MoLAGPredictor.from_hub("organisation/model-name")
```

The predictor performs the same translation and scale normalization used during
training, builds the complete geometric graph, applies the run's calibrated threshold,
and returns the connected components as arrays of input-point indices.

## Extending MoLAG

The default configuration retains the interfaces used to extend the method:

- `TrackerBase`, `TrackerCodeBase`, and `TrackerGeometryBase` define new tracker
  architectures and use the tracker registry.
- `AffinityLossComponentBase` supports alternative structured loss terms and contexts.
- `MetricsBase` registers streaming calibration and evaluation metrics.
- `ModelArgs` and `LossArgs` expose architecture widths and loss hyperparameters for
  further experiments.

## Scope

The included evaluation uses generated coordinate-level scenes. It does not establish
performance under camera distortion, detector-specific localisation errors, motion,
or other effects absent from the synthetic distribution. Candidate groups are intended
for subsequent geometry-specific marker assignment and pose estimation, which are not
implemented here.

## Tests

```bash
uv run --frozen --extra dev pytest
```

## Citation

Citation metadata will be added with the archival paper record.
