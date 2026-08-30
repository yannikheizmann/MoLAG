# Output artifacts

A completed run keeps model, calibration, evaluation, and provenance information in
one directory.

```text
results/
├── config.yaml
├── dataset_profile.yaml
├── model.safetensors
├── calibration_predictions.npz
├── calibration.json
├── predictions.npz
├── samples.csv
├── tracker_samples.csv
└── evaluation.json
```

`calibration.json` records all objective values across the threshold grid as well as
the selected operating threshold. `predictions.npz` stores model outputs separately
from metric results. `evaluation.json` contains aggregate results, breakdowns, and
provenance hashes linking the model, dataset, prediction cache, and calibration
artifact.

By default, recomputed metrics are written below `results/recomputed`, preserving the
original evaluation.
