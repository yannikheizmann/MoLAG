# Calibrated inference

An evaluated run directory contains the model configuration, weights, and calibrated
threshold required by the high-level predictor.

```python
import numpy as np

from molag.inference import MoLAGPredictor

predictor = MoLAGPredictor.from_run_directory("results")

coordinates = np.array(
    [[742.0, 501.0], [811.0, 518.0], [776.0, 566.0]],
    dtype=np.float32,
)
result = predictor.predict(coordinates)

print(result.affinities)
print(result.groups)
```

Models uploaded through the Transformers `push_to_hub` options can be loaded using
`MoLAGPredictor.from_hub("organisation/model-name")`. The predictor returns the
affinity matrix and connected components as arrays of input-point indices.
