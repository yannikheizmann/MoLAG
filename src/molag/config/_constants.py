"""Static configuration shared by MoLAG components."""

import math
from pathlib import Path

import numpy as np

CAMERA_WIDTH_PIXELS = 1440
CAMERA_HEIGHT_PIXELS = 1080
CAMERA_HORIZONTAL_FIELD_OF_VIEW_DEG = 90.0
MAX_SCENE_GENERATION_ATTEMPTS = 10_000
CALIBRATION_RESULT_FILENAME = "calibration.json"
CALIBRATION_PREDICTION_CACHE_FILENAME = "calibration_predictions.npz"
EVALUATION_RESULT_FILENAME = "evaluation.json"
PREDICTION_CACHE_FILENAME = "predictions.npz"
DEFAULT_DATASET_PROFILE = (
    Path(__file__).parent.parent / "dataset" / "profiles" / "molag_standard.yaml"
)

TRIANGULAR_TRACKER_SIDE_LENGTH_MM = 64.0

TRIANGULAR_TRACKER_CANDIDATE_COORDINATES = np.array(
    [[0.25, 0.50, 0.75], [-0.02, 0.02, -0.02]],
    dtype=np.float64,
)

TRIANGULAR_TRACKER_BARYCENTRIC_TRANSFORM = np.array(
    [[1.0, -1.0 / math.sqrt(3.0)], [0.0, 2.0 / math.sqrt(3.0)]],
    dtype=np.float64,
)
