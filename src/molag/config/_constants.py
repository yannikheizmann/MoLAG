"""Static configuration shared by MoLAG components."""

import math

import numpy as np

TRIANGULAR_TRACKER_SIDE_LENGTH_MM = 64.0

TRIANGULAR_TRACKER_CANDIDATE_COORDINATES = np.array(
    [[0.25, 0.50, 0.75], [-0.02, 0.02, -0.02]],
    dtype=np.float64,
)

TRIANGULAR_TRACKER_BARYCENTRIC_TRANSFORM = np.array(
    [[1.0, -1.0 / math.sqrt(3.0)], [0.0, 2.0 / math.sqrt(3.0)]],
    dtype=np.float64,
)
