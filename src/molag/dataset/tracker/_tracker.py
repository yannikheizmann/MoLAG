"""Bind the triangular tracker code and geometry implementations."""

from __future__ import annotations

from ._base import TrackerBase
from ._code import TriangularTrackerCode
from ._geometry import TriangularTrackerGeometry


class TriangularTracker(TrackerBase):
    """Coded seven-LED triangular tracker at a rigid pose."""

    CodeClass = TriangularTrackerCode
    GeometryClass = TriangularTrackerGeometry
