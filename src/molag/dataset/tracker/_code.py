"""Encode the triangular tracker's selectable LED positions."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from ._base import TrackerCodeBase

CodeDigit = Literal[0, 1, 2]


class TriangularTrackerCode(BaseModel, TrackerCodeBase):
    """Three base-3 digits specifying the side-LED arrangement."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    c0: CodeDigit
    c1: CodeDigit
    c2: CodeDigit

    @classmethod
    def from_id(cls, id: int) -> TriangularTrackerCode:
        """Decode a zero-based identifier into three base-3 digits."""
        if isinstance(id, bool) or not isinstance(id, int):
            raise TypeError("id must be an integer")
        if not 0 <= id < cls.num_unique_ids():
            raise ValueError("id must lie in [0, 26]")
        return cls(
            c0=id // 9,
            c1=(id // 3) % 3,
            c2=id % 3,
        )

    def to_id(self) -> int:
        """Encode the three base-3 digits as a zero-based identifier."""
        return self.c0 * 9 + self.c1 * 3 + self.c2

    @classmethod
    def num_unique_ids(cls) -> int:
        """Return the number of representable triangular tracker codes."""
        return 27

    def as_tuple(self) -> tuple[int, int, int]:
        """Return the three code digits in side order."""
        return self.c0, self.c1, self.c2

    def __getitem__(self, index: int) -> int:
        if index not in {0, 1, 2}:
            raise IndexError("tracker code index must be 0, 1, or 2")
        return self.as_tuple()[index]
