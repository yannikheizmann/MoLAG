from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from numpy.typing import NDArray

from molag.utils.registry import RegistryMeta

if TYPE_CHECKING:
    from molag.dataset import PoseConfig

    from ._pose import TrackerPose

FloatArray = NDArray[np.float64]


class TrackerCodeBase(ABC):
    """Interface for a finite tracker-configuration code."""

    @classmethod
    @abstractmethod
    def from_id(cls, id: int) -> TrackerCodeBase:
        """Construct a code from its zero-based integer identifier."""

    @abstractmethod
    def to_id(self) -> int:
        """Return the code's zero-based integer identifier."""

    @classmethod
    @abstractmethod
    def num_unique_ids(cls) -> int:
        """Return the number of representable configurations."""


class TrackerGeometryBase(ABC):
    """Interface for LED positions in a tracker's local coordinate system."""

    @classmethod
    @abstractmethod
    def from_code(cls, code: TrackerCodeBase) -> TrackerGeometryBase:
        """Construct the LED geometry represented by a tracker code."""

    @classmethod
    @abstractmethod
    def num_leds(cls) -> int:
        """Return the number of LEDs in a complete constellation."""

    @property
    @abstractmethod
    def center(self) -> FloatArray:
        """Return the centre of the rigid tracker body."""

    @abstractmethod
    def as_array(self) -> FloatArray:
        """Return LED coordinates with shape ``(num_leds, 3)``."""


class TrackerBase(ABC, metaclass=RegistryMeta["TrackerBase"]):
    """Interface for a coded rigid tracker at a three-dimensional pose."""

    CodeClass: ClassVar[type[TrackerCodeBase]]
    GeometryClass: ClassVar[type[TrackerGeometryBase]]

    def __init__(
        self,
        code: TrackerCodeBase,
        pose: TrackerPose,
        geometry: TrackerGeometryBase,
    ) -> None:
        self._code = code
        self._pose = pose
        self._geometry = geometry

    @property
    def code(self) -> TrackerCodeBase:
        return self._code

    @property
    def pose(self) -> TrackerPose:
        return self._pose

    @property
    def geometry(self) -> TrackerGeometryBase:
        return self._geometry

    @classmethod
    def from_id(
        cls,
        id: int,
        rng: np.random.Generator,
        pose_cfg: PoseConfig | None = None,
    ) -> TrackerBase:
        from ._pose import TrackerPose

        code = cls.CodeClass.from_id(id)
        pose = TrackerPose.sample(rng, pose_cfg)
        geometry = cls.GeometryClass.from_code(code)
        return cls(code=code, pose=pose, geometry=geometry)

    @property
    def id(self) -> int:
        return self.code.to_id()

    def get_leds_world_coords(self) -> FloatArray:
        local = self.geometry.as_array() - self.geometry.center
        return (self.pose.R @ local.T).T + self.pose.t

    @classmethod
    def num_leds(cls) -> int:
        return cls.GeometryClass.num_leds()

    @classmethod
    def num_unique_ids(cls) -> int:
        return cls.CodeClass.num_unique_ids()
