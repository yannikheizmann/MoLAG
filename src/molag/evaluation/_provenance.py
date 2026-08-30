from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import subprocess
from typing import Any

from torch import nn

from ._loader import ModelLoader


@dataclass(frozen=True)
class FileFingerprint:
    """Identity of one file used or produced by evaluation."""

    path: str
    sha256: str

    @classmethod
    def from_path(cls, path: str | Path) -> FileFingerprint:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"provenance file not found: {source}")
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return cls(path=str(source), sha256=digest.hexdigest())


@dataclass(frozen=True)
class EvaluationProvenance:
    """Reproducibility metadata for one evaluation result."""

    created_at: str
    source_revision: str | None
    source_dirty: bool | None
    python_version: str
    package_versions: dict[str, str]
    model_parameters: int
    trainable_model_parameters: int
    files: dict[str, FileFingerprint]

    @classmethod
    def collect(
        cls,
        run_directory: str | Path,
        dataset: str | Path,
        predictions: str | Path,
        model: nn.Module,
        calibration: str | Path | None = None,
        model_directory: str | Path | None = None,
    ) -> EvaluationProvenance:
        """Collect source, environment, model, and file identities."""
        run_path = Path(run_directory)
        model_path = (
            Path(model_directory) if model_directory is not None else run_path
        )
        files = {
            "configuration": FileFingerprint.from_path(model_path / "config.yaml"),
            "checkpoint": FileFingerprint.from_path(
                ModelLoader.find_checkpoint(model_path)
            ),
            "dataset": FileFingerprint.from_path(dataset),
            "predictions": FileFingerprint.from_path(predictions),
        }
        if calibration is not None:
            files["calibration"] = FileFingerprint.from_path(calibration)
        revision, dirty = cls._source_state()
        parameters = tuple(model.parameters())
        return cls(
            created_at=datetime.now(timezone.utc).isoformat(),
            source_revision=revision,
            source_dirty=dirty,
            python_version=platform.python_version(),
            package_versions=cls._package_versions(),
            model_parameters=sum(parameter.numel() for parameter in parameters),
            trainable_model_parameters=sum(
                parameter.numel() for parameter in parameters if parameter.requires_grad
            ),
            files=files,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def _package_versions() -> dict[str, str]:
        result = {}
        for package in (
            "molag",
            "numpy",
            "pydantic",
            "safetensors",
            "torch",
            "torch-geometric",
            "transformers",
        ):
            try:
                result[package] = version(package)
            except PackageNotFoundError:
                result[package] = "not-installed"
        return result

    @staticmethod
    def _source_state() -> tuple[str | None, bool | None]:
        repository = Path(__file__).resolve().parents[3]
        try:
            revision = subprocess.run(
                ["git", "-C", str(repository), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            status = subprocess.run(
                ["git", "-C", str(repository), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None, None
        return revision, bool(status.strip())
