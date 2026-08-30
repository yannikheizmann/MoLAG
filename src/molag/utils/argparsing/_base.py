"""Define strict Pydantic models and shared argument-model operations."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict


class _StrictModel(BaseModel):
    """Common validation policy for public argument models."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )


class AdditionalArgsBase(_StrictModel):
    """Argument group passed as ``--group key=value ...``."""


class PydanticArgsBase(_StrictModel):
    """Complete, validated argument model for a command."""

    def flattened(self) -> dict[str, Any]:
        """Return top-level and one-level nested values as one dictionary.

        This supports dependency injection into constructors while rejecting
        ambiguous duplicate field names.
        """

        flattened: dict[str, Any] = {}
        for key, value in self:
            nested = value.model_dump() if isinstance(value, BaseModel) else None
            items = nested.items() if nested is not None else ((key, value),)
            for nested_key, nested_value in items:
                if nested_key in flattened:
                    raise ValueError(
                        f"Cannot flatten duplicate argument name {nested_key!r}."
                    )
                flattened[nested_key] = nested_value
        return flattened

    def call(
        self,
        callable_: Callable[..., Any] | type,
        **overrides: Any,
    ) -> Any:
        """Call a function or class with matching fields from this model."""

        target = callable_.__init__ if inspect.isclass(callable_) else callable_
        parameters = inspect.signature(target).parameters
        values = self.flattened()
        values.update(overrides)
        matching = {key: value for key, value in values.items() if key in parameters}
        return callable_(**matching)

    def save(self, path: str | Path, *, format: str = "json") -> Path:
        """Save the fully resolved arguments to a JSON or YAML file."""

        output = Path(path)
        if output.suffix:
            destination = output
        else:
            destination = output / f"config.{format}"
        destination.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            destination.write_text(self.model_dump_json(indent=2) + "\n")
        elif format in {"yaml", "yml"}:
            destination.write_text(
                yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)
            )
        else:
            raise ValueError("format must be 'json', 'yaml', or 'yml'")
        return destination
