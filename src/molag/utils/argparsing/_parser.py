from __future__ import annotations

import argparse
import ast
import difflib
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any, get_type_hints

import yaml
from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from ._base import AdditionalArgsBase, PydanticArgsBase


class ConfigKeyError(ValueError):
    """Raised when a YAML configuration contains fields outside its schema."""


class ArgsParser:
    """Merge Pydantic defaults, YAML overrides, and explicit CLI overrides.

    Values are applied in increasing order of precedence:

    1. defaults declared by the Pydantic models;
    2. values loaded through ``--config``;
    3. values explicitly supplied on the command line.

    Nested command-line groups retain the project's established syntax::

        --training_args learning_rate=0.0001 num_train_epochs=5
    """

    def __init__(
        self,
        args_type: type[PydanticArgsBase],
        *,
        prog: str | None = None,
    ) -> None:
        self._args_type = args_type
        self._prog = prog

    def parse(self, argv: Sequence[str] | None = None) -> PydanticArgsBase:
        namespace = self._build_parser().parse_args(argv)
        explicit = vars(namespace)
        config_path = explicit.pop("config", None)

        yaml_values: dict[str, Any] = {}
        if config_path is not None:
            yaml_values = self._load_yaml(Path(config_path))
            unknown = self._unknown_keys(self._args_type, yaml_values)
            if unknown:
                raise ConfigKeyError(self._format_unknown_keys(unknown))

        cli_values = self._normalise_cli_values(explicit)
        if config_path is not None and "config" in self._args_type.model_fields:
            cli_values["config"] = config_path
        defaults = self._model_defaults(self._args_type)
        merged = self._deep_merge(defaults, yaml_values)
        merged = self._deep_merge(merged, cli_values)
        result = self._args_type.model_validate(merged)

        self._print_report(config_path, yaml_values, cli_values)
        return result

    @classmethod
    def _model_defaults(cls, model_type: type[BaseModel]) -> dict[str, Any]:
        """Collect declared defaults without requiring the model to validate.

        A reusable argument model may contain required fields supplied only by
        YAML or the CLI. Instantiating it before those layers are merged would
        therefore fail even though the final configuration is valid.
        """

        defaults: dict[str, Any] = {}
        hints = get_type_hints(model_type)
        for name, field in model_type.model_fields.items():
            if field.default_factory is not None:
                value = field.default_factory()
            elif field.default is not PydanticUndefined:
                value = field.default
            else:
                continue

            field_type = hints.get(name)
            if isinstance(value, BaseModel):
                defaults[name] = value.model_dump()
            elif (
                value is None
                and isinstance(field_type, type)
                and issubclass(field_type, BaseModel)
            ):
                defaults[name] = None
            else:
                defaults[name] = value
        return defaults

    def _build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(prog=self._prog)
        if "config" not in self._args_type.model_fields:
            parser.add_argument(
                "--config",
                type=Path,
                default=argparse.SUPPRESS,
                help="YAML file whose values override the Pydantic defaults.",
            )

        type_hints = get_type_hints(self._args_type)
        for field_name, field_type in type_hints.items():
            field = self._args_type.model_fields[field_name]
            option_strings = [f"--{field_name}"]
            if field.alias and field.alias != field_name:
                option_strings.append(f"-{field.alias}")

            kwargs: dict[str, Any] = {
                "default": argparse.SUPPRESS,
                "help": field.description,
            }
            if self._is_additional_args(field_type):
                kwargs.update(type=str, nargs="+")
            else:
                kwargs["type"] = str
                if self._is_enum(field_type):
                    kwargs["choices"] = [member.value for member in field_type]

            parser.add_argument(*option_strings, dest=field_name, **kwargs)
        return parser

    def _normalise_cli_values(self, values: Mapping[str, Any]) -> dict[str, Any]:
        normalised = dict(values)
        type_hints = get_type_hints(self._args_type)
        for field_name, value in tuple(normalised.items()):
            field_type = type_hints[field_name]
            if self._is_additional_args(field_type):
                normalised[field_name] = self._parse_key_value_group(value)
        return normalised

    @staticmethod
    def _parse_key_value_group(items: Sequence[str]) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        for item in items:
            if "=" not in item:
                raise argparse.ArgumentTypeError(
                    f"Expected key=value in nested argument group, got {item!r}."
                )
            key, raw_value = item.split("=", 1)
            if not key:
                raise argparse.ArgumentTypeError("Nested argument keys cannot be empty.")
            try:
                parsed[key] = ast.literal_eval(raw_value)
            except (SyntaxError, ValueError):
                parsed[key] = raw_value
        return parsed

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        with path.open(encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
        if loaded is None:
            return {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Configuration root must be a mapping: {path}")
        return loaded

    @classmethod
    def _unknown_keys(
        cls,
        model_type: type[BaseModel],
        values: Mapping[str, Any],
        prefix: str = "",
    ) -> list[tuple[str, Sequence[str]]]:
        unknown: list[tuple[str, Sequence[str]]] = []
        hints = get_type_hints(model_type)
        valid_names = tuple(model_type.model_fields)

        for key, value in values.items():
            path = f"{prefix}.{key}" if prefix else key
            if key not in model_type.model_fields:
                unknown.append((path, valid_names))
                continue

            nested_type = hints.get(key)
            if (
                isinstance(value, Mapping)
                and isinstance(nested_type, type)
                and issubclass(nested_type, BaseModel)
            ):
                unknown.extend(cls._unknown_keys(nested_type, value, path))
        return unknown

    @staticmethod
    def _format_unknown_keys(
        unknown: Sequence[tuple[str, Sequence[str]]],
    ) -> str:
        lines = ["Unknown configuration keys:"]
        for path, candidates in unknown:
            leaf = path.rsplit(".", 1)[-1]
            suggestion = difflib.get_close_matches(leaf, candidates, n=1)
            message = f"  - {path}"
            if suggestion:
                parent = path.rsplit(".", 1)[0] if "." in path else ""
                suggested_path = (
                    f"{parent}.{suggestion[0]}" if parent else suggestion[0]
                )
                message += f" (did you mean {suggested_path!r}?)"
            lines.append(message)
        return "\n".join(lines)

    @staticmethod
    def _deep_merge(
        base: Mapping[str, Any],
        overrides: Mapping[str, Any],
    ) -> dict[str, Any]:
        merged = dict(base)
        for key, value in overrides.items():
            if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
                merged[key] = ArgsParser._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    @staticmethod
    def _print_report(
        config_path: Path | None,
        yaml_values: Mapping[str, Any],
        cli_values: Mapping[str, Any],
    ) -> None:
        if config_path is not None:
            print(f"Loaded configuration: {config_path}")
            print("Values supplied by YAML:")
            for key in ArgsParser._leaf_paths(yaml_values):
                print(f"  {key}")
        if cli_values:
            print("Values supplied by CLI:")
            for key in ArgsParser._leaf_paths(cli_values):
                print(f"  {key}")

    @staticmethod
    def _leaf_paths(values: Mapping[str, Any], prefix: str = "") -> list[str]:
        paths: list[str] = []
        for key, value in values.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, Mapping):
                paths.extend(ArgsParser._leaf_paths(value, path))
            else:
                paths.append(path)
        return paths

    @staticmethod
    def _is_additional_args(field_type: Any) -> bool:
        return isinstance(field_type, type) and issubclass(
            field_type, AdditionalArgsBase
        )

    @staticmethod
    def _is_enum(field_type: Any) -> bool:
        return isinstance(field_type, type) and issubclass(field_type, Enum)
