from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from torch.utils.data import DataLoader, Dataset

from molag.dataset import EvalDataset, EvalSample, PyGTrackingAffinityCollator

PROFILE = Path("src/molag/dataset/profiles/molag_standard.yaml")


@pytest.fixture(scope="module")
def frozen_dataset() -> EvalDataset:
    return EvalDataset.generate("frozen", PROFILE, size=4, seed=41)


def test_generated_dataset_is_deterministic() -> None:
    first = EvalDataset.generate("first", PROFILE, size=3, seed=41)
    second = EvalDataset.generate("second", PROFILE, size=3, seed=41)

    assert first.samples == second.samples
    assert first.candidate_seed_ranges == [[41, 43]]


def test_different_seeds_generate_different_scenes() -> None:
    first = EvalDataset.generate("first", PROFILE, size=1, seed=41)
    second = EvalDataset.generate("second", PROFILE, size=1, seed=42)

    assert first.samples != second.samples


def test_generation_records_metadata(frozen_dataset: EvalDataset) -> None:
    assert frozen_dataset.name == "frozen"
    assert frozen_dataset.profile == str(PROFILE)
    assert frozen_dataset.size == 4
    assert frozen_dataset.seed == 41
    assert frozen_dataset.created_at
    assert frozen_dataset.candidate_seed_ranges == [[41, 44]]


def test_yaml_round_trip_preserves_every_field(
    frozen_dataset: EvalDataset,
    tmp_path: Path,
) -> None:
    destination = frozen_dataset.to_yaml(tmp_path / "evaluation.yaml")

    restored = EvalDataset.from_yaml(destination)

    assert restored == frozen_dataset


def test_eval_dataset_implements_torch_dataset(
    frozen_dataset: EvalDataset,
) -> None:
    sample = frozen_dataset[0]

    assert isinstance(frozen_dataset, Dataset)
    assert len(frozen_dataset) == 4
    assert set(sample) == {"x", "y"}
    assert sample["x"].dtype == torch.float32
    assert sample["y"].dtype == torch.int64
    assert sample["x"].ndim == 2
    assert sample["x"].shape[1] == 2
    assert sample["y"].shape == sample["x"].shape


def test_dataset_works_with_dataloader_and_production_collator(
    frozen_dataset: EvalDataset,
) -> None:
    loader = DataLoader(
        frozen_dataset,
        batch_size=2,
        collate_fn=PyGTrackingAffinityCollator(),
    )

    batches = list(loader)

    assert len(batches) == 2
    assert batches[0]["data"].num_graphs == 2
    assert batches[0]["edge_labels"].ndim == 1


def test_variable_scene_sizes_are_batched_correctly() -> None:
    frozen = EvalDataset(
        name="variable",
        profile=str(PROFILE),
        size=2,
        seed=0,
        created_at="2026-01-01T00:00:00+00:00",
        candidate_seed_ranges=[[0, 1]],
        samples=[
            EvalSample(
                x=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                y=[[0, 0], [0, 1], [0, 2]],
            ),
            EvalSample(
                x=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
                y=[[0, 0], [0, 1], [1, 0], [1, 1]],
            ),
        ],
    )

    batch = next(
        iter(
            DataLoader(
                frozen,
                batch_size=2,
                collate_fn=PyGTrackingAffinityCollator(),
            )
        )
    )

    assert batch["data"].num_graphs == 2
    assert batch["data"].num_nodes == 7
    assert batch["edge_labels"].numel() == 9


def test_out_of_range_index_is_reported(frozen_dataset: EvalDataset) -> None:
    with pytest.raises(IndexError):
        _ = frozen_dataset[len(frozen_dataset)]


def test_returned_tensors_do_not_mutate_frozen_sample(
    frozen_dataset: EvalDataset,
) -> None:
    original = frozen_dataset[0]["x"].clone()

    frozen_dataset[0]["x"].zero_()

    torch.testing.assert_close(frozen_dataset[0]["x"], original)


def test_negative_spurious_labels_survive_yaml_round_trip(tmp_path: Path) -> None:
    dataset = EvalDataset(
        name="spurious",
        profile=str(PROFILE),
        size=1,
        seed=0,
        created_at="2026-01-01T00:00:00+00:00",
        candidate_seed_ranges=[[0, 0]],
        samples=[
            EvalSample(
                x=[[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]],
                y=[[0, 0], [-1, -1], [-2, -1]],
            )
        ],
    )

    restored = EvalDataset.from_yaml(dataset.to_yaml(tmp_path / "spurious.yaml"))

    assert restored[0]["y"][:, 0].tolist() == [0, -1, -2]


@pytest.mark.parametrize(
    ("x", "y", "message"),
    [
        ([], [], "at least one point"),
        ([[0.0, 0.0]], [], "same number"),
        ([[0.0]], [[0, 0]], "x must have shape"),
        ([[0.0, 0.0]], [[0]], "y must have shape"),
        ([[np.inf, 0.0]], [[0, 0]], "finite coordinates"),
    ],
)
def test_invalid_sample_shapes_are_rejected(x, y, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        EvalSample(x=x, y=y)


def test_size_must_match_frozen_samples() -> None:
    with pytest.raises(ValidationError, match="size must equal"):
        EvalDataset(
            name="invalid",
            profile=str(PROFILE),
            size=2,
            seed=0,
            created_at="now",
            candidate_seed_ranges=[[0, 1]],
            samples=[],
        )


@pytest.mark.parametrize("seed_ranges", [[], [[3]], [[4, 3]]])
def test_invalid_seed_provenance_is_rejected(seed_ranges) -> None:
    sample = EvalSample(x=[[0.0, 0.0]], y=[[0, 0]])

    with pytest.raises(ValidationError):
        EvalDataset(
            name="invalid",
            profile=str(PROFILE),
            size=1,
            seed=0,
            created_at="now",
            candidate_seed_ranges=seed_ranges,
            samples=[sample],
        )


def test_missing_yaml_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="evaluation dataset"):
        EvalDataset.from_yaml(tmp_path / "missing.yaml")


def test_non_mapping_yaml_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("- not\n- a\n- mapping\n")

    with pytest.raises(ValueError, match="must contain a mapping"):
        EvalDataset.from_yaml(path)
