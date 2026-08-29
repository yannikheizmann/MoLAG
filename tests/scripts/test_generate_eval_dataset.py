from pathlib import Path

from molag.config import EvalDatasetGenerationArgs
from molag.dataset import EvalDataset
from molag.main import Main

PROFILE = Path("src/molag/dataset/profiles/molag_standard.yaml")


def test_generate_eval_dataset_writes_frozen_scenes(tmp_path: Path) -> None:
    output = tmp_path / "datasets" / "calibration.yaml"
    args = EvalDatasetGenerationArgs(
        name="calibration",
        dataset_profile=PROFILE,
        size=3,
        seed=5_010_000,
        output=output,
        description="Held-out threshold calibration scenes.",
    )

    generated = Main._generate_eval_dataset(args)
    restored = EvalDataset.from_yaml(output)

    assert restored == generated
    assert restored.name == "calibration"
    assert restored.description == "Held-out threshold calibration scenes."
    assert restored.candidate_seed_ranges == [[5_010_000, 5_010_002]]


def test_run_uses_generation_argument_schema(monkeypatch, tmp_path: Path) -> None:
    captured: list[EvalDatasetGenerationArgs] = []
    monkeypatch.setattr(
        Main,
        "_generate_eval_dataset",
        lambda args: captured.append(args),
    )

    Main.run(
        "generate_eval_dataset",
        [
            "--eval_dataset_generation_args",
            "name=test",
            f"dataset_profile={PROFILE}",
            "size=10",
            "seed=123",
            f"output={tmp_path / 'test.yaml'}",
        ],
    )

    assert captured[0].name == "test"
    assert captured[0].size == 10
    assert captured[0].seed == 123


def test_generate_stratified_eval_dataset(tmp_path: Path) -> None:
    output = tmp_path / "stratified.yaml"
    generated = Main._generate_eval_dataset(
        EvalDatasetGenerationArgs(
            name="stratified",
            dataset_profile=PROFILE,
            samples_per_tracker_count=1,
            min_trackers=1,
            max_trackers=2,
            seed=200,
            output=output,
        )
    )

    assert generated.size == 2
    assert len(generated.candidate_seed_ranges) == 2
    assert EvalDataset.from_yaml(output) == generated


def test_generate_entrypoint_routes_through_run(monkeypatch) -> None:
    modes: list[str] = []
    monkeypatch.setattr(Main, "run", lambda mode: modes.append(mode))

    Main.generate_eval_dataset()

    assert modes == ["generate_eval_dataset"]
