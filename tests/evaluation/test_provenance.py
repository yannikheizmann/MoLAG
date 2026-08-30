from pathlib import Path

from torch import nn

from molag.evaluation import EvaluationProvenance, FileFingerprint


def test_file_fingerprint_is_stable(tmp_path: Path) -> None:
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"molag")

    first = FileFingerprint.from_path(source)
    second = FileFingerprint.from_path(source)

    assert first == second
    assert len(first.sha256) == 64


def test_evaluation_provenance_fingerprints_all_inputs(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    configuration = run_directory / "config.yaml"
    checkpoint = run_directory / "model.safetensors"
    dataset = tmp_path / "dataset.yaml"
    predictions = run_directory / "predictions.npz"
    calibration = run_directory / "calibration.json"
    for path in (
        configuration,
        checkpoint,
        dataset,
        predictions,
        calibration,
    ):
        path.write_text(path.name)

    result = EvaluationProvenance.collect(
        run_directory=run_directory,
        dataset=dataset,
        predictions=predictions,
        calibration=calibration,
        model=nn.Linear(2, 3),
    )

    assert set(result.files) == {
        "configuration",
        "checkpoint",
        "dataset",
        "predictions",
        "calibration",
    }
    assert result.model_parameters == 9
    assert result.trainable_model_parameters == 9
    assert "torch" in result.package_versions
    assert result.to_dict()["files"]["dataset"]["path"] == str(dataset)
