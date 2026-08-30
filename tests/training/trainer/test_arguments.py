from pathlib import Path

from molag.config import TrainingArgs
from molag.training.trainer import HuggingFaceTrainingAdapter


def test_training_arguments_are_mapped_to_hugging_face() -> None:
    source = TrainingArgs(
        output_dir=Path("custom-results"),
        learning_rate=2e-4,
        per_device_train_batch_size=16,
        dataloader_num_workers=0,
        dataloader_persistent_workers=True,
        bf16=False,
    )

    target = HuggingFaceTrainingAdapter.create(source)

    assert target.output_dir == "custom-results"
    assert target.learning_rate == 2e-4
    assert target.per_device_train_batch_size == 16
    assert target.dataloader_persistent_workers is False
    assert target.remove_unused_columns is False
    assert target.label_names == ["edge_labels", "tracker_labels"]
    assert target.metric_for_best_model == "loss"


def test_external_integrations_are_mapped_only_when_enabled() -> None:
    target = HuggingFaceTrainingAdapter.create(
        TrainingArgs(
            bf16=False,
            report_to=["wandb"],
            run_name="paper-run",
            push_to_hub=True,
            hub_model_id="example/molag",
            hub_private_repo=True,
            hub_strategy="checkpoint",
        ),
    )

    assert target.report_to == ["wandb"]
    assert target.run_name == "paper-run"
    assert target.push_to_hub is True
    assert target.hub_model_id == "example/molag"
    assert target.hub_private_repo is True
    assert target.hub_strategy == "checkpoint"
