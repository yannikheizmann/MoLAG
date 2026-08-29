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
