from pathlib import Path

from molag.config import LossArgs, ModelArgs, TrainingArgs
from molag.dataset import DatasetConfig, PoseConfig, TrackingDataset
from molag.model import MoLAGModel
from molag.training.trainer import HuggingFaceAffinityTrainer


def test_tiny_training_run(tmp_path: Path) -> None:
    dataset = TrackingDataset.from_config(
        DatasetConfig(
            size=4,
            num_trackers=2,
            pose=PoseConfig(
                x_min=-1,
                x_max=1,
                y_min=-1,
                y_max=1,
                z_min=300,
                z_max=301,
                max_tilt_deg=5,
            ),
        )
    )
    model = MoLAGModel(
        ModelArgs(hidden_dims=[4, 4], edge_head_dims=[4]),
        LossArgs(supcon_weight=0),
    )
    args = TrainingArgs(
        output_dir=tmp_path,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        dataloader_num_workers=0,
        dataloader_persistent_workers=False,
        logging_steps=1,
        eval_steps=100,
        save_steps=100,
        load_best_model_at_end=False,
        bf16=False,
    )
    trainer = HuggingFaceAffinityTrainer(model, dataset, dataset, args)

    metrics = trainer.train()

    evaluation_metrics = trainer.evaluate()

    assert "train_loss" in metrics
    assert "eval_edge_accuracy" in evaluation_metrics
    assert (tmp_path / "model.safetensors").exists()
