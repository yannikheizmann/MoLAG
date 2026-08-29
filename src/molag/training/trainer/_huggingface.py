from typing import Any

from torch.utils.data import Dataset
from transformers import Trainer as TransformersTrainer

from molag.config import TrainingArgs
from molag.dataset import PyGTrackingAffinityCollator
from molag.evaluation import AffinityMetrics
from molag.model import MoLAGModel

from ._arguments import HuggingFaceTrainingAdapter
from ._base import TrainerBase


class _StreamingAffinityTrainer(TransformersTrainer):
    def __init__(self, *args: Any, metrics: AffinityMetrics, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._metrics = metrics

    def prediction_step(
        self, model, inputs, prediction_loss_only, ignore_keys=None
    ):
        loss, logits, labels = super().prediction_step(
            model,
            inputs,
            prediction_loss_only=False,
            ignore_keys=["node_embeddings"],
        )
        if logits is not None and labels is not None:
            edge_logits = logits[0] if isinstance(logits, tuple) else logits
            edge_labels = labels[0] if isinstance(labels, tuple) else labels
            self._metrics.update(
                logits=edge_logits.detach().float().cpu().numpy(),
                labels=edge_labels.detach().cpu().numpy(),
            )
        return loss, None, None

    def evaluation_loop(self, *args: Any, metric_key_prefix="eval", **kwargs: Any):
        self._metrics.reset()
        output = super().evaluation_loop(
            *args, metric_key_prefix=metric_key_prefix, **kwargs
        )
        output.metrics.update(
            {
                f"{metric_key_prefix}_{name}": value
                for name, value in self._metrics.compute().items()
            }
        )
        return output


class HuggingFaceAffinityTrainer(TrainerBase):
    """Train MoLAG with Hugging Face while streaming evaluation metrics."""

    def __init__(
        self,
        model: MoLAGModel,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        training_args: TrainingArgs,
        collator: PyGTrackingAffinityCollator | None = None,
    ) -> None:
        self._resume_from_checkpoint = training_args.resume_from_checkpoint
        self._arguments = HuggingFaceTrainingAdapter.create(training_args)
        self._trainer = _StreamingAffinityTrainer(
            model=model,
            args=self._arguments,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator or PyGTrackingAffinityCollator(),
            metrics=AffinityMetrics(),
        )

    def train(self) -> dict[str, float]:
        result = self._trainer.train(
            resume_from_checkpoint=(
                str(self._resume_from_checkpoint)
                if self._resume_from_checkpoint is not None
                else None
            )
        )
        self._trainer.save_model()
        self._trainer.save_state()
        return result.metrics or {}

    def evaluate(self) -> dict[str, float]:
        return self._trainer.evaluate()

    def predict(self, dataset: Dataset) -> Any:
        return self._trainer.predict(dataset)

    @property
    def output_dir(self) -> str:
        return self._arguments.output_dir
