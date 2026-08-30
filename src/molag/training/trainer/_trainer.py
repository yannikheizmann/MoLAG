"""Transformers trainer with streaming graph-affinity metrics."""

from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset
from transformers import Trainer as TransformersTrainer

from molag.config import TrainingArgs
from molag.evaluation import MetricsBase

from ._arguments import HuggingFaceTrainingAdapter


class Trainer(TransformersTrainer):
    """Transformers trainer with injected collation and streaming metrics."""

    def __init__(
        self,
        model,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        training_args: TrainingArgs,
        data_collator,
        metrics: MetricsBase,
    ) -> None:
        self._metrics = metrics
        self._resume_from_checkpoint = training_args.resume_from_checkpoint
        self._push_to_hub_after_training = training_args.push_to_hub
        super().__init__(
            model=model,
            args=HuggingFaceTrainingAdapter.create(training_args),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        """Run one prediction step and stream outputs into the metric collection."""
        loss, logits, labels = super().prediction_step(
            model,
            inputs,
            prediction_loss_only=False,
            ignore_keys=["node_embeddings"],
        )
        if logits is not None and labels is not None:
            predictions = logits[0] if isinstance(logits, tuple) else logits
            targets = labels[0] if isinstance(labels, tuple) else labels
            self._metrics.update(
                logits=predictions.detach().float().cpu().numpy(),
                labels=targets.detach().cpu().numpy(),
                inputs=inputs,
            )
        return loss, None, None

    def evaluation_loop(
        self,
        *args: Any,
        metric_key_prefix: str = "eval",
        **kwargs: Any,
    ):
        """Evaluate the model and append the accumulated streaming metrics."""
        self._metrics.reset()
        output = super().evaluation_loop(
            *args,
            metric_key_prefix=metric_key_prefix,
            **kwargs,
        )
        output.metrics.update(
            {
                f"{metric_key_prefix}_{name}": value
                for name, value in self._metrics.compute().items()
            }
        )
        return output

    def train(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        """Train, persist the final state, and optionally publish the model."""
        if "resume_from_checkpoint" not in kwargs:
            kwargs["resume_from_checkpoint"] = (
                str(self._resume_from_checkpoint)
                if self._resume_from_checkpoint is not None
                else None
            )
        result = super().train(*args, **kwargs)
        self.save_model()
        self.save_state()
        if self._push_to_hub_after_training:
            self.push_to_hub()
        return result.metrics or {}
