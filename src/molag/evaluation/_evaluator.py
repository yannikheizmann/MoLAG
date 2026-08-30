from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .metrics import MetricsBase


class Evaluator:
    """Run batched model inference with modular streaming metrics."""

    def __init__(
        self,
        model,
        dataset: Dataset,
        data_collator: Callable,
        metrics: MetricsBase,
        batch_size: int,
        device: str | torch.device,
        dataloader_num_workers: int = 0,
    ) -> None:
        self._model = model
        self._metrics = metrics
        self._device = torch.device(device)
        self._loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=dataloader_num_workers,
            collate_fn=data_collator,
        )

    def evaluate(self) -> dict[str, float]:
        self._model.to(self._device)
        self._model.eval()
        self._metrics.reset()
        with torch.inference_mode():
            for inputs in self._loader:
                data = inputs["data"].to(self._device)
                outputs: dict[str, Any] = self._model(data=data)
                self._metrics.update(
                    logits=outputs["edge_logits"].detach().float().cpu().numpy(),
                    labels=inputs["edge_labels"].cpu().numpy(),
                    inputs=inputs,
                )
        return self._metrics.compute()

    def breakdown(self) -> dict[str, Any]:
        """Return structured diagnostics from the latest evaluation pass."""
        return self._metrics.breakdown()

    def sample_records(self) -> list[dict[str, Any]]:
        """Return scene-level records from the latest evaluation pass."""
        return self._metrics.sample_records()

    def tracker_records(self) -> list[dict[str, Any]]:
        """Return tracker-level records from the latest evaluation pass."""
        return self._metrics.tracker_records()
