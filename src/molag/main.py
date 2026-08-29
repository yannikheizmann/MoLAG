from __future__ import annotations

from collections.abc import Sequence
import json
import logging
from pathlib import Path
from typing import Literal

from molag.config import Args, EvaluationArgs
from molag.dataset import (
    DatasetConfig,
    EvalDataset,
    PyGTrackingAffinityCollator,
    TrackingDataset,
)
from molag.evaluation import (
    AffinityMetrics,
    CombinedMetrics,
    Evaluator,
    ModelLoader,
)
from molag.model import MoLAGModel
from molag.training.trainer import Trainer
from molag.utils.argparsing import ArgsParser
from molag.utils.logging import setup_logging
from molag.utils.registry import Registry

LOGGER = logging.getLogger(__name__)

Mode = Literal["finetune", "evaluate"]


class Main:
    """Entrypoints for MoLAG workflows."""

    @staticmethod
    def run(mode: Mode, argv: Sequence[str] | None = None) -> None:
        """Set up a command and route it to the selected workflow."""
        setup_logging()
        try:
            match mode:
                case "finetune":
                    args = ArgsParser(Args, prog=mode).parse(argv)
                    Main._finetune(args)
                case "evaluate":
                    args = ArgsParser(EvaluationArgs, prog=mode).parse(argv)
                    Main._evaluate(args)
        except Exception:
            LOGGER.exception("The %s workflow failed.", mode)
            raise

    @staticmethod
    def finetune() -> None:
        """Run the finetuning entrypoint."""
        Main.run("finetune")

    @staticmethod
    def evaluate() -> None:
        """Run the evaluation entrypoint."""
        Main.run("evaluate")

    @staticmethod
    def _finetune(args: Args) -> dict[str, float]:
        """Train MoLAG on disjoint generated training and evaluation splits."""
        profile = DatasetConfig.from_yaml(args.dataset_args.dataset_profile)
        train_config = profile.model_copy(
            update={"size": args.dataset_args.train_size, "seed": profile.seed}
        )
        eval_config = profile.model_copy(
            update={
                "size": args.dataset_args.eval_size,
                "seed": profile.seed + args.dataset_args.train_size,
            }
        )

        output_dir = args.training_args.output_dir
        args.save(output_dir, format="yaml")
        profile.to_yaml(output_dir / "dataset_profile.yaml")

        trainer = Trainer(
            model=MoLAGModel(args.model_args, args.loss_args),
            train_dataset=TrackingDataset.from_config(train_config),
            eval_dataset=TrackingDataset.from_config(eval_config),
            training_args=args.training_args,
            data_collator=PyGTrackingAffinityCollator(),
            metrics=AffinityMetrics(),
        )
        return trainer.train()

    @staticmethod
    def _evaluate(args: EvaluationArgs) -> dict[str, float]:
        """Evaluate a finetuned model on frozen scenes."""
        dataset = EvalDataset.from_yaml(args.dataset)
        metrics = CombinedMetrics(
            [
                Registry.get("MetricsBase", name)(threshold=args.threshold)
                for name in args.metrics
            ]
        )
        evaluator = Evaluator(
            model=ModelLoader.from_run_directory(args.run_directory, args.device),
            dataset=dataset,
            data_collator=PyGTrackingAffinityCollator(),
            metrics=metrics,
            batch_size=args.batch_size,
            device=args.device,
            dataloader_num_workers=args.dataloader_num_workers,
        )
        result = evaluator.evaluate()
        Main._save_evaluation_result(args, dataset, result)
        return result

    @staticmethod
    def _save_evaluation_result(
        args: EvaluationArgs,
        dataset: EvalDataset,
        metrics: dict[str, float],
    ) -> None:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_directory": str(args.run_directory),
            "dataset": str(args.dataset),
            "dataset_name": dataset.name,
            "candidate_seed_ranges": dataset.candidate_seed_ranges,
            "threshold": args.threshold,
            "metrics": metrics,
        }
        output.write_text(json.dumps(payload, indent=2) + "\n")
