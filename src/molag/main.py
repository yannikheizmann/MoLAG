from __future__ import annotations

from collections.abc import Sequence
import json
import logging
from pathlib import Path
from typing import Literal

from molag.config import (
    Args,
    CALIBRATION_PREDICTION_CACHE_FILENAME,
    CALIBRATION_RESULT_FILENAME,
    EVALUATION_RESULT_FILENAME,
    EvalDatasetGenerationArgs,
    EvaluationArgs,
    PREDICTION_CACHE_FILENAME,
)
from molag.dataset import (
    DatasetConfig,
    EvalDataset,
    PyGTrackingAffinityCollator,
    TrackingDataset,
)
from molag.evaluation import (
    CombinedMetrics,
    Evaluator,
    EvaluationProvenance,
    EvaluationResult,
    ModelLoader,
    ThresholdCalibrator,
)
from molag.model import MoLAGModel
from molag.inference import PredictionGenerator
from molag.training.trainer import Trainer
from molag.utils.argparsing import ArgsParser
from molag.utils.logging import setup_logging
from molag.utils.registry import Registry

LOGGER = logging.getLogger(__name__)

Mode = Literal["finetune", "evaluate", "generate_eval_dataset"]


class Main:
    """Entrypoints for MoLAG workflows."""

    @staticmethod
    def run(mode: Mode, argv: Sequence[str] | None = None) -> None:
        """Set up a command and route it to the selected workflow."""
        setup_logging()
        args = ArgsParser(Args).parse(argv)
        try:
            match mode:
                case "finetune":
                    Main._finetune(args)
                case "evaluate":
                    Main._evaluate(args.evaluation_args)
                case "generate_eval_dataset":
                    Main._generate_eval_dataset(
                        args.eval_dataset_generation_args
                    )
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
    def generate_eval_dataset() -> None:
        """Run the frozen evaluation-dataset generation entrypoint."""
        Main.run("generate_eval_dataset")

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
            metrics=Main._create_metrics(args.training_args.training_metrics),
        )
        return trainer.train()

    @staticmethod
    def _evaluate(args: EvaluationArgs) -> dict[str, float]:
        """Calibrate and evaluate a finetuned model on frozen scenes."""
        model_directory = (
            ModelLoader.from_hub(
                args.hub_model_id,
                revision=args.hub_revision,
            )
            if args.hub_model_id is not None
            else args.run_directory
        )
        model = ModelLoader.from_run_directory(model_directory, args.device)
        calibration = None
        if args.threshold is None:
            calibration = Main._calibrate(args, model)
            threshold = calibration["threshold"]
        else:
            threshold = args.threshold
        dataset = EvalDataset.from_yaml(args.dataset)
        metrics = Main._create_metrics(args.metrics, threshold)
        evaluator = Evaluator(
            model=model,
            dataset=dataset,
            data_collator=PyGTrackingAffinityCollator(),
            metrics=metrics,
            batch_size=args.batch_size,
            device=args.device,
            dataloader_num_workers=args.dataloader_num_workers,
        )
        predictions = evaluator.predict()
        prediction_path = predictions.to_npz(
            args.run_directory / PREDICTION_CACHE_FILENAME
        )
        result = evaluator.evaluate(predictions)
        breakdown = evaluator.breakdown()
        provenance = EvaluationProvenance.collect(
            run_directory=args.run_directory,
            model_directory=model_directory,
            dataset=args.dataset,
            predictions=prediction_path,
            model=model,
            calibration=(
                args.run_directory / CALIBRATION_RESULT_FILENAME
                if calibration is not None
                else None
            ),
            calibration_predictions=(
                args.run_directory / CALIBRATION_PREDICTION_CACHE_FILENAME
                if calibration is not None
                else None
            ),
        )
        Main._save_evaluation_result(
            args,
            dataset,
            threshold,
            result,
            breakdown,
            evaluator.sample_records(),
            evaluator.tracker_records(),
            provenance,
            calibration is not None,
        )
        return result

    @staticmethod
    def _save_evaluation_result(
        args: EvaluationArgs,
        dataset: EvalDataset,
        threshold: float,
        metrics: dict[str, float],
        breakdown: dict,
        samples: list[dict],
        trackers: list[dict],
        provenance: EvaluationProvenance,
        calibrated: bool,
    ) -> None:
        output = args.run_directory / EVALUATION_RESULT_FILENAME
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "run_directory": str(args.run_directory),
            "model_source": (
                {
                    "hub_model_id": args.hub_model_id,
                    "hub_revision": args.hub_revision,
                }
                if args.hub_model_id is not None
                else {"run_directory": str(args.run_directory)}
            ),
            "dataset": str(args.dataset),
            "dataset_name": dataset.name,
            "candidate_seed_ranges": dataset.candidate_seed_ranges,
            "calibration": (
                str(args.run_directory / CALIBRATION_RESULT_FILENAME)
                if calibrated
                else None
            ),
            "threshold": threshold,
            "predictions": str(
                args.run_directory / PREDICTION_CACHE_FILENAME
            ),
            "provenance": provenance.to_dict(),
        }
        EvaluationResult(
            metrics=metrics,
            breakdown=breakdown,
            samples=samples,
            trackers=trackers,
        ).write(output, metadata)

    @staticmethod
    def _generate_eval_dataset(
        args: EvalDatasetGenerationArgs,
    ) -> EvalDataset:
        """Materialize and save a frozen evaluation dataset."""
        if args.samples_per_tracker_count is None:
            dataset = EvalDataset.generate(
                name=args.name,
                profile_path=args.dataset_profile,
                size=args.size,
                seed=args.seed,
                description=args.description,
            )
        else:
            dataset = EvalDataset.generate_stratified(
                name=args.name,
                profile_path=args.dataset_profile,
                samples_per_tracker_count=args.samples_per_tracker_count,
                min_trackers=args.min_trackers,
                max_trackers=args.max_trackers,
                seed=args.seed,
                max_attempts_per_tracker_count=(
                    args.max_attempts_per_tracker_count
                ),
                description=args.description,
            )
        dataset.to_yaml(args.output)
        return dataset

    @staticmethod
    def _calibrate(args: EvaluationArgs, model) -> dict:
        """Calibrate the grouping threshold on frozen scenes."""
        dataset = EvalDataset.from_yaml(args.calibration_dataset)
        predictions = PredictionGenerator(
            model=model,
            dataset=dataset,
            data_collator=PyGTrackingAffinityCollator(),
            batch_size=args.batch_size,
            device=args.device,
            dataloader_num_workers=args.dataloader_num_workers,
        ).predict()
        prediction_path = predictions.to_npz(
            args.run_directory / CALIBRATION_PREDICTION_CACHE_FILENAME
        )
        steps = round(
            (args.threshold_max - args.threshold_min) / args.threshold_step
        )
        thresholds = [
            round(args.threshold_min + index * args.threshold_step, 10)
            for index in range(steps + 1)
        ]
        result = ThresholdCalibrator(
            metric_factory=lambda threshold: Main._create_metrics(
                args.metrics, threshold
            ),
            objective=args.objective,
            thresholds=thresholds,
        ).calibrate(predictions)
        payload = {
            "run_directory": str(args.run_directory),
            "dataset": str(args.calibration_dataset),
            "dataset_name": dataset.name,
            "candidate_seed_ranges": dataset.candidate_seed_ranges,
            "predictions": str(prediction_path),
            "metrics": args.metrics,
            "objective": result.objective,
            "threshold": result.threshold,
            "objective_value": result.objective_value,
            "results": [
                {"threshold": threshold, "metrics": metrics}
                for threshold, metrics in result.metrics_by_threshold.items()
            ],
        }
        output = args.run_directory / CALIBRATION_RESULT_FILENAME
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n")
        return payload

    @staticmethod
    def _create_metrics(
        names: Sequence[str],
        threshold: float = 0.5,
    ) -> CombinedMetrics:
        """Instantiate registered streaming metrics with a shared threshold."""
        return CombinedMetrics(
            [
                Registry.get("MetricsBase", name)(threshold=threshold)
                for name in names
            ]
        )
