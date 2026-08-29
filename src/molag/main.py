from __future__ import annotations

from collections.abc import Sequence
import logging
from typing import Literal

from molag.config import Args
from molag.dataset import (
    DatasetConfig,
    PyGTrackingAffinityCollator,
    TrackingDataset,
)
from molag.evaluation import AffinityMetrics
from molag.model import MoLAGModel
from molag.training.trainer import Trainer
from molag.utils.argparsing import ArgsParser
from molag.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)

Mode = Literal["finetune"]


class Main:
    """Entrypoints for MoLAG workflows."""

    @staticmethod
    def run(mode: Mode, argv: Sequence[str] | None = None) -> None:
        """Set up a command and route it to the selected workflow."""
        setup_logging()
        args = ArgsParser(Args, prog=mode).parse(argv)
        try:
            match mode:
                case "finetune":
                    Main._finetune(args)
        except Exception:
            LOGGER.exception("The %s workflow failed.", mode)
            raise

    @staticmethod
    def finetune() -> None:
        """Run the finetuning entrypoint."""
        Main.run("finetune")

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
