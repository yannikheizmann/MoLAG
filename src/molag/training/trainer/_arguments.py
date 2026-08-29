from transformers import TrainingArguments

from molag.config import TrainingArgs


class HuggingFaceTrainingAdapter:
    """Translate backend-neutral arguments to Hugging Face configuration."""

    @staticmethod
    def create(args: TrainingArgs) -> TrainingArguments:
        persistent_workers = (
            args.dataloader_persistent_workers and args.dataloader_num_workers > 0
        )
        return TrainingArguments(
            output_dir=str(args.output_dir),
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_train_epochs=args.num_train_epochs,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_ratio=args.warmup_ratio,
            logging_steps=args.logging_steps,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            dataloader_num_workers=args.dataloader_num_workers,
            dataloader_persistent_workers=persistent_workers,
            fp16=args.fp16,
            bf16=args.bf16,
            seed=args.seed,
            metric_for_best_model=args.metric_for_best_model,
            greater_is_better=args.metric_for_best_model != "loss",
            load_best_model_at_end=args.load_best_model_at_end,
            remove_unused_columns=False,
            label_names=["edge_labels", "tracker_labels"],
            report_to="none",
        )
