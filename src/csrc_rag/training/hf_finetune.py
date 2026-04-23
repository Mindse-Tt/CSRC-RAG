from __future__ import annotations

from dataclasses import dataclass

from csrc_rag.training.data import build_label_vocab, encode_multilabel


@dataclass(frozen=True)
class HFFineTuneConfig:
    model_name: str
    max_length: int
    batch_size: int
    learning_rate: float
    num_train_epochs: int


def run_hf_multilabel_finetune(
    train_rows: list[dict],
    valid_rows: list[dict],
    test_rows: list[dict],
    config: HFFineTuneConfig,
    output_dir: str,
) -> dict:
    try:
        import numpy as np  # noqa: F401
        from datasets import Dataset  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Hugging Face fine-tuning dependencies are unavailable. Install transformers, datasets, and torch in a Python 3.11 environment first."
        ) from exc

    from csrc_rag.training.metrics import multilabel_metrics

    label_vocab = build_label_vocab(train_rows + valid_rows + test_rows)
    num_labels = len(label_vocab)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    )

    def to_dataset(rows: list[dict]) -> Dataset:
        labels = encode_multilabel(rows, label_vocab).tolist()
        return Dataset.from_dict(
            {
                "text": [row["input_text"] for row in rows],
                "labels": labels,
            }
        )

    train_ds = to_dataset(train_rows)
    valid_ds = to_dataset(valid_rows)
    test_ds = to_dataset(test_rows)

    def preprocess(batch):
        return tokenizer(batch["text"], truncation=True, max_length=config.max_length)

    train_ds = train_ds.map(preprocess, batched=True)
    valid_ds = valid_ds.map(preprocess, batched=True)
    test_ds = test_ds.map(preprocess, batched=True)

    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_train_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_Micro-F1",
        greater_is_better=True,
        report_to="none",
        use_cpu=True,
        save_safetensors=False,
    )

    def compute_metrics(eval_pred):
        import numpy as np

        logits, labels = eval_pred
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= 0.5).astype(np.float32)
        return multilabel_metrics(labels, preds)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    test_metrics = trainer.evaluate(test_ds)
    return {
        "label_vocab": label_vocab,
        "test_metrics": test_metrics,
        "model_name": config.model_name,
    }
