"""The Process Reward Model (PRM).

A small Qwen3-0.6B backbone + a regression head, trained (MSE) on the LLM-judge
scores from judge.py to predict a step-quality score in [0,1]. In Phase 3 the
trained PRM becomes the DAPO reward signal.

    uv run python -m train.prm train  --labels artifacts/prm_labels.jsonl
    uv run python -m train.prm score  --model  artifacts/prm

Runs on the GPU box — needs the `rl` extra (torch + transformers). Heavy
imports are deferred into the commands so the module imports fine without them.
"""
import typer

from train.data import load_prm_examples

app = typer.Typer(help="PRM — train / sanity-check the process reward model")

_BASE_MODEL = "Qwen/Qwen3-0.6B"
_MAX_LEN = 1024


@app.command()
def train(
    labels: str = typer.Option("artifacts/prm_labels.jsonl", "--labels"),
    base_model: str = typer.Option(_BASE_MODEL, "--base-model"),
    out: str = typer.Option("artifacts/prm", "--out"),
    epochs: int = typer.Option(3, "--epochs"),
    batch_size: int = typer.Option(8, "--batch-size"),
    lr: float = typer.Option(2e-5, "--lr"),
    eval_frac: float = typer.Option(0.1, "--eval-frac"),
):
    """Train the PRM (MSE regression on judge scores)."""
    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments)

    examples = load_prm_examples(labels)
    if not examples:
        raise SystemExit(f"No examples in {labels} — run `train.judge` first.")
    print(f"loaded {len(examples)} labeled steps")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch: dict) -> dict:
        enc = tokenizer(batch["text"], truncation=True, max_length=_MAX_LEN)
        enc["labels"] = [float(s) for s in batch["score"]]
        return enc

    ds = Dataset.from_list([{"text": e.text, "score": e.score} for e in examples])
    ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
    split = ds.train_test_split(test_size=eval_frac, seed=42)

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=1, problem_type="regression",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    def compute_metrics(eval_pred) -> dict:
        preds, gold = eval_pred
        preds = np.asarray(preds).reshape(-1)
        return {"mae": float(np.mean(np.abs(preds - gold)))}

    args = TrainingArguments(
        output_dir=out,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="mae",
        greater_is_better=False,
        logging_steps=10,
        bf16=torch.cuda.is_available(),
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(out)
    tokenizer.save_pretrained(out)
    print(f"PRM saved -> {out}")


class PRM:
    """Inference wrapper — scores step text in [0,1]. Used by the DAPO reward."""

    def __init__(self, model_dir: str = "artifacts/prm", device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tok = AutoTokenizer.from_pretrained(model_dir)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self._model.to(self._device).eval()

    def score(self, texts: list[str]) -> list[float]:
        """Score a batch of build_prm_input() strings — each clamped to [0,1]."""
        enc = self._tok(
            texts, truncation=True, max_length=_MAX_LEN, padding=True, return_tensors="pt",
        ).to(self._device)
        with self._torch.no_grad():
            logits = self._model(**enc).logits.reshape(-1)
        return [max(0.0, min(1.0, float(x))) for x in logits]


@app.command()
def score(
    model: str = typer.Option("artifacts/prm", "--model"),
    labels: str = typer.Option("artifacts/prm_labels.jsonl", "--labels"),
    n: int = typer.Option(10, "--n", help="number of examples to spot-check"),
):
    """Sanity-check a trained PRM: predicted score vs judge label on N examples."""
    import random

    examples = load_prm_examples(labels)
    sample = random.Random(0).sample(examples, min(n, len(examples)))
    prm = PRM(model)
    preds = prm.score([e.text for e in sample])
    print(f"{'predicted':>10} {'judge':>8}  action")
    for ex, p in zip(sample, preds):
        print(f"{p:>10.3f} {ex.score:>8.2f}  {ex.meta.get('action', '')}")
    mae = sum(abs(p - e.score) for e, p in zip(sample, preds)) / len(sample)
    print(f"\nMAE on {len(sample)} samples: {mae:.3f}")


if __name__ == "__main__":
    app()
