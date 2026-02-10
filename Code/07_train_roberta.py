"""
07_train_roberta.py

Fine-tune RoBERTa on the tokenized 1980s paragraph-filtered dataset.

Inputs:
    - Tokenized dataset directory (either streaming chunks or merged dataset)

Outputs:
    - models/roberta_1980s_paragraph/
    - evaluation/roberta_metrics.txt
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk, concatenate_datasets
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
    get_linear_schedule_with_warmup,
)

from utils import ModelEvaluator, get_device


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokenized_dataset(path: Path):
    # If user points to merged dataset directory
    if (path / "dataset_info.json").exists():
        return load_from_disk(str(path))

    # Otherwise treat as chunk directory
    chunk_dirs = sorted(
        [p for p in path.iterdir() if p.is_dir() and p.name.startswith("chunk_")],
        key=lambda x: int(x.name.split("_")[1]),
    )
    if not chunk_dirs:
        raise RuntimeError(f"No dataset found at: {path}")

    datasets = [load_from_disk(str(p)) for p in chunk_dirs]
    return concatenate_datasets(datasets)


def evaluate_roberta(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            preds = torch.argmax(outputs.logits, dim=-1)
            all_labels.extend(batch["labels"].cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return all_labels, all_preds


def main(args):
    set_seed(args.seed)
    device = get_device()
    print("Using device:", device)

    tokenized_path = Path(args.tokenized_path)
    model_dir = Path(args.model_dir)

    model_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tokenized dataset...")
    dataset = load_tokenized_dataset(tokenized_path)
    dataset = dataset.shuffle(seed=args.seed)

    train_test = dataset.train_test_split(test_size=0.2, seed=args.seed)
    test_valid = train_test["test"].train_test_split(test_size=0.5, seed=args.seed)

    train_ds = train_test["train"]
    valid_ds = test_valid["train"]
    test_ds = test_valid["test"]

    cols = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=cols)
    valid_ds.set_format(type="torch", columns=cols)
    test_ds.set_format(type="torch", columns=cols)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    print("Loading model...")
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    evaluator = ModelEvaluator(args.eval_dir)

    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            scheduler.step()
            total_loss += loss.item()

            if (step + 1) % args.log_every == 0:
                elapsed = time.time() - start_time
                steps_per_sec = (step + 1) / max(elapsed, 1e-8)
                eta = (len(train_loader) - (step + 1)) / steps_per_sec
                print(
                    f"Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | "
                    f"Loss {total_loss/(step+1):.4f} | ETA {eta/60:.2f} min"
                )

        val_labels, val_preds = evaluate_roberta(model, valid_loader, device)
        val_metrics = evaluator.compute_metrics(val_labels, val_preds, pos_label=1)
        print(f"Validation acc: {val_metrics['accuracy']:.4f}, F1 (R): {val_metrics['f1']:.4f}")

    test_labels, test_preds = evaluate_roberta(model, test_loader, device)
    test_metrics = evaluator.compute_metrics(test_labels, test_preds, pos_label=1)
    print(f"Test accuracy: {test_metrics['accuracy']:.4f} | Test F1 (R): {test_metrics['f1']:.4f}")

    evaluator.save_report(test_metrics, "roberta_metrics.txt", header="RoBERTa Test Metrics")
    evaluator.plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        labels=["D", "R"],
        filename="roberta_confusion_matrix.png",
        title="RoBERTa Confusion Matrix (Test)",
    )

    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    print("Saved model to:", model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenized_path",
        type=str,
        default="data/processed/tokenized_1980s_paragraph_full",
        help="Path to tokenized dataset (merged dir or chunk dir)",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models/roberta_1980s_paragraph",
        help="Directory to save the fine-tuned model",
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        default="evaluation",
        help="Directory to save evaluation outputs",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=200)
    args = parser.parse_args()
    main(args)
