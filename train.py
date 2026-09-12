"""Train our clean-room image-only ConvNeXt-Tiny Phase 2 baseline."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from convnext_baseline import (
    INDEX_TO_CLASS, balanced_class_weights, build_convnext_tiny,
    classification_metrics, collect_predictions, parameter_counts, set_seed,
)
from dataset_preparation import CLASS_TO_INDEX, prepare_ham10000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train image-only ConvNeXt-Tiny")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--csv-path", type=Path)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/convnext_tiny"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=7)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--allow-invalid-images", action="store_true")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)


def make_loader(dataset, batch_size, shuffle, workers, seed, pin_memory):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        pin_memory=pin_memory, worker_init_fn=seed_worker, generator=generator,
        persistent_workers=workers > 0,
    )


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)  # Metadata is intentionally ignored in Phase 2.
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    if len(loader.dataset) == 0:
        raise ValueError("Training dataset is empty")
    return total_loss / len(loader.dataset)


def atomic_save_checkpoint(checkpoint: dict, output_path: Path) -> None:
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(output_path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_ham10000(
        dataset_root=args.dataset_root, csv_path=args.csv_path,
        artifacts_dir=args.artifacts_dir, seed=args.seed,
        allow_invalid_images=args.allow_invalid_images,
        reuse_existing_splits=True,
    )
    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        prepared.train_dataset, args.batch_size, True, args.num_workers,
        args.seed, pin_memory,
    )
    val_loader = make_loader(
        prepared.val_dataset, args.batch_size, False, args.num_workers,
        args.seed + 1, pin_memory,
    )

    print("Loading pretrained ConvNeXt-Tiny weights...")
    model = build_convnext_tiny(pretrained=True).to(device)
    weights = balanced_class_weights(prepared.train_frame["label"].to_numpy()).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=args.scheduler_factor,
        patience=args.scheduler_patience, min_lr=1e-7,
    )
    checkpoint_path = args.output_dir / "best_convnext_tiny.pt"
    history_path = args.output_dir / "training_history.csv"
    fields = ["epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1", "lr"]
    best_f1, best_epoch, stale_epochs = -1.0, 0, 0
    print("Device:", device)
    print("Class weights:", dict(zip(INDEX_TO_CLASS, weights.detach().cpu().tolist())))
    print("Parameters:", parameter_counts(model))

    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, labels, probabilities = collect_predictions(
                model, val_loader, criterion, device
            )
            metrics = classification_metrics(labels, probabilities)
            current_lr = optimizer.param_groups[0]["lr"]
            row = {
                "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                "val_accuracy": metrics["accuracy"],
                "val_macro_f1": metrics["macro_f1"], "lr": current_lr,
            }
            writer.writerow(row)
            history_file.flush()
            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.5f} | "
                f"val_loss={val_loss:.5f} | val_accuracy={metrics['accuracy']:.5f} | "
                f"val_macro_f1={metrics['macro_f1']:.5f} | lr={current_lr:.3e}"
            )
            if metrics["macro_f1"] > best_f1:
                best_f1, best_epoch, stale_epochs = metrics["macro_f1"], epoch, 0
                atomic_save_checkpoint({
                    "phase": 2, "architecture": "torchvision_convnext_tiny",
                    "pretrained_weights": "ConvNeXt_Tiny_Weights.DEFAULT",
                    "num_classes": 7, "class_to_index": CLASS_TO_INDEX,
                    "model_state_dict": model.state_dict(),
                    "best_val_macro_f1": best_f1, "best_epoch": best_epoch,
                    "seed": args.seed, "training_arguments": vars(args),
                }, checkpoint_path)
                print(f"  Saved new best checkpoint: {checkpoint_path}")
            else:
                stale_epochs += 1
            scheduler.step(metrics["macro_f1"])
            if stale_epochs >= args.early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch}; best epoch={best_epoch}, "
                    f"best validation Macro-F1={best_f1:.5f}"
                )
                break
    print(f"Training complete. Best checkpoint: {checkpoint_path}")
    print(f"Best validation Macro-F1: {best_f1:.5f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
