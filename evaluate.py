"""Evaluate our clean-room image-only ConvNeXt-Tiny Phase 2 baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from convnext_baseline import (
    balanced_class_weights, build_convnext_tiny, classification_metrics,
    collect_predictions, file_size_megabytes, parameter_counts,
    per_class_report, save_confusion_matrix, set_seed,
)
from dataset_preparation import CLASS_TO_INDEX, prepare_ham10000
from train import choose_device, make_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate image-only ConvNeXt-Tiny")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--csv-path", type=Path)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("outputs/convnext_tiny/best_convnext_tiny.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/convnext_tiny/test"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--allow-invalid-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_ham10000(
        dataset_root=args.dataset_root, csv_path=args.csv_path,
        artifacts_dir=args.artifacts_dir, seed=args.seed,
        allow_invalid_images=args.allow_invalid_images,
        reuse_existing_splits=True,
    )
    loader = make_loader(
        prepared.test_dataset, args.batch_size, False, args.num_workers,
        args.seed + 2, device.type == "cuda",
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("class_to_index") != CLASS_TO_INDEX:
        raise ValueError("Checkpoint class mapping does not match this pipeline")
    if checkpoint.get("architecture") != "torchvision_convnext_tiny":
        raise ValueError("Checkpoint is not the Phase 2 ConvNeXt-Tiny baseline")
    model = build_convnext_tiny(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    weights = balanced_class_weights(prepared.train_frame["label"].to_numpy()).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    test_loss, labels, probabilities = collect_predictions(
        model, loader, criterion, device
    )
    metrics = classification_metrics(labels, probabilities)
    metrics["test_loss"] = float(test_loss)
    metrics.update(parameter_counts(model))
    metrics["checkpoint_size_mb"] = file_size_megabytes(checkpoint_path)
    metrics["checkpoint_best_val_macro_f1"] = float(checkpoint["best_val_macro_f1"])
    metrics["checkpoint_best_epoch"] = int(checkpoint["best_epoch"])
    report = per_class_report(labels, probabilities)
    matrix = save_confusion_matrix(
        labels, probabilities, args.output_dir / "confusion_matrix.png"
    )
    with (args.output_dir / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    with (args.output_dir / "classification_report.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)
    with (args.output_dir / "confusion_matrix.json").open("w", encoding="utf-8") as file:
        json.dump(matrix.tolist(), file, indent=2)

    print("Device:", device)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test loss: {test_loss:.6f}")
    print(f"Accuracy: {metrics['accuracy']:.6f}")
    print(f"Precision (macro): {metrics['precision_macro']:.6f}")
    print(f"Recall (macro): {metrics['recall_macro']:.6f}")
    print(f"Macro-F1: {metrics['macro_f1']:.6f}")
    if metrics["auc_ovr_macro"] is None:
        print("AUC:", metrics["auc_note"])
    else:
        print(f"AUC (macro OVR, softmax probabilities): {metrics['auc_ovr_macro']:.6f}")
    print("Total parameters:", metrics["total_parameters"])
    print("Trainable parameters:", metrics["trainable_parameters"])
    print(f"Checkpoint file size: {metrics['checkpoint_size_mb']:.2f} MB")
    print("\nPer-class classification report:")
    for class_name in CLASS_TO_INDEX:
        row = report[class_name]
        print(
            f"  {class_name:5s} precision={row['precision']:.4f} "
            f"recall={row['recall']:.4f} f1={row['f1-score']:.4f} "
            f"support={int(row['support'])}"
        )
    print(f"Reports saved under: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
