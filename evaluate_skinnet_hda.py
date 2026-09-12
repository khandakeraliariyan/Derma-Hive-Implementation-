"""Evaluate our Phase 3 SkinNet-HDA clean-room reconstruction."""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
from pathlib import Path
import torch

from convnext_baseline import (
    balanced_class_weights, classification_metrics, file_size_megabytes,
    parameter_counts, per_class_report, save_confusion_matrix, set_seed,
)
from dataset_preparation import CLASS_TO_INDEX, prepare_ham10000
from skinnet_hda import FocalLoss, SkinNetHDA
from train_skinnet_hda import choose_device, collect_predictions, make_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Phase 3 SkinNet-HDA")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--csv-path", type=Path)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("outputs/skinnet_hda/best_skinnet_hda.pt"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/skinnet_hda/test"))
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
    split_path = args.artifacts_dir / "ham10000_splits.csv"
    if not split_path.is_file():
        raise FileNotFoundError(f"Existing Phase 1 split required: {split_path.resolve()}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_ham10000(
        args.dataset_root, args.artifacts_dir, args.csv_path, args.seed,
        allow_invalid_images=args.allow_invalid_images,
        reuse_existing_splits=True,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "skinnet_hda_clean_room_convnext_tiny":
        raise ValueError("Checkpoint is not the Phase 3 SkinNet-HDA reconstruction")
    if checkpoint.get("class_to_index") != CLASS_TO_INDEX:
        raise ValueError("Checkpoint class mapping does not match the dataset pipeline")
    expected_features = checkpoint["metadata_feature_names"]
    if expected_features != prepared.metadata_feature_names:
        raise ValueError("Checkpoint metadata features do not match preprocessing output")
    metadata_dim = checkpoint["model_config"]["metadata_dim"]
    model = SkinNetHDA(metadata_dim=metadata_dim, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    alpha = balanced_class_weights(prepared.train_frame["label"].to_numpy()).to(device)
    criterion = FocalLoss(alpha, gamma=2.0)
    loader = make_loader(
        prepared.test_dataset, args.batch_size, False, args.num_workers,
        args.seed + 2, device.type == "cuda",
    )
    test_loss, labels, probabilities = collect_predictions(
        model, loader, criterion, device
    )
    metrics = classification_metrics(labels, probabilities)
    metrics.update(parameter_counts(model))
    metrics.update({
        "test_loss": float(test_loss),
        "checkpoint_size_mb": file_size_megabytes(checkpoint_path),
        "checkpoint_best_val_macro_f1": float(checkpoint["best_val_macro_f1"]),
        "checkpoint_best_epoch": int(checkpoint["best_epoch"]),
        "metadata_dim": metadata_dim,
    })
    report = per_class_report(labels, probabilities)
    matrix = save_confusion_matrix(
        labels, probabilities, args.output_dir / "confusion_matrix.png"
    )
    for name, value in (
        ("test_metrics.json", metrics),
        ("classification_report.json", report),
        ("confusion_matrix.json", matrix.tolist()),
    ):
        with (args.output_dir / name).open("w", encoding="utf-8") as file:
            json.dump(value, file, indent=2)
    print("Device:", device)
    print(f"Test loss: {test_loss:.6f}")
    print(f"Accuracy: {metrics['accuracy']:.6f}")
    print(f"Precision (macro): {metrics['precision_macro']:.6f}")
    print(f"Recall (macro): {metrics['recall_macro']:.6f}")
    print(f"Macro-F1: {metrics['macro_f1']:.6f}")
    if metrics["auc_ovr_macro"] is None:
        print("AUC:", metrics["auc_note"])
    else:
        print(f"AUC (macro OVR, softmax probabilities): "
              f"{metrics['auc_ovr_macro']:.6f}")
    print("Total parameters:", metrics["total_parameters"])
    print("Trainable parameters:", metrics["trainable_parameters"])
    print(f"Checkpoint file size: {metrics['checkpoint_size_mb']:.2f} MB")
    print("Per-class classification report:")
    for class_name in CLASS_TO_INDEX:
        row = report[class_name]
        print(f"  {class_name:5s} precision={row['precision']:.4f} "
              f"recall={row['recall']:.4f} f1={row['f1-score']:.4f} "
              f"support={int(row['support'])}")
    print(f"Reports saved under: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

