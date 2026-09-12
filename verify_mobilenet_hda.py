"""Forward/backward verification for Phase 4 MobileNet-HDA."""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from pathlib import Path
import torch

from convnext_baseline import balanced_class_weights, parameter_counts, set_seed
from dataset_preparation import prepare_ham10000
from mobilenet_hda import FocalLoss, MobileNetHDA
from train_mobilenet_hda import choose_device, make_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Phase 4 MobileNet-HDA")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--csv-path", type=Path)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def verify_tensors(model, images, metadata, labels, criterion) -> None:
    model.train()
    logits, shapes = model(images, metadata, return_shapes=True)
    batch = images.shape[0]
    expected = {
        "image": (batch, 3, 224, 224),
        "mobilenet": (batch, 960, 7, 7),
        "se": (batch, 960, 7, 7),
        "spatial_attention": (batch, 960, 7, 7),
        "tokens": (batch, 49, 960),
        "transformer": (batch, 49, 960),
        "pooled_image_feature": (batch, 960),
        "metadata": (batch, metadata.shape[1]),
        "fused_feature": (batch, 960 + metadata.shape[1]),
        "logits": (batch, 7),
    }
    for name, shape in shapes.items():
        print(f"{name:22s} {shape}")
        if shape != expected[name]:
            raise AssertionError(f"{name}: expected {expected[name]}, got {shape}")
    loss = criterion(logits, labels)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()
                 if parameter.requires_grad]
    if not gradients or any(gradient is None for gradient in gradients):
        raise AssertionError("A trainable parameter did not receive a gradient")
    if not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise AssertionError("Non-finite gradient detected")
    print(f"focal_loss              {loss.item():.6f}")
    print("backward                PASS")
    print("parameters              ", parameter_counts(model))
    print("Phase 4 verification    PASS")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    if not (args.artifacts_dir / "ham10000_splits.csv").is_file():
        raise FileNotFoundError("Existing Phase 1 split is required")
    prepared = prepare_ham10000(
        args.dataset_root, args.artifacts_dir, args.csv_path, args.seed,
        reuse_existing_splits=True,
    )
    loader = make_loader(prepared.train_dataset, args.batch_size, False,
                         args.num_workers, args.seed, device.type == "cuda")
    batch = next(iter(loader))
    images = batch["image"].to(device)
    metadata = batch["metadata"].to(device)
    labels = batch["label"].to(device)
    model = MobileNetHDA(len(prepared.metadata_feature_names), pretrained=True).to(device)
    alpha = balanced_class_weights(prepared.train_frame["label"].to_numpy()).to(device)
    verify_tensors(model, images, metadata, labels, FocalLoss(alpha, gamma=2.0))


if __name__ == "__main__":
    main()

