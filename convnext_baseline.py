"""Shared Phase 2 ConvNeXt-Tiny baseline utilities.

This is our own clean-room implementation, not original DermaScanAI code.
It intentionally contains no metadata fusion, attention, or Transformer.
"""
from __future__ import annotations

import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

NUM_CLASSES = 7
INDEX_TO_CLASS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_convnext_tiny(pretrained: bool = True) -> nn.Module:
    """Build ConvNeXt-Tiny and replace its ImageNet head with seven outputs."""
    weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = convnext_tiny(weights=weights)
    input_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(input_features, NUM_CLASSES)
    return model


def balanced_class_weights(labels, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """N/(K*n_class) weights calculated from training labels only."""
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes)
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"Training split has no samples for class indices {missing}")
    return torch.tensor(len(labels) / (num_classes * counts), dtype=torch.float32)


def collect_predictions(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    labels, probabilities = [], []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["label"].to(device, non_blocking=True)
            logits = model(images)  # Phase 2 deliberately ignores batch["metadata"].
            loss = criterion(logits, targets)
            total_loss += loss.item() * images.size(0)
            labels.append(targets.cpu().numpy())
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    if not labels:
        raise ValueError("Evaluation loader is empty")
    return (
        total_loss / len(loader.dataset),
        np.concatenate(labels),
        np.concatenate(probabilities),
    )


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = probabilities.argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(precision_score(
            labels, predictions, average="macro", zero_division=0
        )),
        "recall_macro": float(recall_score(
            labels, predictions, average="macro", zero_division=0
        )),
        "macro_f1": float(f1_score(
            labels, predictions, average="macro", zero_division=0
        )),
    }
    try:
        metrics["auc_ovr_macro"] = float(roc_auc_score(
            labels, probabilities, labels=np.arange(NUM_CLASSES),
            multi_class="ovr", average="macro",
        ))
        metrics["auc_note"] = "Macro one-vs-rest AUC from softmax probabilities"
    except ValueError as exc:
        metrics["auc_ovr_macro"] = None
        metrics["auc_note"] = f"Not applicable: {exc}"
    return metrics


def per_class_report(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    return classification_report(
        labels, probabilities.argmax(axis=1), labels=np.arange(NUM_CLASSES),
        target_names=INDEX_TO_CLASS, output_dict=True, zero_division=0,
    )


def save_confusion_matrix(labels, probabilities, output_path: Path) -> np.ndarray:
    matrix = confusion_matrix(
        labels, probabilities.argmax(axis=1), labels=np.arange(NUM_CLASSES)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(NUM_CLASSES), yticks=np.arange(NUM_CLASSES),
        xticklabels=INDEX_TO_CLASS, yticklabels=INDEX_TO_CLASS,
        xlabel="Predicted label", ylabel="True label", title="Test confusion matrix",
    )
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(NUM_CLASSES):
        for column in range(NUM_CLASSES):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center",
                      color="white" if matrix[row, column] > threshold else "black")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return matrix


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def file_size_megabytes(path: Path) -> float:
    return path.stat().st_size / (1024 ** 2)
