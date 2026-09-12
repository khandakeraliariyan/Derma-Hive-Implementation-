"""Separate Phase 4 FLOP and batch-one latency measurement."""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
from pathlib import Path
import time
import torch

from convnext_baseline import file_size_megabytes, parameter_counts
from mobilenet_hda import MobileNetHDA
from train_mobilenet_hda import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure Phase 4 efficiency")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("outputs/mobilenet_hda/best_mobilenet_hda.pt"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/mobilenet_hda/efficiency.json"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--warmup-runs", type=int, default=30)
    parser.add_argument("--timed-runs", type=int, default=100)
    return parser.parse_args()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "mobilenet_v3_large_hda_clean_room":
        raise ValueError("Checkpoint is not the Phase 4 MobileNet-HDA model")
    metadata_dim = checkpoint["model_config"]["metadata_dim"]
    model = MobileNetHDA(metadata_dim, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval().to(device)
    image = torch.randn(1, 3, 224, 224, device=device)
    metadata = torch.zeros(1, metadata_dim, device=device)
    with torch.inference_mode():
        for _ in range(args.warmup_runs):
            model(image, metadata)
        sync(device)
        start = time.perf_counter()
        for _ in range(args.timed_runs):
            model(image, metadata)
        sync(device)
    latency_ms = (time.perf_counter() - start) * 1000 / args.timed_runs
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError as exc:
        raise RuntimeError("Install fvcore from requirements_phase3.txt") from exc
    analysis = FlopCountAnalysis(model, (image, metadata))
    # fvcore reports unsupported operators as stderr warnings by default. They
    # are not execution failures. Keep the same fvcore methodology as Phase 3,
    # silence the noisy warnings, and preserve the exclusions in the JSON so
    # the reported FLOP convention is transparent and reproducible.
    analysis.unsupported_ops_warnings(False)
    analysis.uncalled_modules_warnings(False)
    flops = int(analysis.total())
    unsupported_ops = {
        str(operator): int(count)
        for operator, count in analysis.unsupported_ops().items()
    }
    results = {
        **parameter_counts(model), "flops": flops, "gflops": flops / 1e9,
        "checkpoint_size_mb": file_size_megabytes(checkpoint_path),
        "batch_size": 1, "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "warmup_runs": args.warmup_runs, "timed_runs": args.timed_runs,
        "mean_inference_latency_ms": latency_ms,
        "flop_note": (
            "fvcore supported-operator forward-pass count for image+metadata, "
            "batch size 1; activation/elementwise and fused SDPA exclusions "
            "are listed in unsupported_operators"
        ),
        "unsupported_operators": unsupported_ops,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
