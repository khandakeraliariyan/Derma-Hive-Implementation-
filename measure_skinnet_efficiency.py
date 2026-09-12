"""Separate FLOP and inference-latency measurement for trained Phase 3 models."""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
from pathlib import Path
import time
import torch

from convnext_baseline import file_size_megabytes, parameter_counts
from skinnet_hda import SkinNetHDA
from train_skinnet_hda import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure Phase 3 efficiency")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("outputs/skinnet_hda/best_skinnet_hda.pt"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/skinnet_hda/efficiency.json"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--warmup-runs", type=int, default=30)
    parser.add_argument("--timed-runs", type=int, default=100)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata_dim = checkpoint["model_config"]["metadata_dim"]
    model = SkinNetHDA(metadata_dim, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval().to(device)
    image = torch.randn(1, 3, 224, 224, device=device)
    metadata = torch.zeros(1, metadata_dim, device=device)

    with torch.inference_mode():
        for _ in range(args.warmup_runs):
            model(image, metadata)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(args.timed_runs):
            model(image, metadata)
        synchronize(device)
    latency_ms = (time.perf_counter() - start) * 1000 / args.timed_runs

    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError as exc:
        raise RuntimeError(
            "FLOP measurement requires fvcore: pip install -r requirements_phase3.txt"
        ) from exc
    flops = int(FlopCountAnalysis(model, (image, metadata)).total())
    results = {
        **parameter_counts(model),
        "flops": flops,
        "gflops": flops / 1e9,
        "checkpoint_size_mb": file_size_megabytes(checkpoint_path),
        "batch_size": 1,
        "device": str(device),
        "warmup_runs": args.warmup_runs,
        "timed_runs": args.timed_runs,
        "mean_inference_latency_ms": latency_ms,
        "flop_note": "fvcore forward-pass count for image+metadata, batch size 1",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

