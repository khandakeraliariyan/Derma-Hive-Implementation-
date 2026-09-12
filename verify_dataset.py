"""Verify our clean-room HAM10000 Phase 1 pipeline and save one sample plot.

This is our own implementation, not code recovered from the original repository.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from dataset_preparation import CLASS_TO_INDEX, assert_no_lesion_overlap, prepare_ham10000
from preprocessing import IMAGENET_MEAN, IMAGENET_STD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the HAM10000 pipeline")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--csv-path", type=Path)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-invalid-images", action="store_true")
    parser.add_argument("--overwrite-splits", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared = prepare_ham10000(
        args.dataset_root, args.artifacts_dir, args.csv_path, args.seed,
        allow_invalid_images=args.allow_invalid_images,
        reuse_existing_splits=not args.overwrite_splits,
    )
    frame = prepared.full_frame
    assert_no_lesion_overlap(frame)
    print(f"Total samples: {len(frame)}")
    print(f"Unique lesions: {frame['lesion_id'].nunique()}")
    print("Class mapping:", CLASS_TO_INDEX)
    print("Class distribution:", frame["dx"].value_counts().sort_index().to_dict())
    print("Missing metadata counts:", prepared.missing_metadata_counts)
    print("Split sizes:", {
        "train": len(prepared.train_frame), "val": len(prepared.val_frame),
        "test": len(prepared.test_frame),
    })
    lesions = {name: set(frame.loc[frame["split"] == name, "lesion_id"])
               for name in ("train", "val", "test")}
    print("Lesion overlap counts (all must be 0):", {
        "train_val": len(lesions["train"] & lesions["val"]),
        "train_test": len(lesions["train"] & lesions["test"]),
        "val_test": len(lesions["val"] & lesions["test"]),
    })
    print("Metadata features:", prepared.metadata_feature_names)
    sample = prepared.val_dataset[0]
    print("Image tensor shape:", tuple(sample["image"].shape))
    print("Metadata tensor shape:", tuple(sample["metadata"].shape))
    print("Label:", int(sample["label"]), f"(dx={prepared.val_frame.iloc[0]['dx']})")
    print("Sample image_id:", sample["image_id"])
    print("Sample lesion_id:", sample["lesion_id"])

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image = (sample["image"].cpu() * std + mean).clamp(0, 1)
    plt.figure(figsize=(5, 5))
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.title(
        f"{sample['image_id']} | {prepared.val_frame.iloc[0]['dx']} | "
        f"label={int(sample['label'])}"
    )
    plt.axis("off")
    plt.tight_layout()
    output = args.artifacts_dir.resolve() / "verification_sample.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print("Visualized sample saved to:", output)
    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
