"""HAM10000 preparation for our clean-room DermaScanAI Phase 1.

This is our own implementation, not code recovered from the original repository.
"""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from preprocessing import (
    build_image_transform, fit_metadata_preprocessor, metadata_feature_names,
)
from skin_dataset import HAM10000Dataset

REQUIRED_COLUMNS = (
    "lesion_id", "image_id", "dx", "age", "sex", "localization",
)
CLASS_TO_INDEX = {
    "akiec": 0, "bcc": 1, "bkl": 2, "df": 3,
    "mel": 4, "nv": 5, "vasc": 6,
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass
class PreparedHAM10000:
    full_frame: pd.DataFrame
    train_frame: pd.DataFrame
    val_frame: pd.DataFrame
    test_frame: pd.DataFrame
    train_dataset: HAM10000Dataset
    val_dataset: HAM10000Dataset
    test_dataset: HAM10000Dataset
    metadata_feature_names: list[str]
    missing_metadata_counts: dict[str, int]


def locate_metadata_csv(root: Path, csv_path: Path | None = None) -> Path:
    if csv_path is not None:
        path = csv_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Metadata CSV not found: {path}")
        return path
    matches = list(root.rglob("HAM10000_metadata.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one HAM10000_metadata.csv under {root}; found {len(matches)}"
        )
    return matches[0].resolve()


def load_metadata(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    missing = set(REQUIRED_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Metadata CSV is missing columns: {sorted(missing)}")
    frame = frame.loc[:, REQUIRED_COLUMNS].copy()
    for column in ("image_id", "lesion_id"):
        frame[column] = frame[column].astype("string").str.strip()
    frame["dx"] = frame["dx"].astype("string").str.strip().str.lower()
    if frame[["image_id", "lesion_id", "dx"]].isna().any().any():
        raise ValueError("image_id, lesion_id, and dx cannot be missing")
    if frame["image_id"].duplicated().any():
        duplicates = frame.loc[frame["image_id"].duplicated(), "image_id"].tolist()
        raise ValueError(f"Duplicate image_id values: {duplicates[:10]}")
    unknown = sorted(set(frame["dx"]) - set(CLASS_TO_INDEX))
    if unknown:
        raise ValueError(f"Unknown diagnosis codes: {unknown}")
    diagnosis_counts = frame.groupby("lesion_id")["dx"].nunique()
    inconsistent = diagnosis_counts[diagnosis_counts > 1]
    if not inconsistent.empty:
        raise ValueError(
            f"Lesions with multiple diagnoses: {inconsistent.index[:10].tolist()}"
        )
    frame["label"] = frame["dx"].map(CLASS_TO_INDEX).astype("int64")
    return frame


def _choose_case_mirrored_path(paths: list[Path]) -> Path | None:
    """Choose the canonical copy when Kaggle mirrors a folder by case only.

    The Kaggle HAM10000 package contains both ``HAM10000_images_part_1`` and
    ``ham10000_images_part_1`` (and likewise for part 2). On its Linux runtime
    these are separate paths containing identical image IDs. We only collapse
    candidates when their immediate parent names differ solely by letter case;
    other duplicate layouts remain errors.
    """
    if len(paths) < 2:
        return paths[0] if paths else None
    if len(paths) != 2:
        return None
    actual_parent_names = {path.parent.name for path in paths}
    parent_names = {name.casefold() for name in actual_parent_names}
    recognized = {"ham10000_images_part_1", "ham10000_images_part_2"}
    if len(parent_names) != 1 or not parent_names.issubset(recognized):
        return None
    folded_parent = next(iter(parent_names))
    expected_names = {
        folded_parent,
        "HAM10000" + folded_parent[len("ham10000"):],
    }
    if actual_parent_names != expected_names:
        return None
    filenames = {path.name.casefold() for path in paths}
    if len(filenames) != 1:
        return None
    canonical = [
        path for path in paths
        if path.parent.name in {"HAM10000_images_part_1", "HAM10000_images_part_2"}
    ]
    return canonical[0]


def build_image_index(root: Path) -> tuple[dict[str, Path], dict[str, list[str]]]:
    """Find images recursively, including both HAM10000 image-part folders."""
    candidates: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            candidates.setdefault(path.stem, []).append(path.resolve())
    unique: dict[str, Path] = {}
    duplicates: dict[str, list[str]] = {}
    mirrored_count = 0
    for image_id, paths in candidates.items():
        if len(paths) == 1:
            unique[image_id] = paths[0]
            continue
        mirrored = _choose_case_mirrored_path(paths)
        if mirrored is not None:
            unique[image_id] = mirrored
            mirrored_count += 1
        else:
            duplicates[image_id] = [str(path) for path in paths]
    if mirrored_count:
        warnings.warn(
            f"Detected {mirrored_count} image IDs in Kaggle's case-mirrored "
            "HAM10000 folders; using the canonical uppercase folder copies."
        )
    return unique, duplicates


def attach_and_validate_images(
    frame: pd.DataFrame, root: Path, report_path: Path,
    allow_invalid_images: bool = False,
) -> pd.DataFrame:
    index, duplicates = build_image_index(root)
    problems: list[dict[str, str]] = []
    paths: list[str | None] = []
    for image_id in frame["image_id"].astype(str):
        if image_id in duplicates:
            problems.append({"image_id": image_id, "problem": "duplicate_image_id",
                             "details": " | ".join(duplicates[image_id])})
            paths.append(None)
            continue
        path = index.get(image_id)
        if path is None:
            problems.append({"image_id": image_id, "problem": "missing", "details": ""})
            paths.append(None)
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.convert("RGB").load()
            paths.append(str(path))
        except (OSError, ValueError) as exc:
            problems.append({"image_id": image_id, "problem": "corrupted",
                             "details": str(exc)})
            paths.append(None)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(problems, columns=["image_id", "problem", "details"]).to_csv(
        report_path, index=False
    )
    result = frame.copy()
    result["image_path"] = paths
    if problems and not allow_invalid_images:
        counts = pd.Series([item["problem"] for item in problems]).value_counts()
        raise ValueError(f"Invalid images {counts.to_dict()}; see {report_path}")
    if problems:
        warnings.warn(f"Excluding {len(problems)} invalid images; see {report_path}")
        result = result[result["image_path"].notna()].copy()
    return result


def _split(items: pd.DataFrame, test_size: float, seed: int, context: str):
    try:
        return train_test_split(
            items, test_size=test_size, random_state=seed, shuffle=True,
            stratify=items["dx"],
        )
    except ValueError as exc:
        warnings.warn(
            f"Could not fully stratify {context} ({exc}); using a seeded "
            "lesion-level random split."
        )
        return train_test_split(items, test_size=test_size, random_state=seed,
                                shuffle=True)


def create_lesion_splits(
    frame: pd.DataFrame, train_fraction: float = 0.70,
    val_fraction: float = 0.15, test_fraction: float = 0.15, seed: int = 42,
) -> pd.DataFrame:
    if abs(train_fraction + val_fraction + test_fraction - 1) > 1e-9:
        raise ValueError("Split fractions must sum to 1.0")
    if min(train_fraction, val_fraction, test_fraction) <= 0:
        raise ValueError("All split fractions must be positive")
    lesions = frame[["lesion_id", "dx"]].drop_duplicates("lesion_id")
    train, remainder = _split(
        lesions, val_fraction + test_fraction, seed, "train/remainder"
    )
    val, test = _split(
        remainder, test_fraction / (val_fraction + test_fraction), seed + 1,
        "validation/test",
    )
    split_map = {
        **dict.fromkeys(train["lesion_id"], "train"),
        **dict.fromkeys(val["lesion_id"], "val"),
        **dict.fromkeys(test["lesion_id"], "test"),
    }
    result = frame.copy()
    result["split"] = result["lesion_id"].map(split_map)
    if result["split"].isna().any():
        raise RuntimeError("At least one lesion was not assigned a split")
    return result


def assert_no_lesion_overlap(frame: pd.DataFrame) -> None:
    sets = {name: set(frame.loc[frame["split"] == name, "lesion_id"])
            for name in ("train", "val", "test")}
    overlaps = {
        "train_val": sets["train"] & sets["val"],
        "train_test": sets["train"] & sets["test"],
        "val_test": sets["val"] & sets["test"],
    }
    if any(overlaps.values()):
        raise ValueError("lesion_id leakage: " + json.dumps(
            {key: sorted(value)[:10] for key, value in overlaps.items()}
        ))


def load_or_create_splits(
    frame: pd.DataFrame, path: Path, seed: int, train_fraction: float,
    val_fraction: float, test_fraction: float, reuse_existing: bool = True,
) -> pd.DataFrame:
    if reuse_existing and path.exists():
        saved = pd.read_csv(path, dtype={"image_id": "string"})
        if not {"image_id", "lesion_id", "split"}.issubset(saved.columns):
            raise ValueError(f"Malformed saved split file: {path}")
        if set(saved["image_id"]) != set(frame["image_id"]):
            raise ValueError(
                "Saved split IDs differ from this dataset; use --overwrite-splits."
            )
        result = frame.copy()
        result["split"] = result["image_id"].map(
            saved.set_index("image_id")["split"]
        )
    else:
        result = create_lesion_splits(
            frame, train_fraction, val_fraction, test_fraction, seed
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        result[["lesion_id", "image_id", "dx", "label", "split", "image_path"]].to_csv(
            path, index=False
        )
    assert_no_lesion_overlap(result)
    return result


def missing_metadata_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = {"age": int(pd.to_numeric(frame["age"], errors="coerce").isna().sum())}
    for column in ("sex", "localization"):
        values = frame[column].astype("string").str.strip().str.lower()
        counts[column] = int(
            (values.isna() | values.isin(["", "nan", "none", "unknown"])).sum()
        )
    return counts


def prepare_ham10000(
    dataset_root, artifacts_dir="artifacts", csv_path=None, seed: int = 42,
    train_fraction: float = 0.70, val_fraction: float = 0.15,
    test_fraction: float = 0.15, allow_invalid_images: bool = False,
    reuse_existing_splits: bool = True,
) -> PreparedHAM10000:
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root not found: {root}")
    artifacts = Path(artifacts_dir).expanduser().resolve()
    csv = locate_metadata_csv(root, Path(csv_path) if csv_path else None)
    frame = load_metadata(csv)
    missing_counts = missing_metadata_counts(frame)
    frame = attach_and_validate_images(
        frame, root, artifacts / "image_validation_report.csv", allow_invalid_images
    )
    frame = load_or_create_splits(
        frame, artifacts / "ham10000_splits.csv", seed, train_fraction,
        val_fraction, test_fraction, reuse_existing_splits,
    )
    train = frame[frame["split"] == "train"].reset_index(drop=True)
    val = frame[frame["split"] == "val"].reset_index(drop=True)
    test = frame[frame["split"] == "test"].reset_index(drop=True)
    processor = fit_metadata_preprocessor(
        train, artifacts / "metadata_preprocessor.joblib"
    )
    return PreparedHAM10000(
        frame.reset_index(drop=True), train, val, test,
        HAM10000Dataset(train, processor, build_image_transform(True)),
        HAM10000Dataset(val, processor, build_image_transform(False)),
        HAM10000Dataset(test, processor, build_image_transform(False)),
        metadata_feature_names(processor), missing_counts,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare HAM10000")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--csv-path", type=Path)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-invalid-images", action="store_true")
    parser.add_argument("--overwrite-splits", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepared = prepare_ham10000(
        args.dataset_root, args.artifacts_dir, args.csv_path, args.seed,
        allow_invalid_images=args.allow_invalid_images,
        reuse_existing_splits=not args.overwrite_splits,
    )
    print(f"Prepared {len(prepared.full_frame)} images")
    print(prepared.full_frame["split"].value_counts().to_dict())
