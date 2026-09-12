"""Phase 1 preprocessing for our clean-room DermaScanAI reimplementation.

This is our own implementation, not code recovered from the original repository.
"""
from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
METADATA_COLUMNS = ("age", "sex", "localization")


def build_image_transform(training: bool, image_size: int = 224):
    """Build augmentation followed by ImageNet normalization."""
    operations = [transforms.Resize((image_size, image_size))]
    if training:
        operations += [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(20),
            transforms.RandomAffine(0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.ColorJitter(0.15, 0.15, 0.1, 0.02),
        ]
    operations += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(operations)


def clean_metadata_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize missing-value spellings without learning from the data."""
    cleaned = frame.loc[:, METADATA_COLUMNS].copy()
    cleaned["age"] = pd.to_numeric(cleaned["age"], errors="coerce")
    for column in ("sex", "localization"):
        cleaned[column] = cleaned[column].astype("string").str.strip().str.lower()
        cleaned[column] = cleaned[column].replace(
            {"": pd.NA, "nan": pd.NA, "none": pd.NA, "unknown": pd.NA}
        )
        # sklearn's imputers expect np.nan rather than pandas' scalar pd.NA.
        cleaned[column] = cleaned[column].astype(object).where(
            cleaned[column].notna(), np.nan
        )
    return cleaned


def build_metadata_preprocessor() -> ColumnTransformer:
    """Median-scale age and one-hot encode non-ordinal categorical fields."""
    age = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("age", age, ["age"]),
         ("categorical", categorical, ["sex", "localization"])],
        sparse_threshold=0,
        verbose_feature_names_out=False,
    )


def fit_metadata_preprocessor(training_frame: pd.DataFrame, output_path=None):
    """Fit on training rows only and optionally save the fitted object."""
    preprocessor = build_metadata_preprocessor()
    preprocessor.fit(clean_metadata_frame(training_frame))
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, output_path)
    return preprocessor


def transform_metadata(preprocessor, frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        preprocessor.transform(clean_metadata_frame(frame)), dtype=np.float32
    )


def metadata_feature_names(preprocessor) -> list[str]:
    return list(preprocessor.get_feature_names_out())
