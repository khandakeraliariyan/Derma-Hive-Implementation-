# DermaScanAI: Clean-Room Skin Lesion Classification Pipeline

A PyTorch-based skin lesion classification project for the HAM10000 dataset, designed as a clean-room reconstruction and efficiency study of the DermaScanAI / SkinNet-HDA thesis direction.

This repository contains three experimental phases:

- Phase 2: Image-only ConvNeXt-Tiny baseline
- Phase 3: SkinNet-HDA reconstruction using ConvNeXt-Tiny + metadata fusion
- Phase 4: MobileNetV3-Large HDA variant to test a lighter architecture

The project focuses on clinically relevant skin lesion classification across 7 diagnostic classes, with careful dataset splitting, metadata handling, class balancing, and evaluation reporting.

## Project overview

The work is structured around a controlled efficiency experiment rather than a black-box model chase. The core question is:

> Can the SkinNet-HDA-style multimodal classifier be made smaller and faster while retaining competitive classification performance?

The repo includes:

- HAM10000 dataset preparation and validation
- lesion-aware train/validation/test splitting
- preprocessing and metadata feature engineering
- model training for baseline and HDA variants
- checkpointing and training history logging
- evaluation metrics and confusion matrices
- efficiency-oriented analysis scripts

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── convnext_baseline.py
├── dataset_preparation.py
├── preprocessing.py
├── skin_dataset.py
├── skinnet_hda.py
├── mobilenet_hda.py
├── train.py
├── evaluate.py
├── train_skinnet_hda.py
├── evaluate_skinnet_hda.py
├── train_mobilenet_hda.py
├── evaluate_mobilenet_hda.py
├── measure_skinnet_efficiency.py
├── measure_mobilenet_efficiency.py
├── verify_dataset.py
├── verify_skinnet_hda.py
├── verify_mobilenet_hda.py
├── DermaScanAI_Thesis_Complete_Progress_README.md
├── DermaScanAI_SkinNetHDA_Implementation.ipynb
├── artifacts/
├── outputs/
└── ...
```

## Core concepts

### 1. Dataset pipeline
The project uses the HAM10000 dataset and prepares it through a custom pipeline that:

- validates image files and metadata presence
- verifies lesion-level consistency
- splits data by lesion ID to avoid leakages
- maps diagnoses to 7 classes
- preprocesses image and metadata features
- stores reusable split artifacts for repeated experiments

### 2. Baseline model
The baseline is a ConvNeXt-Tiny classifier trained on images only. It provides a clean reference point for evaluating multimodal fusion and efficiency gains.

### 3. SkinNet-HDA reconstruction
The SkinNet-HDA model uses:

- ConvNeXt-Tiny backbone
- SE channel attention
- spatial attention
- Transformer encoder on visual tokens
- metadata fusion at the classifier stage
- focal loss with balanced class weights

### 4. MobileNet-HDA variant
A lighter MobileNetV3-Large adaptation was created to test whether the architecture can be compressed while preserving performance.

## Dataset

This project is designed for the HAM10000 dataset.

Required data:

- HAM10000 image files
- HAM10000 metadata CSV file containing at least:
  - lesion_id
  - image_id
  - dx
  - age
  - sex
  - localization

The basic diagnosis labels are:

- akiec
- bcc
- bkl
- df
- mel
- nv
- vasc

The repository expects a dataset folder structure that can be discovered recursively, and resolves HAM10000 image-part folder naming variations automatically.

## Environment setup

### Python version
Recommended:

```bash
Python 3.10+
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Optional environment notes
If using CUDA, PyTorch should be installed with a compatible CUDA build for your hardware.

## Quick start

### 1. Verify dataset pipeline

```bash
python verify_dataset.py --dataset-root /path/to/HAM10000 --csv-path /path/to/HAM10000_metadata.csv --artifacts-dir artifacts
```

This checks:

- metadata integrity
- class mappings
- lesion split integrity
- image validity
- sample preprocessed tensor shapes
- sample visualization output

### 2. Train the ConvNeXt baseline

```bash
python train.py --dataset-root /path/to/HAM10000 --csv-path /path/to/HAM10000_metadata.csv --artifacts-dir artifacts --output-dir outputs/convnext_tiny
```

### 3. Evaluate the baseline

```bash
python evaluate.py --dataset-root /path/to/HAM10000 --csv-path /path/to/HAM10000_metadata.csv --artifacts-dir artifacts --checkpoint outputs/convnext_tiny/best_convnext_tiny.pt --output-dir outputs/convnext_tiny/test
```

## Training scripts

### Image-only baseline
```bash
python train.py \
  --dataset-root /path/to/HAM10000 \
  --csv-path /path/to/HAM10000_metadata.csv \
  --artifacts-dir artifacts \
  --output-dir outputs/convnext_tiny \
  --epochs 30 \
  --batch-size 32 \
  --learning-rate 3e-4 \
  --seed 42 \
  --device auto
```

### SkinNet-HDA model
```bash
python train_skinnet_hda.py \
  --dataset-root /path/to/HAM10000 \
  --csv-path /path/to/HAM10000_metadata.csv \
  --artifacts-dir artifacts \
  --output-dir outputs/skinnet_hda \
  --epochs 30 \
  --batch-size 32 \
  --learning-rate 1e-4 \
  --seed 42 \
  --device auto
```

### MobileNet-HDA model
```bash
python train_mobilenet_hda.py \
  --dataset-root /path/to/HAM10000 \
  --csv-path /path/to/HAM10000_metadata.csv \
  --artifacts-dir artifacts \
  --output-dir outputs/mobilenet_hda \
  --epochs 30 \
  --batch-size 32 \
  --learning-rate 1e-4 \
  --seed 42 \
  --device auto
```

## Evaluation scripts

### Baseline evaluation
```bash
python evaluate.py --dataset-root /path/to/HAM10000 --csv-path /path/to/HAM10000_metadata.csv --artifacts-dir artifacts --checkpoint outputs/convnext_tiny/best_convnext_tiny.pt --output-dir outputs/convnext_tiny/test
```

### SkinNet-HDA evaluation
```bash
python evaluate_skinnet_hda.py --dataset-root /path/to/HAM10000 --csv-path /path/to/HAM10000_metadata.csv --artifacts-dir artifacts --checkpoint outputs/skinnet_hda/best_skinnet_hda.pt --output-dir outputs/skinnet_hda/test
```

### MobileNet-HDA evaluation
```bash
python evaluate_mobilenet_hda.py --dataset-root /path/to/HAM10000 --csv-path /path/to/HAM10000_metadata.csv --artifacts-dir artifacts --checkpoint outputs/mobilenet_hda/best_mobilenet_hda.pt --output-dir outputs/mobilenet_hda/test
```

## Output artifacts

Each training run generates:

- checkpoint files in the corresponding output directory
- training history CSV logs
- metrics JSON summaries
- confusion matrix images and JSON arrays
- per-class classification reports

Example output structure:

```text
outputs/
├── convnext_tiny/
│   ├── best_convnext_tiny.pt
│   ├── training_history.csv
│   └── test/
│       ├── test_metrics.json
│       ├── classification_report.json
│       ├── confusion_matrix.png
│       └── confusion_matrix.json
├── skinnet_hda/
│   ├── best_skinnet_hda.pt
│   ├── training_history.csv
│   └── test/
│       ├── test_metrics.json
│       ├── classification_report.json
│       ├── confusion_matrix.png
│       └── confusion_matrix.json
└── mobilenet_hda/
    ├── best_mobilenet_hda.pt
    ├── training_history.csv
    └── test/
        ├── test_metrics.json
        ├── classification_report.json
        ├── confusion_matrix.png
        └── confusion_matrix.json
```

## Key metrics

The evaluation scripts report:

- accuracy
- macro precision
- macro recall
- macro F1
- macro one-vs-rest AUC (when applicable)
- total and trainable parameters
- checkpoint file size
- per-class metrics

These metrics are computed with the same logic used for the project’s comparison experiments.

## Notes on implementation

This repository is intentionally a clean-room implementation, not a direct copy of the original paper code. The project documents implementation decisions when the published paper omits low-level details such as:

- exact feature-stage choice
- attention hyperparameters
- Transformer configuration
- metadata encoding design
- training setup details

This is noted explicitly in the project documentation and training checkpoint metadata to keep the work reproducible and academically honest.

## Recommended workflow

1. Prepare the dataset and verify the pipeline.
2. Train the ConvNeXt baseline.
3. Train the SkinNet-HDA reconstruction.
4. Train the MobileNet-HDA lightweight variant.
5. Compare metrics, parameter counts, and checkpoint sizes.
6. Use the generated reports for analysis and thesis writing.

## Important caveats

- The code expects the HAM10000 dataset and metadata CSV to be available locally.
- The split generation is lesion-aware to avoid patient/lesion leakage.
- If you want to regenerate splits, ensure you understand the effect on comparisons.
- For best reproducibility, keep the same seed and dataset version across experiments.

## License

This repository is intended for research and educational use. Please check whether your local institution or project requires a specific licensing model before using the code in a publication or production context.

## References

This project is grounded in:

- HAM10000 dataset
- ConvNeXt-Tiny backbone from torchvision
- SkinNet-HDA-style multimodal architecture
- standard medical image classification metrics and reproducible evaluation practices

## Support

For code or experiment workflow questions, review the training and evaluation scripts first. The project is intentionally modular, so the pipeline can be understood by reading the scripts in order:

- dataset preparation
- model training
- evaluation
- measurement scripts

The file [DermaScanAI_Thesis_Complete_Progress_README.md](DermaScanAI_Thesis_Complete_Progress_README.md) contains a more detailed thesis-oriented project narrative.
