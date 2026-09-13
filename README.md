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

## Research motivation and problem statement

Skin lesion classification is a high-impact medical imaging task, but model development in this area is often constrained by two competing priorities:

1. achieving strong predictive performance, and
2. keeping the model practical for deployment in a real clinical or edge-computing environment.

This project addresses that trade-off by studying whether a multimodal SkinNet-HDA-style architecture can be made lighter without sacrificing diagnostic quality. The repository follows a clean-room experimental framework in which each design choice is explicitly documented and compared to a simpler baseline.

The central thesis question is not simply "Which model is most accurate?" but rather:

- How much performance is retained after reducing model complexity?
- Is metadata fusion worth the added complexity?
- Does a lighter MobileNet-based backbone maintain stable classification behavior?
- What is the cost of the architecture in terms of parameters, checkpoint size, and runtime efficiency?

## Experimental phases

### Phase 1: dataset preparation and validation
This phase builds the data pipeline for HAM10000 by:

- validating metadata integrity
- checking for missing images and malformed records
- ensuring there is no lesion-level overlap across splits
- creating consistent train/validation/test partitions
- standardizing feature preprocessing for age, sex, and lesion localization

This step is essential because leakage between splits would invalidate the final comparison between architectures.

### Phase 2: ConvNeXt-Tiny baseline
The baseline model is an image-only ConvNeXt-Tiny classifier trained directly on dermoscopic images. This phase acts as a reference model and helps measure the value of multimodal fusion and attention-based modules.

### Phase 3: SkinNet-HDA reconstruction
The SkinNet-HDA variant follows a multimodal design:

- image features are extracted with a ConvNeXt-Tiny backbone
- channel and spatial attention improve the visual response
- visual tokens are passed through a lightweight Transformer encoder
- metadata such as age, sex, and lesion location are encoded and fused
- a final classifier produces the 7-class lesion prediction

This is the primary architecture of interest for the project.

### Phase 4: MobileNetV3-Large efficiency variant
A second model swaps the backbone with MobileNetV3-Large to investigate whether similar or better efficiency can be achieved with a much smaller architecture. This is an important step for thesis analysis because a lightweight model can be more viable for deployment and experimentation.

## Architecture summary

The project includes several complementary design patterns:

- ConvNeXt backbone for strong visual feature extraction
- SE channel attention to recalibrate learned feature channels
- spatial attention to emphasize useful regions in the image
- Transformer token mixing for global context aggregation
- metadata fusion through concatenation with image features
- balanced class-weighted focal loss to handle class imbalance

The code is intentionally modular so each component can be examined, replaced, or compared independently.

## Data and preprocessing strategy

The dataset preprocessing workflow includes:

- resizing all input images to 224 x 224
- applying ImageNet normalization to match pretrained backbone assumptions
- using augmentation during training for improved generalization
- imputing missing metadata values with clinically sensible defaults
- standardizing numerical metadata like age
- one-hot encoding categorical metadata such as sex and localization

This standardization is critical because metadata is later fused with image features and should be numerically consistent across training and evaluation.

## Class imbalance handling

HAM10000 is a highly imbalanced dataset. Some classes such as melanoma and vascular lesions appear far less often than others like melanocytic nevi. To handle this, the project uses:

- stratified train/validation/test splits where possible
- lesion-level split isolation to prevent overlap
- balanced class weights derived from the training labels
- focal loss rather than standard cross-entropy in the multimodal models

These choices help keep model training stable and reduce the risk of the model disproportionately favoring majority classes.

## Training workflow

The standard flow is:

1. verify the dataset and save split artifacts
2. train the baseline model
3. train the multimodal SkinNet-HDA model
4. train the MobileNetHDA efficiency model
5. evaluate each checkpoint on the test set
6. compare metrics, parameter count, and checkpoint size

This workflow makes each experiment reproducible and directly comparable in the same environment.

## Evaluation and reporting

The evaluation scripts produce:

- test loss values
- overall accuracy
- macro precision/recall/F1
- macro one-vs-rest AUC where applicable
- confusion matrices
- per-class classification reports
- parameter counts and model size statistics

These outputs are designed for both research analysis and thesis reporting.

## Recommended experimental sequence

For a clean comparison, follow this order:

1. Run dataset verification.
2. Train the baseline image-only ConvNeXt model.
3. Train and evaluate SkinNet-HDA.
4. Train and evaluate the MobileNetV3-Large variant.
5. Compare the models using macro F1, parameter count, file size, and validation behavior.
6. Record the best checkpoint and the final evaluation outputs.

This approach makes the efficiency claim stronger because all models are tested under the same dataset split and preprocessing regime.

## Common pitfalls

- Using the same lesion in both training and test sets
- forgetting to set the proper dataset root and metadata CSV path
- running with a mismatched PyTorch/CUDA setup
- training without checking that the metadata preprocessing matches the saved checkpoint
- comparing models trained on different split artifacts

## Troubleshooting

### Missing dataset file
If the dataset or metadata CSV is not found, verify that:

- the HAM10000 directory is correctly downloaded
- the CSV file exists and includes the expected columns
- the folder structure contains the image part directories

### Split leakage errors
If lesion overlap appears across train/validation/test splits, the pipeline will warn or fail. This usually indicates the dataset was not split by lesion_id correctly.

### CUDA issues
If CUDA is unavailable or incompatible, run the scripts with:

```bash
--device cpu
```

### Model checkpoint mismatch
If evaluation fails because a checkpoint is incompatible, ensure the checkpoint was produced by the same architecture and metadata configuration that the evaluation script expects.

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
