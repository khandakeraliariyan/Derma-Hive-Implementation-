# DermaScanAI Thesis Project --- Complete Progress & Supervisor Meeting README

## 1. Project Overview

**Working thesis direction:** Computationally Efficient Multimodal Deep
Learning for Skin Lesion Classification

**Current experimental question:**

> Can a SkinNet-HDA-style multimodal skin-lesion classifier be made
> substantially smaller and faster while retaining competitive
> diagnostic classification performance?

The project began as a broader skin-disease/skin-lesion detection thesis
idea involving deep learning, image processing, multimodal clinical
metadata, explainability, fairness, and deployment. As the semester
deadline approached, the work was deliberately narrowed into a
**controlled, reproducible architecture-efficiency experiment**.

The final experimental scope became:

1.  Build a reliable HAM10000 dataset pipeline.
2.  Establish a plain ConvNeXt-Tiny baseline.
3.  Reconstruct the published SkinNet-HDA architecture as faithfully as
    possible.
4.  Identify its computational cost.
5.  Replace its ConvNeXt-Tiny backbone with MobileNetV3-Large.
6.  Measure whether the new model becomes smaller/faster and how much
    predictive performance is retained.

The implementation and experimentation phases are now **closed and
frozen**.

------------------------------------------------------------------------

# 2. How the Thesis Idea Evolved

## 2.1 Initial Idea

The initial thesis direction was broadly:

> Early skin disease / skin lesion detection using image processing and
> deep learning.

Several possible research gaps were considered:

-   image-only models ignoring patient metadata;
-   limited use of age, sex, and lesion location;
-   explainability of predictions;
-   skin-tone/fairness evaluation;
-   external validation;
-   CNN limitations in modeling global image context;
-   Transformer computational cost;
-   deployment efficiency;
-   parameter count and inference speed.

An early conceptual architecture, referred to as **HMX-Derm**, included:

-   image preprocessing;
-   CNN + Transformer visual features;
-   clinical metadata fusion;
-   explainability;
-   fairness analysis;
-   external validation;
-   deployment evaluation.

This was intentionally reduced because completing all of those
directions in one semester would not allow a controlled experimental
study.

------------------------------------------------------------------------

# 3. Supervisor-Oriented Research Direction

The semester objective was refined around a practical definition of
**improvement**.

Improvement does **not** have to mean only higher accuracy.

A model can be improved through a better trade-off involving:

-   predictive performance;
-   parameter count;
-   FLOPs;
-   inference latency;
-   checkpoint/model size.

This led to the final research direction:

> **Investigate computationally efficient multimodal deep-learning
> architectures for skin-lesion classification.**

### Final problem statement

> **Can SkinNet-HDA be made smaller and faster while preserving
> competitive classification performance?**

This framing allowed the project to investigate efficiency honestly
rather than attempting only to beat a published headline accuracy.

------------------------------------------------------------------------

# 4. Base Paper Selection

The selected paper was:

**Murali, P. & Mazumder, D. H. (2026).\
"DermaScanAI: an explainable hybrid deep learning framework for
automated skin lesion classification using dual attention and metadata
fusion."\
Scientific Reports.**

The paper describes a model called **SkinNet-HDA**.

Its high-level architecture includes:

-   ConvNeXt-Tiny visual feature extraction;
-   SE-style channel attention;
-   spatial attention;
-   lightweight Transformer processing;
-   clinical metadata;
-   image-metadata fusion;
-   seven-class classification;
-   focal loss;
-   explainability components such as Grad-CAM++ and SHAP.

The paper was attractive as a thesis base because it combines:

1.  modern CNN features;
2.  attention;
3.  Transformer global-context modeling;
4.  clinical metadata;
5.  an explicit future need for efficiency/compression.

------------------------------------------------------------------------

# 5. Important Reproducibility Finding

The authors provide a public GitHub repository, but inspection showed
that the public repository does **not contain a complete runnable
implementation of the defining SkinNet-HDA architecture**.

Several important source files were placeholders or incomplete.

Therefore, this project does **not** claim:

> "We reproduced the authors' official code."

The academically correct description is:

> **SkinNet-HDA clean-room reconstruction based on the published
> high-level architecture, with explicitly documented implementation
> decisions for undisclosed details.**

This distinction is important for the supervisor meeting and final
thesis.

------------------------------------------------------------------------

# 6. Problems Found in the Source Paper

The paper provides the overall architecture but does not provide enough
low-level information for exact reproduction.

Examples of missing or ambiguous details include:

-   exact ConvNeXt feature stage;
-   Transformer embedding configuration;
-   number of Transformer layers;
-   number of attention heads;
-   Transformer feed-forward dimension;
-   positional encoding;
-   Transformer dropout;
-   SE reduction ratio;
-   spatial-attention convolution kernel;
-   some metadata encoding details;
-   some preprocessing details.

There are also reported inconsistencies, including:

-   PyTorch vs TensorFlow/Keras implementation descriptions;
-   different descriptions of sex encoding;
-   multiple dataset-splitting descriptions;
-   different early-stopping descriptions;
-   different reported headline performance values;
-   an unexplained 8,192-dimensional representation;
-   reported **9.6M parameters / 1.4 GFLOPs**, which is difficult to
    reconcile with a normal ConvNeXt-Tiny backbone of roughly 28M
    parameters.

Therefore, all undisclosed values required to make the model runnable
were explicitly documented as:

> **OUR IMPLEMENTATION DECISION --- NOT CLAIMED BY THE PAPER**

------------------------------------------------------------------------

# 7. Dataset

The experiments use the classic **HAM10000** dataset.

## Dataset properties

-   **10,015 dermoscopic images**
-   **7 diagnostic classes**
-   **7,470 unique lesions**

Classes:

  Code      Diagnosis category
  --------- -----------------------------------------------
  `akiec`   Actinic keratoses / intraepithelial carcinoma
  `bcc`     Basal cell carcinoma
  `bkl`     Benign keratosis-like lesions
  `df`      Dermatofibroma
  `mel`     Melanoma
  `nv`      Melanocytic nevi
  `vasc`    Vascular lesions

Available metadata includes:

-   age;
-   sex;
-   anatomical localization;
-   image ID;
-   lesion ID;
-   diagnosis.

------------------------------------------------------------------------

# 8. Leakage-Safe Experimental Split

An important methodological improvement over a simple random image split
is that the experiment splits the dataset by **`lesion_id`**.

Multiple HAM10000 images may belong to the same physical lesion.

If one image from a lesion enters training and another image of the same
lesion enters testing, the evaluation can become artificially
optimistic.

Therefore, the project uses a fixed lesion-level split.

## Frozen split

  Partition          Images
  ------------ ------------
  Train               7,002
  Validation          1,508
  Test                1,505
  **Total**      **10,015**

Unique lesions:

**7,470**

The same exact split is reused for every experiment.

It is never regenerated during model training.

------------------------------------------------------------------------

# 9. Phase 1 --- Dataset and Preprocessing Pipeline

Phase 1 implemented and verified the complete real-data pipeline.

Main files included:

-   `dataset_preparation.py`
-   `preprocessing.py`
-   `skin_dataset.py`
-   `verify_dataset.py`

## Image processing

Images are:

1.  validated;
2.  converted to RGB;
3.  resized to **224 × 224**;
4.  augmented only during training;
5.  ImageNet-normalized.

Training augmentation includes controlled:

-   horizontal/vertical flipping;
-   rotation;
-   translation/scaling;
-   color jitter.

Validation/test data do not use random augmentation.

## Metadata processing

Metadata processing is fitted using **training data only**.

### Age

-   missing values handled using training-derived statistics;
-   numerical scaling applied.

### Sex

-   categorical encoding;
-   unknown/missing values safely handled.

### Localization

-   categorical encoding;
-   avoids treating anatomical locations as ordinal numbers.

The final frozen real-data metadata vector has dimension:

``` text
[B, 7]
```

## Phase 1 artifacts

Examples:

``` text
artifacts/
├── ham10000_splits.csv
├── metadata_preprocessor.joblib
├── image_validation_report.csv
└── verification_sample.png
```

------------------------------------------------------------------------

# 10. Real Kaggle Dataset Issue Found and Fixed

During real-data verification, all 10,015 images were initially reported
as duplicate IDs.

This was **not a HAM10000 data problem**.

The Kaggle dataset contained duplicated image-directory representations,
and the original recursive image discovery found both copies.

The loader was corrected to select one canonical image source while
still detecting genuine duplicate IDs.

This is an example of why real-data verification was performed before
model training.

------------------------------------------------------------------------

# 11. Compute Environment

Training was performed on **Kaggle GPU notebooks**.

The experiments used Kaggle GPUs, including Tesla T4 during setup and
Tesla P100 for the final efficiency comparisons.

Important experimental artifacts were saved after every run:

-   checkpoints;
-   training histories;
-   metrics JSON;
-   classification reports;
-   confusion matrices;
-   efficiency JSON;
-   notebook versions.

CUDA reproducibility configuration and random seeds were applied.

PyTorch nevertheless reported that its memory-efficient Transformer
attention backward implementation can be nondeterministic. Therefore,
exact bit-for-bit reruns are not guaranteed.

This limitation is documented rather than hidden.

------------------------------------------------------------------------

# 12. Phase 2 --- Plain ConvNeXt-Tiny Baseline

Before reconstructing SkinNet-HDA, a simple image-only baseline was
trained.

## Architecture

``` text
RGB image
    ↓
Pretrained ConvNeXt-Tiny
    ↓
Replace original classifier
    ↓
7-class classifier
```

No:

-   metadata;
-   Transformer;
-   SE attention;
-   spatial attention;
-   MobileNet;
-   Grad-CAM;
-   SHAP.

This provided a controlled sanity baseline.

## Training

Key settings included:

-   pretrained torchvision ConvNeXt-Tiny;
-   AdamW;
-   weighted cross-entropy;
-   class weights calculated using training data only;
-   maximum 30 epochs;
-   best checkpoint selected by validation Macro-F1.

## Phase 2 results

Best validation Macro-F1:

**0.72416 at epoch 30**

### Test performance

  Metric                      Result
  ----------------- ----------------
  Accuracy                **82.59%**
  Macro-F1               **0.68677**
  Macro precision        **0.70470**
  Macro recall           **0.71030**
  Multiclass AUC         **0.96518**
  Test loss              **0.76532**
  Parameters          **27,825,511**
  Checkpoint size      **106.21 MB**

### Per-class F1

  Class           F1
  --------- --------
  `akiec`     0.4571
  `bcc`       0.7391
  `bkl`       0.7256
  `df`        0.6667
  `mel`       0.5211
  `nv`        0.9222
  `vasc`      0.7755

This result demonstrated why accuracy alone is insufficient: the
dominant `nv` class performs very strongly while difficult/minority
classes have substantially lower F1.

------------------------------------------------------------------------

# 13. Phase 3 --- SkinNet-HDA Clean-Room Reconstruction

After the baseline was validated, the SkinNet-HDA architecture was
reconstructed from the paper.

## Reconstructed architecture

``` text
Image
[B,3,224,224]
        ↓
Pretrained ConvNeXt-Tiny
        ↓
[B,768,7,7]
        ↓
SE Channel Attention
768 → 48 → 768
        ↓
Spatial Attention
        ↓
[B,768,7,7]
        ↓
Flatten spatial positions
        ↓
[B,49,768]
        ↓
Learnable positional embedding
        ↓
Lightweight Transformer
        ↓
[B,49,768]
        ↓
Mean pooling
        ↓
Visual feature [B,768]

Metadata [B,7]
        ↓
Concatenation
        ↓
[B,775]
        ↓
Linear classifier
        ↓
7 logits
```

## Clean-room implementation decisions

Because the paper did not specify all required values, the
reconstruction uses documented decisions:

### SE attention

-   reduction ratio = **16**
-   `768 → 48 → 768`

### Spatial attention

-   standard channel average + channel maximum maps;
-   7 × 7 convolution;
-   sigmoid mask.

### Transformer

-   `d_model = 768`
-   1 encoder layer
-   8 attention heads
-   feed-forward dimension = 1536
-   dropout = 0.1
-   GELU activation
-   pre-layer normalization
-   batch-first representation
-   learnable positional embedding `[1,49,768]`

### Metadata fusion

No extra metadata MLP was introduced.

The pooled visual vector is concatenated directly with the existing
Phase 1 metadata vector.

------------------------------------------------------------------------

# 14. Phase 3 Training Protocol

Phase 3 uses:

  Setting             Value
  ------------------- -----------------------------------------
  Optimizer           AdamW
  Learning rate       `1e-4`
  Minimum LR          `1e-6`
  Scheduler           Cosine annealing
  Weight decay        `1e-4`
  Loss                Multiclass focal loss
  Gamma               `2`
  Alpha               Reciprocal-frequency from training data
  Batch size          32
  Maximum epochs      30
  Early stopping      Patience 5
  Checkpoint metric   Validation Macro-F1
  Seed                42

------------------------------------------------------------------------

# 15. Phase 3 Verification

Before training, complete forward and backward propagation was verified.

``` text
image                  (2, 3, 224, 224)
convnext               (2, 768, 7, 7)
se                     (2, 768, 7, 7)
spatial_attention      (2, 768, 7, 7)
tokens                 (2, 49, 768)
transformer            (2, 49, 768)
pooled_image_feature   (2, 768)
metadata               (2, 7)
fused_feature          (2, 775)
logits                 (2, 7)
backward               PASS
```

------------------------------------------------------------------------

# 16. Phase 3 Results

Best validation Macro-F1:

**0.73484 at epoch 23**

Early stopping:

**epoch 28**

### Test results

  Metric                              Result
  ---------------------- -------------------
  Accuracy                       **82.392%**
  Macro-F1                      **0.712856**
  Macro precision               **0.732711**
  Macro recall                  **0.719924**
  Multiclass AUC                **0.956055**
  Test focal loss               **0.942675**
  Parameters                  **32,663,422**
  FLOPs                    **4.7013 GFLOPs**
  P100 batch-1 latency          **8.969 ms**
  Checkpoint                   **124.67 MB**

### Per-class F1

  Class           F1
  --------- --------
  `akiec`     0.4957
  `bcc`       0.7166
  `bkl`       0.6851
  `df`        0.7333
  `mel`       0.5460
  `nv`        0.9196
  `vasc`      0.8936

------------------------------------------------------------------------

# 17. What SkinNet-HDA Improved

Compared with the plain ConvNeXt baseline:

  Metric              ConvNeXt   SkinNet-HDA         Change
  ----------------- ---------- ------------- --------------
  Accuracy              82.59%       82.392%       −0.20 pp
  Macro-F1             0.68677      0.712856   **+0.02609**
  Macro precision      0.70470      0.732711   **+0.02801**
  Macro recall         0.71030      0.719924   **+0.00962**
  AUC                  0.96518      0.956055       −0.00912

The important finding is:

> SkinNet-HDA did not materially increase overall accuracy, but it
> improved balanced classification performance as measured by Macro-F1,
> macro precision, and macro recall.

Several difficult/minority classes also improved:

-   `akiec`: 0.4571 → **0.4957**
-   `df`: 0.6667 → **0.7333**
-   `mel`: 0.5211 → **0.5460**
-   `vasc`: 0.7755 → **0.8936**

However, the improved balanced performance came with additional
computational cost.

This created the motivation for Phase 4.

------------------------------------------------------------------------

# 18. Phase 4 --- Proposed MobileNetV3-Large HDA

## Objective

The Phase 4 hypothesis was:

> Can we retain the SkinNet-HDA architecture concept while replacing its
> computationally expensive ConvNeXt-Tiny backbone with a lightweight
> MobileNetV3-Large backbone?

Only the backbone was intentionally replaced.

The following concepts remained:

-   channel attention;
-   spatial attention;
-   Transformer;
-   metadata;
-   fusion;
-   classifier;
-   focal loss;
-   optimizer;
-   scheduler;
-   split;
-   preprocessing;
-   evaluation protocol.

------------------------------------------------------------------------

# 19. Required MobileNet Interface Adaptation

Torchvision MobileNetV3-Large was directly inspected using a 224 × 224
input.

Its feature output was:

``` text
[B, 960, 7, 7]
```

ConvNeXt produced:

``` text
[B, 768, 7, 7]
```

Therefore, the downstream dimensions had to change.

These changes are explicitly documented as:

> **REQUIRED BACKBONE-INTERFACE ADAPTATION --- not an additional
> proposed module.**

No projection layer was introduced merely to force MobileNet features
into 768 dimensions.

------------------------------------------------------------------------

# 20. Proposed MobileNet-HDA Architecture

``` text
Image
[B,3,224,224]
        ↓
Pretrained MobileNetV3-Large
        ↓
[B,960,7,7]
        ↓
SE Channel Attention
960 → 60 → 960
        ↓
Spatial Attention
        ↓
[B,960,7,7]
        ↓
49 spatial tokens
        ↓
[B,49,960]
        ↓
Transformer
d_model = 960
8 heads
FFN = 1920
        ↓
Mean pooling
        ↓
[B,960]

Metadata
[B,7]
        ↓
Concatenation
        ↓
[B,967]
        ↓
Classifier
        ↓
7 logits
```

------------------------------------------------------------------------

# 21. Phase 4 Verification

Before training:

``` text
image                  (2, 3, 224, 224)
mobilenet              (2, 960, 7, 7)
se                     (2, 960, 7, 7)
spatial_attention      (2, 960, 7, 7)
tokens                 (2, 49, 960)
transformer            (2, 49, 960)
pooled_image_feature   (2, 960)
metadata               (2, 7)
fused_feature          (2, 967)
logits                 (2, 7)
backward               PASS
```

Parameter count before training:

**10,525,530**

This immediately represented approximately a **67.8% parameter
reduction** relative to Phase 3.

------------------------------------------------------------------------

# 22. Phase 4 Results

Best validation Macro-F1:

**0.63114 at epoch 5**

Early stopping:

**epoch 10**

### Test results

  Metric                              Result
  ---------------------- -------------------
  Accuracy                       **74.485%**
  Macro-F1                      **0.580893**
  Macro precision               **0.542266**
  Macro recall                  **0.649338**
  Multiclass AUC                **0.931670**
  Test focal loss               **0.619819**
  Parameters                  **10,525,530**
  FLOPs                    **0.5854 GFLOPs**
  P100 batch-1 latency          **6.188 ms**
  Checkpoint size               **40.36 MB**

### Per-class F1

  Class           F1
  --------- --------
  `akiec`     0.4553
  `bcc`       0.6389
  `bkl`       0.5429
  `df`        0.3077
  `mel`       0.4418
  `nv`        0.8797
  `vasc`      0.8000

------------------------------------------------------------------------

# 23. Final Three-Model Comparison

  Metric              Plain ConvNeXt    SkinNet-HDA   Proposed MobileNet-HDA
  ----------------- ---------------- -------------- ------------------------
  Accuracy                **82.59%**        82.392%                  74.485%
  Macro-F1                   0.68677   **0.712856**                 0.580893
  Macro precision            0.70470   **0.732711**                 0.542266
  Macro recall               0.71030   **0.719924**                 0.649338
  AUC                    **0.96518**       0.956055                 0.931670
  Parameters                  27.83M         32.66M               **10.53M**
  FLOPs                 Not measured        4.7013G              **0.5854G**
  P100 latency          Not measured       8.969 ms             **6.188 ms**
  Checkpoint               106.21 MB      124.67 MB             **40.36 MB**

------------------------------------------------------------------------

# 24. What We Improved

The proposed MobileNet-HDA achieved major efficiency improvements
relative to reconstructed SkinNet-HDA. + add attention model + add attention model (cnn architecture)

## Parameter reduction

``` text
32.66M → 10.53M
```

**67.8% fewer parameters**

## FLOP reduction

``` text
4.7013G → 0.5854G
```

**87.5% fewer counted FLOPs**

## Checkpoint reduction

``` text
124.67 MB → 40.36 MB
```

**67.6% smaller**

## P100 latency reduction

``` text
8.969 ms → 6.188 ms
```

**31.0% lower latency**

Both Phase 3 and Phase 4 latency measurements used a **Tesla P100**,
making this a same-hardware comparison.

------------------------------------------------------------------------

# 25. What We Did NOT Improve

The proposed model is **not performance-equivalent** to reconstructed
SkinNet-HDA.

Accuracy:

``` text
82.392% → 74.485%
```

Reduction:

**7.91 percentage points**

Macro-F1:

``` text
0.712856 → 0.580893
```

Reduction:

**0.1320**

The lightweight model especially weakened several difficult/minority
classes.

Examples:

``` text
df:  0.7333 → 0.3077
mel: 0.5460 → 0.4418
bkl: 0.6851 → 0.5429
bcc: 0.7166 → 0.6389
```

Therefore, it would be academically incorrect to claim that
MobileNet-HDA is simply "better."

------------------------------------------------------------------------

# 26. Final Thesis Finding

The central finding is an **efficiency--performance trade-off**.

The final defensible conclusion is:

> **MobileNet-HDA reduced parameters by 67.8%, supported-operator FLOPs
> by 87.5%, checkpoint size by 67.6%, and P100 latency by 31.0%. These
> gains were accompanied by a 7.91 percentage-point accuracy reduction
> and a 0.1320 decrease in Macro-F1. The proposed architecture is
> therefore substantially more computationally efficient, but not a
> performance-equivalent replacement for reconstructed SkinNet-HDA.**

This is the primary thesis result.

------------------------------------------------------------------------

# 27. Why This Is Still a Valid Research Result

The original research question did not require the proposed model to
automatically outperform SkinNet-HDA on every metric.

The experiment asked whether the architecture could be made more
computationally efficient **while measuring the predictive cost of doing
so**.

The experiment answers that question clearly:

-   **Yes**, the architecture can be made dramatically smaller and
    computationally cheaper.
-   **No**, this aggressive backbone replacement does not preserve
    balanced classification performance sufficiently under the frozen
    protocol.

That negative/partial result is scientifically meaningful.

No post-hoc hyperparameter changes were made after seeing the Phase 4
result merely to obtain a better-looking thesis result.

------------------------------------------------------------------------

# 28. FLOP Measurement Caveat

FLOPs were measured using `fvcore`.

They should be described as:

> **fvcore supported-operator forward-pass FLOP estimates**

Some operations are not counted by the tool.

The efficiency measurement code was updated to:

-   suppress misleading unsupported-operator warning noise;
-   record excluded operators;
-   explicitly document the FLOP-counting convention.

The FLOP result therefore should **not** be presented as an exact
hardware instruction count.

------------------------------------------------------------------------

# 29. Reproducibility Decisions

To maintain experimental integrity:

-   the same lesion-level split was used throughout;
-   preprocessing was frozen;
-   metadata processing was frozen;
-   seed 42 was used;
-   Phase 3 and Phase 4 used the same main training protocol;
-   best checkpoints were selected by validation Macro-F1;
-   test data were not used for model selection;
-   no post-hoc tuning was performed to rescue Phase 4;
-   model checkpoints and notebook versions were frozen after
    completion.

------------------------------------------------------------------------

# 30. Frozen Experiments

The implementation and experimentation phases are now officially closed.

The following models are frozen:

## Model 1

**Plain ConvNeXt-Tiny baseline**

Purpose:

> Establish image-only baseline performance.

## Model 2

**SkinNet-HDA clean-room reconstruction**

Purpose:

> Establish the paper-guided multimodal hybrid baseline.

## Model 3

**Proposed MobileNetV3-Large HDA**

Purpose:

> Evaluate aggressive computational-efficiency improvement through
> backbone replacement.

No further:

-   retraining;
-   tuning;
-   architecture changes;
-   split changes;
-   preprocessing changes

should be performed unless experimentation is explicitly reopened.

------------------------------------------------------------------------

# 31. Important Experimental Evidence to Preserve

Do not overwrite or delete:

### Dataset artifacts

``` text
ham10000_splits.csv
metadata_preprocessor.joblib
image_validation_report.csv
```

### Phase 2

``` text
best_convnext_tiny.pt
training_history.csv
test_metrics.json
classification_report.json
confusion_matrix.json
confusion_matrix.png
```

### Phase 3

``` text
best_skinnet_hda.pt
training_history.csv
efficiency.json
test/
```

### Phase 4

``` text
best_mobilenet_hda.pt
training_history.csv
efficiency.json
test/
```

Also preserve:

-   Kaggle notebook versions;
-   GPU/environment logs;
-   source-code versions;
-   dependency files;
-   verification logs.

------------------------------------------------------------------------

# 32. Current Thesis Status

  Work                           Status
  ------------------------------ -------------------------------
  Research direction             ✅ Complete
  Base-paper selection           ✅ Complete
  Paper architecture audit       ✅ Complete
  Repository audit               ✅ Complete
  HAM10000 setup                 ✅ Complete
  Leakage-safe split             ✅ Complete
  Preprocessing pipeline         ✅ Complete
  ConvNeXt baseline              ✅ Complete
  SkinNet-HDA reconstruction     ✅ Complete
  SkinNet-HDA training           ✅ Complete
  MobileNet-HDA implementation   ✅ Complete
  MobileNet-HDA training         ✅ Complete
  Performance comparison         ✅ Complete
  Efficiency comparison          ✅ Complete
  Experimental phase             ✅ **FROZEN**
  Final thesis report            🔄 Packaging/final formatting
  Final presentation             🔄 Remaining
  Viva preparation               🔄 Remaining

------------------------------------------------------------------------

# 33. Limitations to Discuss With Supervisor

The final thesis should openly acknowledge:

1.  Only HAM10000 was evaluated.
2.  External generalization was not tested.
3.  SkinNet-HDA is a clean-room reconstruction, not an exact
    reproduction.
4.  The source paper omits important implementation details.
5.  The source paper contains several reproducibility ambiguities.
6.  MobileNet-HDA significantly reduced predictive performance.
7.  Minority/difficult classes were particularly affected.
8.  FLOPs are supported-operator estimates.
9.  CUDA Transformer backward execution is not guaranteed to be
    bit-for-bit deterministic.
10. No post-hoc hyperparameter optimization was performed.
11. No knowledge distillation or quantization was tested.
12. Explainability was outside the final experimental scope.
13. Clinical deployment/validation was not performed.
14. The model is a research classifier and not a clinical diagnostic
    system.

------------------------------------------------------------------------

# 34. Future Work

Natural extensions include:

### Better efficiency/performance balance

Try intermediate backbones rather than an aggressive jump directly to
MobileNetV3-Large.

### Knowledge distillation

Use the stronger ConvNeXt SkinNet-HDA as a teacher for the lightweight
model.

### Quantization

Evaluate:

-   post-training quantization;
-   quantization-aware training.

### Transformer optimization

Investigate:

-   smaller embedding dimensions;
-   projection bottlenecks;
-   more efficient attention.

### Explainability

Add:

-   Grad-CAM / Grad-CAM++;
-   SHAP for metadata.

### External validation

Evaluate additional datasets to measure domain shift and generalization.

### Fairness evaluation

Evaluate subgroup/skin-tone performance only when appropriate datasets
contain explicit, defensible labels.

### Edge deployment

Benchmark:

-   CPU;
-   smartphone;
-   edge GPU;
-   memory consumption;
-   energy consumption.

------------------------------------------------------------------------

# 35. Short Supervisor Meeting Explanation

A concise way to explain the complete project:

> "We started with a broad skin-lesion classification thesis involving
> multimodal learning, explainability and efficiency. For this semester
> we narrowed it to a controlled efficiency study based on DermaScanAI's
> SkinNet-HDA. After auditing the paper and repository, we found that
> the public implementation was incomplete, so we performed a
> paper-guided clean-room reconstruction and explicitly documented every
> undisclosed implementation decision. We built a leakage-safe HAM10000
> pipeline using lesion-level splitting and first established a plain
> ConvNeXt-Tiny baseline. The reconstructed SkinNet-HDA improved
> Macro-F1 from 0.6868 to 0.7129 while keeping accuracy around 82.4%,
> but increased computational cost. We then proposed MobileNet-HDA by
> replacing only the ConvNeXt backbone with MobileNetV3-Large. This
> reduced parameters by 67.8%, supported-operator FLOPs by 87.5%,
> checkpoint size by 67.6%, and P100 latency by 31%. However, Macro-F1
> fell to 0.5809. Therefore our final result is not that MobileNet-HDA
> is universally better; it demonstrates a quantified
> efficiency-performance trade-off. All experiments now remain frozen,
> and the remaining work is thesis reporting, presentation and viva
> preparation."

------------------------------------------------------------------------

# 36. Questions to Discuss With Supervisor

Useful final questions for the meeting:

1.  Is the **efficiency-performance trade-off** framing acceptable as
    the final contribution?
2.  Should the final title emphasize **efficient architecture**,
    **backbone replacement**, or **multimodal skin-lesion
    classification**?
3.  How much detail about the DermaScanAI reproducibility
    inconsistencies should appear in the main report versus limitations?
4.  Should the plain ConvNeXt baseline remain in the main results table
    or be treated as a supporting baseline?
5.  Is the lesion-level split methodology sufficient for the semester
    submission?
6.  Does the supervisor want an additional future-work experiment
    proposed, without actually reopening experimentation?
7.  What report format/chapter structure is required by the department?
8.  What should be emphasized most during the defense: predictive
    performance, computational efficiency, or experimental methodology?

------------------------------------------------------------------------

# 37. Final One-Line Contribution

> **This thesis reconstructs a multimodal SkinNet-HDA skin-lesion
> classifier under a leakage-safe experimental protocol and demonstrates
> that replacing ConvNeXt-Tiny with MobileNetV3-Large can dramatically
> reduce computational cost, but with a measurable loss in balanced
> classification performance.**

------------------------------------------------------------------------

## Experimental Status

**IMPLEMENTATION: CLOSED**

**TRAINING: CLOSED**

**EXPERIMENTS: FROZEN**

**CURRENT STAGE: REPORT + PRESENTATION + VIVA**



And what did the low-cost MobileNet achieve?

This is the result you need to understand extremely well.

1. We made the model MUCH smaller

Before:

32.66 million parameters

After:

10.53 million

That's:

🟢 67.8% fewer parameters

In simple language:

Our model needs roughly one-third as many learned numbers to store the model.

2. We massively reduced the amount of calculation

Before:

4.70 billion counted operations

After:

0.585 billion

That's:

🟢 87.5% fewer counted calculations

So theoretically the model is dramatically less computationally demanding.

3. We made the saved model much smaller

Before:

124.67 MB

After:

40.36 MB

That's:

🟢 67.6% smaller

Think:

Original
████████████████████████████████  125 MB

Ours
██████████                        40 MB

That matters when thinking about limited storage or eventual deployment.

4. It actually ran faster

We tested both on the same Tesla P100 GPU.

Original:

8.97 ms

Ours:

6.19 ms

So:

🟢 31% lower prediction time

That's important because we're not only saying:

“According to mathematics it should be cheaper.”

We actually measured it running faster on the same GPU.

So did we succeed?

Half yes, half no.

And this is the most important thing to understand.

We succeeded massively in reducing cost.

But we paid for that efficiency with lower prediction quality.

Original SkinNet-HDA:

82.4% accuracy
0.713 Macro-F1

Our cheap MobileNet version:

74.5% accuracy
0.581 Macro-F1

So:

🔴 Accuracy dropped about 7.9 percentage points
🔴 Macro-F1 dropped about 0.132

That's a meaningful performance loss.