# Multi-Label Chest X-Ray Disease Classification with Grad-CAM Explainability

A DenseNet-121-based deep learning model that predicts the presence of 14 possible thoracic diseases from a single chest X-ray, with Grad-CAM visualizations explaining which regions of the image drove each prediction.

> ⚠️ **This is a portfolio/research project, not a clinical diagnostic tool.** Predictions should never be used for real medical decisions.

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Data Curation](#data-curation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Methodology](#methodology)
- [Results](#results)
- [Error Analysis](#error-analysis)
- [Grad-CAM Explainability](#grad-cam-explainability)
- [Model Improvement: Weight Decay](#model-improvement-weight-decay)
- [Architecture Comparison: DenseNet-121 vs ResNet-50](#architecture-comparison-densenet-121-vs-resnet-50)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)

---

## Problem Statement

**Objective:** Build a multi-label image classification system that predicts the presence of 14 possible thoracic diseases from a chest X-ray, and generate visual explanations (Grad-CAM heatmaps) for each prediction to show which lung regions influenced the model's decision.

**Why this matters:** Manually screening chest X-rays for multiple co-occurring conditions is time-consuming for radiologists. A model that can flag likely findings — and show *why* it flagged them — could support a pre-screening triage workflow, helping prioritize cases for expert review rather than replacing it.

**Success metrics:**
- **Primary:** Macro-averaged AUROC across all 14 disease classes (not accuracy — see [Methodology](#methodology) for why).
- **Secondary:** Per-class AUROC, precision/recall/F1 at class-specific optimal thresholds.
- **Qualitative:** Grad-CAM heatmaps that visually align with clinically relevant regions.

**Related work:** This approach — transfer learning with a pretrained CNN backbone (DenseNet-121) for multi-label chest X-ray classification, paired with Grad-CAM for interpretability — is an established technique in computational radiology research (e.g., CheXNet and related work on this exact dataset). This project applies that established approach rigorously and transparently, rather than claiming a novel method.

---

## Dataset

**Source:** [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) (mirrored via Kaggle for download convenience) — 112,120 frontal-view chest X-ray images from 30,805 unique patients, with labels for 14 disease categories extracted from radiology reports via NLP.

| | |
|---|---|
| Total images | 112,120 |
| Unique patients | 30,805 |
| Image size (raw) | 1024×1024 |
| Disease categories | 14 (multi-label) |
| "No Finding" images | 53.8% |

**Known dataset limitation:** labels were extracted from radiology reports using NLP, not manually verified per image by radiologists. Some label noise is expected and is documented in the dataset's original release papers.

---

## Data Curation

Real, documented data quality issues were identified and handled — not manufactured for demonstration purposes:

| Issue Found | Detection | Handling |
|---|---|---|
| **Anomalous patient ages** | 16 records had ages >100 (up to 414), from a known unit-conversion bug in the original data collection | Flagged and set to null — rows retained, since image and disease labels remain valid |
| **Inconsistent image color mode** | Sampling found `L` (grayscale) and `RGBA` (4-channel) images mixed in the same dataset | Standardized to RGB (`.convert('RGB')`) in the preprocessing pipeline |
| **View Position imbalance** | PA: 67,310 (60%) vs AP: 44,810 (40%) | Documented as a potential confound (AP views correlate with sicker/bedridden patients) rather than corrected, since both are valid clinical views |
| **Duplicate label entries** | Checked `Image Index` uniqueness | None found — confirmed empirically, not assumed |
| **Corner text markers** ("PORTABLE" labels) | Found during sample image review | Investigated via Grad-CAM for shortcut-learning risk (see [Grad-CAM Explainability](#grad-cam-explainability)) — not removed, since it was confirmed the model does not rely on this marker |
| **Non-standard image framing/watermarks** | Found during Grad-CAM review — some images have rotated/triangular framing with a visible watermark | Documented as raw data heterogeneity; confirmed the model's attention stayed on lung tissue despite these artifacts |

---

## Exploratory Data Analysis

Key findings that directly informed modeling decisions:

- **Severe class imbalance:** Infiltration (~20,000 images) is the most common finding; Hernia (~227 images) is the rarest — a ~90x difference in representation.
- **Multi-label is genuinely necessary:** ~18% of images (≈20,000) have 2 or more co-occurring diseases — a single-label classifier would misrepresent a substantial fraction of the data.
- **Disease co-occurrence:** Notable pairings include Infiltration↔Effusion, Infiltration↔Atelectasis, and Mass↔Nodule — the latter pairing was later confirmed as a real source of model confusion (see [Error Analysis](#error-analysis)).

These findings directly motivated: (1) using macro-AUROC instead of accuracy, (2) treating this as a genuine multi-label (not multi-class) problem, and (3) specifically investigating Mass/Nodule confusion during error analysis.

---

## Methodology

### Patient-Level Data Split
Split by unique `Patient ID` (not by image) into 70/15/15 train/val/test, with **zero patient overlap verified** across all three sets. This prevents the model from partially "recognizing" a patient across splits rather than learning genuine disease patterns — a well-known pitfall in medical imaging ML.

| Split | Images | Patients |
|---|---|---|
| Train | 78,566 | 21,563 |
| Validation | 17,063 | 4,621 |
| Test | 16,491 | 4,621 |

### Preprocessing
- Resize to 224×224
- Convert all images to RGB (fixes the mode inconsistency found in curation)
- ImageNet normalization
- Training-only augmentation: random horizontal flip, small random rotation (±5°)

### Model
- **Backbone:** DenseNet-121, pretrained on ImageNet
- **Output layer:** replaced with a 14-unit linear layer (one per disease), sigmoid activation applied internally by the loss function
- **Loss:** `BCEWithLogitsLoss` (numerically stable multi-label loss, operates on raw logits)
- **Optimizer:** Adam, learning rate 1e-4, `ReduceLROnPlateau` scheduler
- **Training:** early stopping (patience = 4 epochs) based on validation macro AUROC, checkpointing best model

### Why Macro AUROC, Not Accuracy
With ~54% of images having no findings and disease frequency varying by up to 90x between classes, a model could achieve high accuracy while performing poorly on rare-but-important diseases. Macro AUROC treats every class equally, better reflecting real diagnostic utility. Per-class thresholds for precision/recall/F1 were derived independently via each class's own precision-recall curve, rather than applying one blanket 0.5 cutoff across all 14 classes.

---

## Results

### Baseline DenseNet-121 (Full Dataset)

| Metric | Value |
|---|---|
| Best Validation Macro AUROC | 0.8402 (epoch 4) |
| Test Set Macro AUROC | 0.8317 |

### Per-Class AUROC (Test Set)

| Disease | AUROC | Positive Count |
|---|---|---|
| Emphysema | 0.928 | 454 |
| Cardiomegaly | 0.908 | 439 |
| Hernia | 0.901 | 27 |
| Edema | 0.894 | 374 |
| Effusion | 0.881 | 1,892 |
| Pneumothorax | 0.871 | 725 |
| Mass | 0.830 | 727 |
| Atelectasis | 0.807 | 1,677 |
| Consolidation | 0.806 | 730 |
| Fibrosis | 0.796 | 271 |
| Pleural_Thickening | 0.793 | 484 |
| Pneumonia | 0.759 | 214 |
| Nodule | 0.757 | 945 |
| Infiltration | 0.711 | 2,932 |

**Notable finding:** AUROC does **not** correlate with training example count. Hernia (27 examples) scored 0.901; Infiltration (2,932 examples, the most common label) scored only 0.711 — the worst of all 14. This is explained in detail in [Error Analysis](#error-analysis) below, and is consistent with known difficulties reported in published research on this exact dataset.

---

## Error Analysis

### Root Causes of Infiltration's Weak Performance (three converging findings)
1. **Class-frequency bias:** confusion matrix showed 3,149 false positives vs. 1,342 false negatives — the model over-predicts Infiltration, consistent with it being the most common label ("safe default" guess).
2. **Weak visual signature:** a manually inspected missed-positive case was visually near-indistinguishable from a correctly-classified negative case — supporting that Infiltration often lacks a strong, consistent radiological appearance.
3. **Image quality sensitivity:** a false-positive case was noticeably hazier/lower-contrast than correctly-classified negatives, suggesting the model may associate general image fog with Infiltration-like appearance.

### Mass ↔ Nodule Cross-Confusion (hypothesis confirmed quantitatively)
- 25.1% of true Mass cases were misclassified as Nodule; 14.3% of true Nodule cases were misclassified as Mass.
- Both rates are 3-5x higher than each class's baseline false-positive rate (~4-5%), confirming the co-occurrence pattern flagged in EDA reflects genuine model confusion between these clinically related findings.

### Cardiomegaly + Portable/AP Imaging (hypothesis tested and revised)
Initial visual review suggested missed Cardiomegaly cases correlated with "PORTABLE"/AP-marked images, raising concern about shortcut learning. **Grad-CAM investigation (below) disproved this** — the model correctly localized the heart region in both missed and correct cases. The actual cause was determined to be confidence calibration on borderline-severity cases, not attentional failure — an important example of hypothesis testing revising an initial conclusion based on evidence.

---

## Grad-CAM Explainability

Grad-CAM heatmaps were generated for all 14 disease classes to verify the model attends to anatomically appropriate regions, not dataset artifacts.

**Findings:**
- **Correct localization confirmed across categories:** central/cardiac region for Cardiomegaly/Edema, bilateral diffuse patterns for Emphysema/Pleural_Thickening (consistent with these being systemic conditions), localized regions for Mass/Nodule/Atelectasis (Nodule notably showed two distinct hot spots — one per lung, on a case with bilateral nodules).
- **No shortcut learning detected:** heatmaps stayed on lung/cardiac tissue even on images with corner text markers ("PORTABLE") or unusual framing/watermarks — tested explicitly, not assumed.
- **One open finding:** Pneumothorax's attention consistently centered on the upper-lung region rather than the more classically expected lung periphery. This pattern held across three independently trained models (original DenseNet-121, weight-decay DenseNet-121, and ResNet-50 comparison), suggesting it reflects a genuine characteristic of how Pneumothorax presents in this dataset's labeled examples, rather than a model-specific quirk.

---

## Model Improvement: Weight Decay

**Observation:** the baseline model's training curve showed clear overfitting — validation AUROC peaked at epoch 4 (0.8402) then declined steadily to 0.7943 by epoch 8, while training loss kept decreasing.

**Change:** added L2 regularization (`weight_decay=1e-4`) to the Adam optimizer — a single, isolated change, to keep the before/after comparison clean and interpretable.

**Result:**

| | Baseline | + Weight Decay |
|---|---|---|
| Best Validation Macro AUROC | 0.8402 | **0.8424** |
| Best Epoch | 4 | 13 |
| Epochs before early stopping | 8 | 17 |
| AUROC decline, 4 epochs post-peak | -0.046 (steep) | -0.006 (gentle) |
| Test Set Macro AUROC | 0.8317 | **0.8356** |

The model trained over 3x longer before overfitting set in, with a substantially more stable post-peak performance — evidence the model learned more generalizable representations rather than memorizing training-specific noise. 12 of 14 individual disease classes showed AUROC improvement. Grad-CAM re-validation on the improved model confirmed the same sound anatomical localization patterns held, with no degradation in interpretability.

**This improved model (`densenet121_weightdecay_best.pth`) is the final reported model for this project.**

---

## Architecture Comparison: DenseNet-121 vs ResNet-50

Both architectures were trained identically (same data, same hyperparameters, same training loop) to isolate the effect of architecture choice alone.

| Metric | DenseNet-121 | ResNet-50 |
|---|---|---|
| Best Validation Macro AUROC | 0.8402 | 0.8357 |
| Total Parameters | ~7.98M | ~23.5M |
| Best Epoch | 4 | 6 |
| Total Training Time | ~114 min (8 epochs) | 142.9 min (10 epochs) |

**DenseNet-121 outperformed ResNet-50 while using ~3x fewer parameters and converging faster** — consistent with DenseNet's dense-connectivity architecture, which promotes feature reuse and parameter efficiency.

**Interpretability comparison:** Grad-CAM applied to both architectures showed strong agreement in anatomical localization for well-performing classes, and neither model showed evidence of shortcut learning from image markers. Notably, **ResNet-50 produced consistently higher-confidence predictions despite slightly lower AUROC** (e.g., Hernia: DenseNet 0.36 vs ResNet 0.97), suggesting DenseNet-121 may offer better-calibrated uncertainty — a meaningful consideration alongside its efficiency advantage.

*Note: two architectures were compared (rather than a larger sweep) to prioritize depth of analysis — rigorous error analysis and interpretability — over breadth of architecture search, consistent with this project's goal of demonstrating pipeline rigor rather than exhaustive benchmarking.*

---

## Limitations

- Labels were extracted from radiology reports via NLP, not manually verified per image by radiologists — some label noise is inherent to the dataset itself.
- Performance varies substantially by class; rare diseases (Hernia, Pneumonia, Fibrosis) have limited training examples and correspondingly less reliable predictions.
- Achieved macro AUROC (~0.84) is consistent with, not exceeding, published research on this exact dataset — this reflects a genuine ceiling from label quality and inherent visual ambiguity in some conditions (particularly Infiltration), not a pipeline shortcoming.
- Grad-CAM shows correlation between image regions and predictions, not proof of causal diagnostic reasoning.
- This is a research/portfolio demonstration, not a clinically validated diagnostic tool.

## Future Work

- Contrast normalization (e.g., CLAHE) during preprocessing, to test whether it reduces Infiltration's image-quality-driven false positives.
- Incorporate View Position (PA/AP) as an explicit model input, given its documented correlation with certain misclassifications.
- Extend architecture comparison to include a third model (e.g., EfficientNet) if pursuing this further.
- Deploy via a lightweight inference script/interface for interactive demonstration.

---

## Repository Structure

```
chest_xray_classifier/
├── data/
│   ├── raw/              # Original NIH ChestX-ray14 download (not included in repo -- see below)
│   └── processed/        # Curated metadata, patient-level splits, cached image path map
├── models/                # Saved model checkpoints (best baseline, weight-decay, ResNet-50)
├── notebooks/
│   ├── eda.ipynb
│   ├── data_preparation.ipynb
│   └── evaluation.ipynb
├── src/
│   ├── dataset.py         # PyTorch Dataset class, image preprocessing
│   ├── model.py            # DenseNet-121 and ResNet-50 model definitions
│   ├── train.py             # Baseline DenseNet-121 training
│   ├── train_weightdecay.py # DenseNet-121 with weight decay (final model)
│   └── train_resnet.py      # ResNet-50 comparison training
├── outputs/                # Grad-CAM galleries, evaluation plots
├── requirements.txt
└── README.md
```

**Note on data/model files:** raw image data (~42GB) and model checkpoints are not included in this repository due to size. See [How to Run](#how-to-run) for download instructions.

## How to Run

```bash
# 1. Set up environment
conda create -n xray_env python=3.11 -y
conda activate xray_env
pip install -r requirements.txt

# 2. Download dataset (requires Kaggle API credentials)
kaggle datasets download -d nih-chest-xrays/data -p data/raw
cd data/raw && unzip -q data.zip && cd ../..

# 3. Run notebooks in order: eda.ipynb -> data_preparation.ipynb

# 4. Train the model
cd src
python3 train_weightdecay.py   # final model, ~4 hours on a single GPU

# 5. Evaluate
# Run notebooks/evaluation.ipynb for metrics, error analysis, and Grad-CAM
```

---

## Tech Stack
Python · PyTorch · torchvision · pandas · scikit-learn · matplotlib/seaborn · Jupyter