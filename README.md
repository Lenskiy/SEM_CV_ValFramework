# Interpreting Morphology: Linking ZnO Sensor SEM Morphology to Sensing Performance

This repository contains the validation and analysis code accompanying our paper on
optimising and interpreting ZnO gas-sensor performance from electron-microscopy
morphology. The pipeline connects the **fabrication parameters** of electrodeposited
ZnO sensors to their **signal-to-noise ratio (SNR)** sensing performance, trains a
dual-head convolutional neural network (CNN) to predict SNR directly from SEM images,
validates it on an independent replicate sensor batch, and uses explainable-AI (XAI)
methods to interpret *which* morphological features drive the prediction.

> **Paper:** _<paper title>_
> **Authors:** _<author list>_
> **Journal / Year:** _<venue, year>_
> **DOI:** _<doi>_
>
> Please update the placeholders above and the [Citation](#citation) section with the
> final publication details.

---

## Overview

The full workflow lives in a single notebook, [`Interpreting_morphology.ipynb`](Interpreting_morphology.ipynb),
organised into five stages:

| Stage | Section in notebook | What it does |
|-------|--------------------|--------------|
| 1. **Sensor SNR compute** | `Sensor SNR Compute` | Parses fabrication parameters from sensor-response spreadsheets, segments exposure cycles, removes baseline drift (AsLS), and computes a linear SNR per sensor. SNR is binned into 4 performance classes (Low / Mid / High / Ultra). |
| 2. **Prepare SEM** | `Prepare SEM` | Recursively collects SEM images, builds a **group-balanced 4-fold cross-validation** split (every sensor appears in both train and test of every fold), and slices each image into non-overlapping 480 px patches. |
| 3. **CV model** | `CV Model` | Defines a custom residual network (`ResNetDual`) with a **dual head** — classification (4 SNR classes) + regression (SNR value) — and runs 4-fold cross-validation with early stopping. |
| 4. **Replicate validation** | `Load Replicated SEMs` | Applies the trained model to an independent replicate sensor batch to test generalisation. |
| 5. **XAI interpretation** | `XAI Interpretation` | Interprets the model via CNN feature cosine-similarity, t-SNE embeddings, and **LayerCAM** activation maps, quantifying within- vs. between-class activation overlap (IoU). |

### Fabrication parameter encoding

Each sensor is identified by three electrodeposition parameters encoded in the
spreadsheet/image filenames, e.g. `0.2M-3000uA-30s-500x`:

- **Solution concentration** (M)
- **Deposition current** (µA)
- **Deposition time** (s)
- **Magnification** (e.g. `500x`)

### SNR performance classes

| Class | Label | SNR range |
|-------|-------|-----------|
| 0 | Low   | < 5 |
| 1 | Mid   | 5 – 10 |
| 2 | High  | 10 – 20 |
| 3 | Ultra | > 20 |

---

## Repository structure

```
.
├── Interpreting_morphology.ipynb   # End-to-end analysis pipeline (this is the entry point)
├── remove_baseline.py              # Asymmetric least-squares (AsLS) baseline removal  [required helper]
├── compute_noise_level.py          # Noise-level estimation                            [required helper]
└── README.md
```

> **Note:** the notebook imports three local helper modules — `remove_baseline`,
> `compute_noise_level`, and `remove_jitter` — which must sit alongside the notebook.
> Several dataset/model utility classes (`ResNetDual`, `CustomClassificationDatasetWithOriginalLabel`,
> `PreprocessDataset`, `CustomCrop`, `SliceImageIntoPatches`, `EarlyStopper`,
> `modelTrain`, `model_eval`, `resume`, `save_img_by_dataloader`, …) are defined in
> earlier cells of the notebook itself.

---

## Data

The analysis expects two artefact types that are **not** included in this repository
(they are produced/measured experimentally):

1. **Sensor response spreadsheets** (`.xlsx`) — one workbook per sensor, containing
   time, resistance, and gas-concentration columns. Used to compute SNR.
2. **SEM images** (`.tiff` / `.png` / `.jpg`) — micrographs of each sensor, organised
   one folder per sensor so the folder name acts as the cross-validation group ID.

Update the `base_response_path` and `base_img_path` variables near the top of each
stage to point at your local data directories. Trained model checkpoints are written
as `best_model_cv_<fold>_480_dual_NoOutlier.pth`, and intermediate SNR tables as
`snr_no_outlier.csv`.

---

## Installation

The code targets Python 3.10+ and PyTorch with CUDA (CPU also works, more slowly).

```bash
# create an environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# install dependencies
pip install torch torchvision torchcam \
            numpy pandas scipy scikit-learn \
            opencv-python pillow openpyxl \
            matplotlib seaborn tqdm \
            scikit-bio
```

Key libraries used:

- **PyTorch / torchvision** — model definition, training, inference
- **torchcam** — LayerCAM / GradCAM activation maps for XAI
- **scikit-learn** — metrics, KFold, t-SNE, scalers
- **scikit-bio** — PERMANOVA on feature distances
- **scipy / OpenCV / Pillow** — signal processing and image handling
- **pandas / openpyxl** — reading sensor-response spreadsheets

---

## Usage

1. Place the helper modules (`remove_baseline.py`, `compute_noise_level.py`,
   `remove_jitter.py`) next to the notebook.
2. Point `base_response_path` (spreadsheets) and `base_img_path` (SEM images) at your data.
3. Open and run [`Interpreting_morphology.ipynb`](Interpreting_morphology.ipynb) top to bottom:

   ```bash
   jupyter lab Interpreting_morphology.ipynb
   ```

   Run the stages in order — each stage consumes the outputs of the previous one
   (SNR labels → SEM patches/CV splits → trained checkpoints → replicate validation → XAI).

---

## Outputs and reported metrics

- **Regression / classification:** cross-validated accuracy and MAE per fold, plus
  per-sensor SNR R², RMSE, and a predicted-vs-true SNR scatter
  (`snr_regression_performance.pdf`).
- **Feature analysis:** within- and between-class cosine similarity of `layer3`
  CNN features; t-SNE embedding of train/test/replicate sets.
- **XAI:** per-sensor mean LayerCAM IoU distribution (`mean_iou_per_sensor.pdf`),
  within-image vs. between-class activation-overlap ratios, and qualitative CAM overlays.

---

## Reproducibility notes

- Cross-validation, t-SNE, and KFold splits use fixed random seeds (`random_state=42`)
  where applicable; some operations (`random_split`, GPU non-determinism) may produce
  minor run-to-run variation.
- The fold split is **group-balanced**: each physical sensor is guaranteed to appear
  in both the training and test partition of every fold, preventing patch leakage
  across the train/test boundary.
- Results in the paper were generated with 480 px patches, the "NoOutlier" SNR table,
  and the dual-head training objective (`temp = 0.001` weighting the regression loss).

---

## Citation

If you use this code or build on this work, please cite:

```bibtex
@article{<key>,
  title   = {<paper title>},
  author  = {<authors>},
  journal = {<journal>},
  year    = {<year>},
  doi     = {<doi>}
}
```

---

## License

_Add the license under which this code is released (e.g. MIT) and a `LICENSE` file._

## Contact

For questions about the code or data, please contact the corresponding author or open
an issue.
