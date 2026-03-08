# CSIRO Image-to-Biomass

Kaggle competition solution that predicts pasture biomass components from field photographs using a dual-backbone ensemble of SigLIP and DINOv3 vision models, blended via a learned stacking gate.

## Competition Task

Given a photograph of a pasture plot, predict five biomass targets (in grams):

- `Dry_Green_g` - dry weight of green grass
- `Dry_Clover_g` - dry weight of clover
- `Dry_Dead_g` - dry weight of dead material
- `GDM_g` - Green Dry Matter (= Dry_Green + Dry_Clover)
- `Dry_Total_g` - total dry biomass (= GDM + Dry_Dead)

Evaluation metric: **Weighted R-squared** across all five targets, with Dry_Total weighted 0.5, GDM weighted 0.2, and each leaf component weighted 0.1.

## Pipeline

```
Raw Image
    |
    v
Clean (crop bottom strip, inpaint orange date-stamps)
    |
    v
Split into left and right halves
    |                              |
    v                              v
SigLIP Branch                 DINOv3 Branch
(patch embeddings              (dual-input Vision Transformer
 + semantic probing)            with FiLM fusion)
    |                              |
    v                              v
Feature Engineering            K-Fold out-of-fold
(PCA + PLS + GMM)              predictions
    |                              |
    v                              |
CatBoost / LightGBM /             |
HistGradientBoosting               |
out-of-fold predictions            |
    |                              |
    +----------+    +--------------+
               |    |
               v    v
         Stacking Gate
    (per-target Logistic Regression)
               |
               v
      Mass Balance Enforcement
               |
               v
        Final Predictions
```

## Repository Structure

```
csiro-image2biomass/
    README.md
    requirements.txt
    src/
        __init__.py
        config.py       - paths, hyperparameters, constants, seed
        data.py         - CSV loading, pivot, path resolution, image preprocessing
        metrics.py      - competition R2 metric and mass balance enforcement
        features.py     - SigLIP embeddings, semantic probing, feature engineering, boosting
        models.py       - DINOv3 dataset, transforms, FiLM, regressor, training loops
        gate.py         - stacking gate that blends SigLIP and DINOv3 predictions
    notebooks/
        train.ipynb     - end-to-end training pipeline
```

## Key Design Decisions

**Dual backbone.** SigLIP captures semantic and texture features through patch-level embeddings and zero-shot text-image similarity probing. DINOv3 learns spatial structure through a fine-tuned Vision Transformer operating on left and right image halves. The two models produce complementary error patterns.

**FiLM fusion in DINOv3.** Left and right image halves are encoded separately through a shared backbone. A Feature-wise Linear Modulation layer conditions each half's features on the average scene context before the regression heads.

**Learned gate, not fixed blend.** A per-target logistic regression classifier decides, sample by sample, whether DINOv3 or SigLIP is more accurate for that particular image. The predicted probability becomes the blending weight.

**Mass balance post-processing.** Predictions are clipped to non-negative values, then GDM and Dry_Total are recomputed from their constituent parts to guarantee physical consistency.

## Models

| Model | Source | Parameters |
|-------|--------|------------|
| SigLIP | `google/siglip-base-patch16-224` (Hugging Face) | ~400 million |
| DINOv3 | `facebook/dinov3-vitb16-pretrain-lvd1689m` (Hugging Face) | ~86 million |

## Environment

Built for **Google Colab** with GPU. Configuration paths default to `/content/`. For Kaggle offline submissions, set `SIGLIP_LOCAL_ONLY = True` and `DINO_LOCAL_ONLY = True` in `src/config.py` and pre-download model weights using the download cells at the bottom of the notebook.

## Quick Start

```bash
pip install -r requirements.txt
```

Open `notebooks/train.ipynb` and run all cells.

## Outputs

The training notebook produces a `models/` directory containing:

- `siglip_feature_engine.pkl` - fitted PCA/PLS/GMM feature transformation pipeline
- `siglip_boosting/` - CatBoost, LightGBM, and HistGradientBoosting models per target
- `dinov3_regressor.pth` - full DINOv3 model checkpoint (Exponential Moving Average weights)
- `dinov3_heads_only.pth` - regression heads only (smaller file, requires Hugging Face backbone)
- `gate_models.pkl` - per-target blending gate classifiers
- `config.json` - configuration metadata for inference
