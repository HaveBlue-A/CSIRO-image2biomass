import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.config import cfg, ALL_TARGETS
from src.metrics import weighted_r2_global, enforce_mass_balance


def build_gate_features(oof_siglip, oof_dino, semantic_features=None, meta_features=None):
    """
    Construct the input feature matrix for the stacking gate classifier.

    Includes:
        - raw out-of-fold predictions from both models
        - absolute disagreement between the two models
        - element-wise mean of the two model predictions
        - (optional) semantic similarity features
        - (optional) tabular metadata features
    """
    parts = [
        oof_siglip,
        oof_dino,
        np.abs(oof_dino - oof_siglip),
        (oof_dino + oof_siglip) / 2,
    ]
    if semantic_features is not None:
        parts.append(semantic_features.astype(np.float32))
    if meta_features is not None:
        parts.append(meta_features.astype(np.float32))
    return np.hstack(parts).astype(np.float32)


def train_gate(
    oof_siglip, oof_dino, ground_truth, semantic_features,
    meta_features, train_wide,
):
    """
    Train a per-target logistic regression gate that learns when to
    trust DINOv3 over SigLIP for each sample.

    For each target:
        1. Label each sample: 1 if DINOv3 was closer to ground truth, 0 if SigLIP was closer
        2. Fit a StandardScaler + LogisticRegression pipeline on the gate features
        3. Use the predicted probability as the blending weight

    Final prediction = weight * DINOv3 + (1 - weight) * SigLIP

    Returns
    -------
    gate_models : list
        One fitted pipeline (or None) per target.
    blended_predictions : numpy.ndarray
        Mass-balanced blended out-of-fold predictions.
    """
    available_meta_columns = [
        column for column in ['Pre_GSHH_NDVI', 'Height_Ave_cm']
        if column in train_wide.columns
    ]
    if available_meta_columns:
        meta_numeric = train_wide[available_meta_columns].fillna(0).values.astype(np.float32)
    else:
        meta_numeric = np.zeros((len(train_wide), 1), dtype=np.float32)

    gate_feature_matrix = build_gate_features(
        oof_siglip, oof_dino,
        semantic_features=semantic_features,
        meta_features=meta_numeric,
    )
    print(f'gate feature matrix: {gate_feature_matrix.shape}')

    gate_models = []
    blending_weights = np.zeros_like(oof_siglip, dtype=np.float32)

    for target_index, target_name in enumerate(ALL_TARGETS):
        print(f'\ngate [{target_name}]')
        target_truth = ground_truth[:, target_index]

        dino_absolute_error = np.abs(target_truth - oof_dino[:, target_index])
        siglip_absolute_error = np.abs(target_truth - oof_siglip[:, target_index])
        labels = (dino_absolute_error < siglip_absolute_error).astype(int)

        dino_wins = labels.sum()
        siglip_wins = len(labels) - dino_wins
        print(f'  DINOv3 wins: {dino_wins}  SigLIP wins: {siglip_wins}')

        # when one model dominates completely, skip training and hardcode the weight
        if dino_wins == 0:
            blending_weights[:, target_index] = 0.0
            gate_models.append(None)
            continue
        if siglip_wins == 0:
            blending_weights[:, target_index] = 1.0
            gate_models.append(None)
            continue

        classifier = Pipeline([
            ('scaler', StandardScaler(with_mean=True, with_std=True)),
            ('logistic_regression', LogisticRegression(
                max_iter=2000, C=1.0, class_weight='balanced',
                random_state=cfg.SEED,
            )),
        ])
        classifier.fit(gate_feature_matrix, labels)

        probability = classifier.predict_proba(gate_feature_matrix)[:, 1].astype(np.float32)
        blending_weights[:, target_index] = probability
        gate_models.append(classifier)
        print(f'  weight: mean={probability.mean():.3f} std={probability.std():.3f}')

    # blend and apply mass balance constraints
    blended = blending_weights * oof_dino + (1 - blending_weights) * oof_siglip
    blended = np.maximum(0, blended)

    r2_before_mass_balance = weighted_r2_global(ground_truth, blended)
    print(f'\nblended R2 (raw): {r2_before_mass_balance:.4f}')

    blended_dataframe = enforce_mass_balance(pd.DataFrame(blended, columns=ALL_TARGETS))
    blended_final = blended_dataframe[ALL_TARGETS].values

    r2_after_mass_balance = weighted_r2_global(ground_truth, blended_final)
    print(f'blended R2 (mass-balanced): {r2_after_mass_balance:.4f}')

    print(f'\nSigLIP:  {weighted_r2_global(ground_truth, oof_siglip):.4f}')
    print(f'DINOv3:  {weighted_r2_global(ground_truth, oof_dino):.4f}')
    print(f'Blended: {r2_after_mass_balance:.4f}')

    return gate_models, blended_final
