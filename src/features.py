import gc
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from transformers import AutoModel, AutoImageProcessor, AutoTokenizer

from src.config import cfg, DEVICE, ALL_TARGETS
from src.data import load_and_preprocess_image

# Patch extraction

def split_into_patches(image, patch_size=448, overlap=64):
    """
    Extract overlapping patches from an image using a sliding window.
    Patches smaller than 32 pixels on either side are discarded.
    If no valid patches are found, the original image is returned as a single patch.
    """
    height, width = image.shape[:2]
    stride = max(1, patch_size - overlap)
    patches = []

    for row in range(0, height, stride):
        for column in range(0, width, stride):
            row_end = min(row + patch_size, height)
            column_end = min(column + patch_size, width)
            row_start = max(0, row_end - patch_size)
            column_start = max(0, column_end - patch_size)
            patch = image[row_start:row_end, column_start:column_end]
            if patch.shape[0] >= 32 and patch.shape[1] >= 32:
                patches.append(patch)

    return patches if patches else [image]

# SigLIP image embeddings

def extract_vision_pooled(model, **inputs):
    """
    Get the pooled image feature vector from the SigLIP vision tower.
    Tries pooler_output first, falls back to the CLS token
    (first token of last_hidden_state).
    """
    outputs = model.vision_model(
        **{key: value for key, value in inputs.items() if key == 'pixel_values'}
    )
    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
        return outputs.pooler_output
    return outputs.last_hidden_state[:, 0]


def compute_siglip_embeddings(image_paths, cache_path, batch_size=16):
    """
    Compute one embedding vector per image by averaging patch-level
    SigLIP embeddings. Results are cached to a .npy file so subsequent
    runs skip the computation entirely.

    Parameters
    ----------
    image_paths : list of str
        Absolute paths to each image on disk.
    cache_path : str or Path
        Where to save or load the cached numpy array.
    batch_size : int
        Number of patches to process through SigLIP in one forward pass.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f'cache hit: {cache_path}')
        return np.load(cache_path)

    print(f'loading SigLIP model: {cfg.SIGLIP_PATH}')
    model = AutoModel.from_pretrained(
        cfg.SIGLIP_PATH, local_files_only=cfg.SIGLIP_LOCAL_ONLY
    ).eval().to(DEVICE)
    processor = AutoImageProcessor.from_pretrained(
        cfg.SIGLIP_PATH, local_files_only=cfg.SIGLIP_LOCAL_ONLY
    )

    # run a dummy forward pass to determine the embedding dimension
    with torch.no_grad():
        dummy_image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        dummy_input = processor(images=[dummy_image], return_tensors='pt').to(DEVICE)
        embedding_dimension = extract_vision_pooled(model, **dummy_input).shape[-1]
    print(f'embedding dimension: {embedding_dimension}')

    all_embeddings = []

    for path in tqdm(image_paths, desc='SigLIP embeddings'):
        image = load_and_preprocess_image(path, clean=True)
        if image is None:
            all_embeddings.append(np.zeros(embedding_dimension, dtype=np.float32))
            continue

        patches = split_into_patches(image, cfg.PATCH_SIZE, cfg.OVERLAP)
        pil_patches = [Image.fromarray(patch) for patch in patches]
        patch_vectors = []

        for start in range(0, len(pil_patches), batch_size):
            batch = pil_patches[start:start + batch_size]
            inputs = processor(images=batch, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                patch_vectors.append(extract_vision_pooled(model, **inputs).float().cpu())

        averaged = torch.cat(patch_vectors, dim=0).mean(dim=0).numpy()
        all_embeddings.append(averaged)

    all_embeddings = np.stack(all_embeddings).astype(np.float32)
    np.save(cache_path, all_embeddings)
    print(f'saved: {cache_path}  shape: {all_embeddings.shape}')

    del model, processor
    torch.cuda.empty_cache()
    gc.collect()
    return all_embeddings



# SigLIP semantic probing (zero-shot text-image similarity)

def extract_text_pooled(model, **inputs):
    """
    Get the pooled text feature vector from the SigLIP text tower.
    Same fallback logic as extract_vision_pooled.
    """
    outputs = model.text_model(
        **{key: value for key, value in inputs.items() if key in ['input_ids', 'attention_mask']}
    )
    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
        return outputs.pooler_output
    return outputs.last_hidden_state[:, 0]


def compute_semantic_features(image_embeddings, cache_path):
    """
    For each image, compute similarity scores against hand-crafted
    concept prompts (bare soil, green pasture, clover, dead matter, etc.)
    using SigLIP text-image dot product. Also derives ratio features
    like greenness, clover fraction, and vegetation cover density.

    Results are cached to a .npy file.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f'cache hit: {cache_path}')
        return np.load(cache_path)

    print('computing semantic features...')
    model = AutoModel.from_pretrained(
        cfg.SIGLIP_PATH, local_files_only=cfg.SIGLIP_LOCAL_ONLY
    ).eval().to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.SIGLIP_PATH, local_files_only=cfg.SIGLIP_LOCAL_ONLY
    )

    # each key is one concept axis; the list values are prompt variants
    # that get averaged into a single prototype vector per concept
    concepts = {
        'bare':   ['bare soil', 'dirt ground', 'sparse vegetation', 'exposed earth'],
        'sparse': ['low density pasture', 'thin grass', 'short clipped grass'],
        'medium': ['average pasture cover', 'medium height grass', 'grazed pasture'],
        'dense':  ['dense tall pasture', 'thick grassy volume', 'high biomass', 'overgrown vegetation'],
        'green':  ['lush green vibrant pasture', 'fresh growth', 'photosynthesizing leaves'],
        'dead':   ['dry brown dead grass', 'yellow straw', 'senesced material'],
        'clover': ['white clover', 'trifolium repens', 'clover flowers', 'clover leaves'],
        'grass':  ['ryegrass', 'blade-like leaves', 'fescue', 'grassy sward'],
    }
    concept_keys = list(concepts.keys())

    # encode each concept into an L2-normalized prototype vector
    prototypes = {}
    with torch.no_grad():
        for key, prompts in concepts.items():
            tokens = tokenizer(
                prompts, padding='max_length', max_length=64,
                truncation=True, return_tensors='pt',
            ).to(DEVICE)
            text_features = extract_text_pooled(model, **tokens)
            text_features = text_features / (text_features.norm(p=2, dim=-1, keepdim=True) + 1e-8)
            prototypes[key] = text_features.mean(dim=0, keepdim=True)

    # normalize image embeddings and compute dot-product similarity per concept
    image_tensor = torch.tensor(image_embeddings, dtype=torch.float32, device=DEVICE)
    image_tensor = image_tensor / (image_tensor.norm(p=2, dim=-1, keepdim=True) + 1e-8)

    raw_scores = []
    for key in concept_keys:
        similarity = (image_tensor @ prototypes[key].T).detach().cpu().numpy().reshape(-1)
        raw_scores.append(similarity)
    raw_scores = np.stack(raw_scores, axis=1)  # shape: (number_of_images, 8)

    # compute ratio features from the raw similarity scores
    epsilon = 1e-6
    bare, sparse, medium, dense, green, dead, clover, grass = [
        raw_scores[:, index] for index in range(8)
    ]
    greenness_ratio = green / (green + dead + epsilon)
    clover_ratio = clover / (clover + grass + epsilon)
    cover_ratio = (dense + medium) / (bare + sparse + epsilon)

    semantic_features = np.stack([
        bare, sparse, medium, dense, green, dead, clover, grass,
        greenness_ratio, clover_ratio, cover_ratio,
    ], axis=1).astype(np.float32)

    np.save(cache_path, semantic_features)
    print(f'saved: {cache_path}  shape: {semantic_features.shape}')

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    return semantic_features


# Feature engineering pipeline

class EmbeddingFeaturizer:
    """
    Transform raw SigLIP embeddings into a feature matrix suitable
    for gradient boosting regressors.

    Pipeline stages:
        1. StandardScaler (zero mean, unit variance)
        2. PCA (dimensionality reduction, retaining pca_var fraction of variance)
        3. PLS (Partial Least Squares, supervised projection toward targets)
        4. GMM (Gaussian Mixture Model cluster membership probabilities)
        5. Optionally append normalized semantic features
    """

    def __init__(self, pca_var=0.80, n_pls=8, n_clusters=6, seed=42):
        self.pca_var = pca_var
        self.n_pls = n_pls
        self.n_clusters = n_clusters
        self.seed = seed

        self.scaler = StandardScaler()
        self.pca = None
        self.pls = None
        self.gmm = None
        self._pls_fitted = False

    def fit(self, X, y=None):
        scaled = self.scaler.fit_transform(X)

        # PCA: when pca_var is a float below 1.0, scikit-learn automatically
        # selects enough components to retain that fraction of total variance
        n_components = min(
            self.pca_var if isinstance(self.pca_var, int) else int(self.pca_var * X.shape[1]),
            X.shape[1], X.shape[0],
        )
        if isinstance(self.pca_var, float) and self.pca_var < 1:
            self.pca = PCA(n_components=self.pca_var, random_state=self.seed)
        else:
            self.pca = PCA(n_components=n_components, random_state=self.seed)
        self.pca.fit(scaled)

        # GMM fitted on PCA-reduced space
        pca_output = self.pca.transform(scaled)
        n_gmm = min(self.n_clusters, pca_output.shape[0] // 2)
        self.gmm = GaussianMixture(
            n_components=max(2, n_gmm), covariance_type='diag',
            random_state=self.seed, max_iter=200,
        )
        self.gmm.fit(pca_output)

        # PLS requires target labels and enough samples
        if y is not None and len(y) > self.n_pls:
            n_pls_components = min(self.n_pls, scaled.shape[1], scaled.shape[0] - 1)
            self.pls = PLSRegression(n_components=n_pls_components, scale=False)
            self.pls.fit(scaled, y)
            self._pls_fitted = True

        return self

    def transform(self, X, semantic_features=None):
        scaled = self.scaler.transform(X)
        parts = [self.pca.transform(scaled)]

        if self._pls_fitted and self.pls is not None:
            parts.append(self.pls.transform(scaled))

        parts.append(self.gmm.predict_proba(self.pca.transform(scaled)))

        if semantic_features is not None:
            mean = semantic_features.mean(axis=0, keepdims=True)
            standard_deviation = semantic_features.std(axis=0, keepdims=True) + 1e-6
            parts.append((semantic_features - mean) / standard_deviation)

        return np.hstack(parts).astype(np.float32)


# backward compatibility alias used by the model-saving cell
SupervisedEmbeddingEngine = EmbeddingFeaturizer


# SigLIP boosting out-of-fold predictions

def siglip_oof_predict(embeddings, semantic_features, targets, n_folds=5):
    """
    Train CatBoost, LightGBM, and HistGradientBoosting regressors
    in a K-fold cross-validation loop. For each validation fold the
    three model predictions are averaged. Returns out-of-fold predictions
    with the same shape as targets.
    """
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=cfg.SEED)
    oof_predictions = np.zeros_like(targets, dtype=np.float32)

    catboost_parameters = dict(
        iterations=1000 if not cfg.FAST_DEBUG else 100,
        learning_rate=0.05, depth=4, random_state=cfg.SEED,
        verbose=0, allow_writing_files=False, early_stopping_rounds=100,
    )
    lightgbm_parameters = dict(
        n_estimators=600 if not cfg.FAST_DEBUG else 50,
        learning_rate=0.03, num_leaves=48, subsample=0.75,
        colsample_bytree=0.75, random_state=cfg.SEED, verbose=-1,
    )
    histgb_parameters = dict(
        max_iter=300 if not cfg.FAST_DEBUG else 30,
        learning_rate=0.05, random_state=cfg.SEED,
    )

    for fold, (train_indices, validation_indices) in enumerate(kfold.split(embeddings)):
        print(f'SigLIP fold {fold + 1}/{n_folds}')

        embeddings_train = embeddings[train_indices]
        embeddings_validation = embeddings[validation_indices]
        semantic_train = semantic_features[train_indices]
        semantic_validation = semantic_features[validation_indices]
        targets_train = targets[train_indices]
        targets_validation = targets[validation_indices]

        engine = EmbeddingFeaturizer(pca_var=0.80, n_pls=8, n_clusters=6, seed=cfg.SEED)
        engine.fit(embeddings_train, y=targets_train)
        features_train = engine.transform(embeddings_train, semantic_train)
        features_validation = engine.transform(embeddings_validation, semantic_validation)
        print(f'  features: train={features_train.shape}  validation={features_validation.shape}')

        for target_index, target_name in enumerate(ALL_TARGETS):
            fold_predictions = []

            try:
                catboost_model = CatBoostRegressor(**catboost_parameters)
                catboost_model.fit(
                    features_train, targets_train[:, target_index],
                    eval_set=(features_validation, targets_validation[:, target_index]),
                    verbose=0,
                )
                fold_predictions.append(catboost_model.predict(features_validation))
            except Exception as error:
                print(f'  CatBoost [{target_name}]: {error}')

            try:
                lightgbm_model = LGBMRegressor(**lightgbm_parameters)
                lightgbm_model.fit(features_train, targets_train[:, target_index])
                fold_predictions.append(lightgbm_model.predict(features_validation))
            except Exception as error:
                print(f'  LightGBM [{target_name}]: {error}')

            try:
                histgb_model = HistGradientBoostingRegressor(**histgb_parameters)
                histgb_model.fit(features_train, targets_train[:, target_index])
                fold_predictions.append(histgb_model.predict(features_validation))
            except Exception as error:
                print(f'  HistGradientBoosting [{target_name}]: {error}')

            if fold_predictions:
                oof_predictions[validation_indices, target_index] = np.mean(fold_predictions, axis=0)

        gc.collect()

    return np.maximum(0, oof_predictions)
