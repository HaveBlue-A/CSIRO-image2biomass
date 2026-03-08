import gc
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.swa_utils import AveragedModel
from tqdm.auto import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import AutoModel as HFAutoModel

from sklearn.model_selection import KFold

from src.config import cfg, DEVICE, ALL_TARGETS, R2_WEIGHTS, IMAGENET_MEAN, IMAGENET_STD
from src.data import load_and_preprocess_image, split_left_right
from src.metrics import weighted_r2_global


# Data augmentation pipelines (albumentations)

def get_train_transforms(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_validation_transforms(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# PyTorch Dataset

class BiomassDataset(Dataset):
    """
    Each sample returns (left_half, right_half, target_vector).
    The image is cleaned, then split vertically into two halves
    so the DINOv3 model can encode spatial context from both sides.
    """

    def __init__(self, dataframe, transform, is_train=True):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.paths = self.dataframe['abs_path'].values

        if is_train or all(target in self.dataframe.columns for target in ALL_TARGETS):
            self.targets = self.dataframe[ALL_TARGETS].values.astype(np.float32)
        else:
            self.targets = np.zeros((len(self.dataframe), len(ALL_TARGETS)), dtype=np.float32)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        image = load_and_preprocess_image(self.paths[index], clean=True)
        if image is None:
            image = np.zeros((900, 2000, 3), dtype=np.uint8)

        left_half, right_half = split_left_right(image)
        left_half = self.transform(image=left_half)['image']
        right_half = self.transform(image=right_half)['image']
        return left_half, right_half, torch.from_numpy(self.targets[index])


# backward compatibility alias
DinoDataset = BiomassDataset

# Feature-wise Linear Modulation (FiLM)

class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation layer (Perez et al., 2018).
    Given a conditioning vector, produces scale (gamma) and shift (beta)
    parameters that modulate a feature vector element-wise.
    """

    def __init__(self, dimension):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dimension, dimension // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dimension // 2, dimension * 2),
        )

    def forward(self, x):
        gamma_beta = self.mlp(x)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        return gamma, beta

# DINOv3 regression model

class DINOv3Regressor(nn.Module):
    """
    Dual-input regressor built on a DINOv3 Vision Transformer backbone.

    Architecture:
        1. Encode left and right image halves through a shared DINOv3 backbone
        2. Compute a context vector by averaging the two CLS token embeddings
        3. Apply FiLM conditioning so each half's features are modulated
           by the overall scene context
        4. Concatenate the modulated features and predict three leaf-level
           biomass components (green, clover, dead) through separate heads
        5. Derive GDM and Total deterministically from the leaf components

    Softplus activation on each head ensures predictions are non-negative.
    """

    def __init__(self, model_name, local_only=False, freeze_backbone=False):
        super().__init__()
        print(f'loading backbone: {model_name}')
        self.backbone = HFAutoModel.from_pretrained(
            model_name, local_files_only=local_only, trust_remote_code=True,
        )
        hidden_size = getattr(self.backbone.config, 'hidden_size', 768)
        self.hidden_dim = hidden_size
        print(f'hidden dimension: {hidden_size}')

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            print('backbone frozen')

        self.film = FiLM(hidden_size)

        def _make_regression_head():
            return nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(hidden_size // 2, 1),
            )

        self.h_green = _make_regression_head()
        self.h_clover = _make_regression_head()
        self.h_dead = _make_regression_head()
        self.softplus = nn.Softplus()

    def _pool(self, pixel_values):
        """Extract CLS token embedding from the backbone."""
        outputs = self.backbone(pixel_values=pixel_values)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0]

    def forward(self, left, right):
        features_left = self._pool(left)
        features_right = self._pool(right)

        context = (features_left + features_right) / 2
        gamma, beta = self.film(context)
        features_left = features_left * (1 + gamma) + beta
        features_right = features_right * (1 + gamma) + beta

        fused = torch.cat([features_left, features_right], dim=1)

        green = self.softplus(self.h_green(fused))
        clover = self.softplus(self.h_clover(fused))
        dead = self.softplus(self.h_dead(fused))
        green_dry_matter = green + clover
        total = green_dry_matter + dead

        # output order matches ALL_TARGETS: [Dry_Green, Dry_Dead, Dry_Clover, GDM, Dry_Total]
        return torch.cat([green, dead, clover, green_dry_matter, total], dim=1)

# K-fold out-of-fold training

def train_dino_oof(dataframe, n_folds=5):
    """
    Train DINOv3Regressor with K-fold cross-validation.
    Returns out-of-fold predictions with shape (number_of_samples, 5).

    Each fold uses:
        - differential learning rates (1e-5 for backbone, 2e-4 for heads)
        - cosine annealing schedule
        - SmoothL1 loss weighted by the competition target weights
        - gradient clipping at max norm 1.0
    """
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=cfg.SEED)
    oof_predictions = np.zeros((len(dataframe), len(ALL_TARGETS)), dtype=np.float32)
    number_of_epochs = cfg.EPOCHS if not cfg.FAST_DEBUG else 1

    for fold, (train_indices, validation_indices) in enumerate(kfold.split(dataframe)):
        print(f'\n{"=" * 50}')
        print(f'DINOv3 fold {fold + 1}/{n_folds}')
        print(f'{"=" * 50}')

        train_dataframe = dataframe.iloc[train_indices].reset_index(drop=True)
        validation_dataframe = dataframe.iloc[validation_indices].reset_index(drop=True)

        train_dataset = BiomassDataset(train_dataframe, get_train_transforms(cfg.IMG_SIZE), is_train=True)
        validation_dataset = BiomassDataset(validation_dataframe, get_validation_transforms(cfg.IMG_SIZE), is_train=True)

        train_loader = DataLoader(
            train_dataset, batch_size=cfg.BATCH_SIZE_DINO, shuffle=True,
            num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True,
        )
        validation_loader = DataLoader(
            validation_dataset, batch_size=cfg.BATCH_SIZE_DINO, shuffle=False,
            num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=False,
        )

        network = DINOv3Regressor(
            cfg.DINO_MODEL_NAME, local_only=cfg.DINO_LOCAL_ONLY, freeze_backbone=False,
        ).to(DEVICE)

        backbone_parameters = list(network.backbone.parameters())
        head_parameters = (
            list(network.film.parameters())
            + list(network.h_green.parameters())
            + list(network.h_clover.parameters())
            + list(network.h_dead.parameters())
        )

        optimizer = optim.AdamW([
            {'params': backbone_parameters, 'lr': 1e-5},
            {'params': head_parameters, 'lr': 2e-4},
        ], weight_decay=1e-2)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=number_of_epochs * len(train_loader),
        )
        loss_function = nn.SmoothL1Loss(beta=5.0, reduction='none')
        target_weights = torch.tensor(R2_WEIGHTS, device=DEVICE)

        for epoch in range(number_of_epochs):
            network.train()
            running_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f'epoch {epoch + 1}/{number_of_epochs}')

            for left, right, targets in progress_bar:
                left = left.to(DEVICE, non_blocking=True)
                right = right.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                predictions = network(left, right)
                per_target_loss = loss_function(predictions, targets)
                loss = (per_target_loss * target_weights).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                running_loss += loss.item()
                progress_bar.set_postfix(loss=f'{loss.item():.4f}')

            print(f'  average loss: {running_loss / len(train_loader):.4f}')

        # collect validation predictions
        network.eval()
        validation_predictions = []
        with torch.no_grad():
            for left, right, _ in tqdm(validation_loader, desc='validation'):
                left = left.to(DEVICE, non_blocking=True)
                right = right.to(DEVICE, non_blocking=True)
                validation_predictions.append(network(left, right).float().cpu().numpy())

        oof_predictions[validation_indices] = np.maximum(0, np.vstack(validation_predictions))

        del network, optimizer, scheduler, train_loader, validation_loader
        del train_dataset, validation_dataset
        torch.cuda.empty_cache()
        gc.collect()

        fold_r2 = weighted_r2_global(
            dataframe.iloc[validation_indices][ALL_TARGETS].values,
            oof_predictions[validation_indices],
        )
        print(f'fold {fold + 1} R2: {fold_r2:.4f}')

    return oof_predictions

# Full-data retraining with Automatic Mixed Precision, Exponential Moving
# Average, and MixUp augmentation
#                                                                          

def train_dino_full(dataframe, targets):
    """
    Retrain DINOv3Regressor on the entire training set (no validation split).
    Returns the trained model and its Exponential Moving Average copy.

    Enhancements over the per-fold training:
        - Automatic Mixed Precision (torch.cuda.amp) for faster training on GPU
        - Exponential Moving Average via torch.optim.swa_utils.AveragedModel
          for smoother, more generalizable weights
        - MixUp data augmentation applied with 30% probability per batch
    """
    print('\ntraining DINOv3 on full data...')

    full_dataset = BiomassDataset(dataframe, get_train_transforms(cfg.IMG_SIZE), is_train=True)
    full_loader = DataLoader(
        full_dataset, batch_size=cfg.BATCH_SIZE_DINO, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True,
    )

    model = DINOv3Regressor(
        cfg.DINO_MODEL_NAME, local_only=cfg.DINO_LOCAL_ONLY, freeze_backbone=False,
    ).to(DEVICE)

    # differential learning rates: slow for pretrained backbone, fast for regression heads
    backbone_parameters = list(model.backbone.parameters()) if hasattr(model, 'backbone') else []
    head_parameters = []
    for attribute_name in ['film', 'h_green', 'h_clover', 'h_dead']:
        module = getattr(model, attribute_name, None)
        if module is not None:
            head_parameters += list(module.parameters())

    if not backbone_parameters or not head_parameters:
        optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    else:
        optimizer = optim.AdamW([
            {'params': backbone_parameters, 'lr': 1e-5},
            {'params': head_parameters, 'lr': 2e-4},
        ], weight_decay=1e-2)

    number_of_epochs = cfg.EPOCHS if not cfg.FAST_DEBUG else 1
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=number_of_epochs * len(full_loader),
    )
    loss_function = nn.SmoothL1Loss(beta=5.0, reduction='none')
    target_weights = torch.tensor(R2_WEIGHTS, device=DEVICE)

    use_amp = (DEVICE == 'cuda')
    amp_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ema_model = AveragedModel(model)

    mixup_probability = 0.30
    mixup_alpha = 0.4

    for epoch in range(number_of_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(full_loader, desc=f'full-train epoch {epoch + 1}/{number_of_epochs}')

        for left, right, batch_targets in progress_bar:
            left = left.to(DEVICE, non_blocking=True)
            right = right.to(DEVICE, non_blocking=True)
            batch_targets = batch_targets.to(DEVICE, non_blocking=True)

            # MixUp: blend pairs of samples within the batch
            if random.random() < mixup_probability and left.size(0) > 1:
                mixing_coefficient = np.random.beta(mixup_alpha, mixup_alpha)
                permuted_indices = torch.randperm(left.size(0), device=DEVICE)
                left = mixing_coefficient * left + (1 - mixing_coefficient) * left[permuted_indices]
                right = mixing_coefficient * right + (1 - mixing_coefficient) * right[permuted_indices]
                batch_targets = mixing_coefficient * batch_targets + (1 - mixing_coefficient) * batch_targets[permuted_indices]

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                predictions = model(left, right)
                per_target_loss = loss_function(predictions, batch_targets)
                loss = (per_target_loss * target_weights).mean()

            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            scheduler.step()
            ema_model.update_parameters(model)

            running_loss += loss.item()
            progress_bar.set_postfix(loss=f'{loss.item():.4f}')

        print(f'  average loss: {running_loss / len(full_loader):.4f}')

    return model, ema_model
