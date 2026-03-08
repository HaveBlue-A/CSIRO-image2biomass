import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class CFG:
    # paths
    DATA_PATH: Path = Path('/content/')
    TRAIN_CSV: str = 'train.csv'
    TEST_CSV: str = 'test.csv'
    TRAIN_IMAGES_DIR: str = 'train_images'
    TEST_IMAGES_DIR: str = 'test_images'

    # SigLIP backbone (Hugging Face hub identifier)
    SIGLIP_PATH: str = 'google/siglip-base-patch16-224'
    SIGLIP_LOCAL_ONLY: bool = False

    # DINOv3 backbone (Hugging Face hub identifier)
    # vits16 = 22 million parameters
    # vitb16 = 86 million parameters
    # vitl16 = 300 million parameters
    DINO_MODEL_NAME: str = 'facebook/dinov3-vitb16-pretrain-lvd1689m'
    DINO_LOCAL_ONLY: bool = False  # set True for Kaggle offline submissions

    # training
    SEED: int = 42
    N_FOLDS: int = 5
    BATCH_SIZE_DINO: int = 8
    BATCH_SIZE_SIGLIP: int = 32
    NUM_WORKERS: int = 2
    IMG_SIZE: int = 640  # must be a multiple of 16 for patch16 Vision Transformer
    EPOCHS: int = 5
    FAST_DEBUG: bool = False
    DEBUG_SAMPLES: int = 50

    # SigLIP patch extraction
    SIGLIP_IMG_SIZE: int = 224
    PATCH_SIZE: int = 448
    OVERLAP: int = 64

    # stacking gate
    GATE_PER_TARGET: bool = True

    # cache directory for precomputed embeddings
    CACHE_DIR: Path = Path('/content/cache_csiro')


cfg = CFG()
cfg.CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# biomass target columns in the order expected by the competition
ALL_TARGETS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

# per-target weights used by the competition R2 metric
# Dry_Total_g carries the most weight (0.5), GDM_g next (0.2), leaf components 0.1 each
R2_WEIGHTS = np.array([0.1, 0.1, 0.1, 0.2, 0.5], dtype=np.float32)

# ImageNet channel-wise mean and standard deviation for normalizing pretrained Vision Transformer inputs
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def seed_everything(seed=42):
    """Set random seeds across all libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed_everything(cfg.SEED)
