import os

import cv2
import numpy as np
import pandas as pd

from src.config import cfg, ALL_TARGETS


def load_dataframes():
    """
    Read competition CSVs and pivot train from long format
    (one row per image-target pair) into wide format
    (one row per image, each target as its own column).
    """
    train_long = pd.read_csv(cfg.DATA_PATH / cfg.TRAIN_CSV)
    test_long = pd.read_csv(cfg.DATA_PATH / cfg.TEST_CSV)

    candidate_meta = [
        'image_path', 'Sampling_Date', 'State',
        'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm',
    ]
    meta_cols = [column for column in candidate_meta if column in train_long.columns]

    # the target value and target name columns vary across competition dataset versions
    target_value_column = 'target' if 'target' in train_long.columns else 'target_value'
    target_name_column = 'target_name' if 'target_name' in train_long.columns else 'component'

    train_wide = train_long.pivot_table(
        index=meta_cols,
        columns=target_name_column,
        values=target_value_column,
        aggfunc='mean',
    ).reset_index()

    for target in ALL_TARGETS:
        if target not in train_wide.columns:
            print(f'WARNING {target} missing from pivot, filling with zeros')
            train_wide[target] = 0.0

    return train_wide, test_long


def resolve_path(relative_path, is_train=True):
    """
    Build an absolute path from a relative image path.
    If the relative path already starts with 'train' or 'test',
    it is joined directly to DATA_PATH. Otherwise the appropriate
    subdirectory (train_images or test_images) is inserted.
    """
    relative_path = str(relative_path)
    if relative_path.startswith('train') or relative_path.startswith('test'):
        return str(cfg.DATA_PATH / relative_path)
    subdirectory = cfg.TRAIN_IMAGES_DIR if is_train else cfg.TEST_IMAGES_DIR
    return str(cfg.DATA_PATH / subdirectory / relative_path)


def attach_absolute_paths(train_wide):
    """
    Add an 'abs_path' column to train_wide. Falls back to a simpler
    join if the first resolved path does not exist on disk.
    """
    train_wide['abs_path'] = train_wide['image_path'].apply(
        lambda path: resolve_path(path, is_train=True)
    )

    probe = train_wide['abs_path'].iloc[0]
    if not os.path.exists(probe):
        print(f'path not found: {probe} -- falling back to direct join')
        train_wide['abs_path'] = train_wide['image_path'].apply(
            lambda path: str(cfg.DATA_PATH / path)
        )

    return train_wide

def clean_image_rgb(image, crop_bottom_fraction=0.10):
    """
    Remove the bottom strip (typically contains metadata artifacts)
    and inpaint orange date-stamps that are commonly burned into
    CSIRO field photographs.

    Parameters
    ----------
    image : numpy.ndarray
        RGB image array with shape (height, width, 3).
    crop_bottom_fraction : float
        Fraction of the image height to crop from the bottom.
    """
    height, width = image.shape[:2]
    crop_height = int(height * (1 - crop_bottom_fraction))
    image = image[:crop_height, :].copy()

    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lower_orange = np.array([5, 150, 150])
        upper_orange = np.array([25, 255, 255])
        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
        if mask.sum() > 0:
            image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    except Exception:
        pass

    return image


def split_left_right(image):
    """Split an image into its left and right halves along the vertical center."""
    midpoint = image.shape[1] // 2
    return image[:, :midpoint].copy(), image[:, midpoint:].copy()


def load_and_preprocess_image(path, clean=True):
    """
    Read an image from disk and optionally clean it.
    Returns an RGB numpy array, or None if the file could not be read.
    """
    raw = cv2.imread(path)
    if raw is None:
        print(f'WARNING could not read: {path}')
        return None
    rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    return clean_image_rgb(rgb) if clean else rgb
