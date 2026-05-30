# ============================================================
# MODEL COMPARISON SCRIPT - V4 CACHED / NO RETRAINING
#
# This script compares already trained segmentation checkpoints.
# It does NOT train any model.
#
# Main changes in V4:
# - Does not import/execute the training scripts, avoiding side effects.
# - Reuses cached image splits, patch inventories and mean/std values.
# - Repairs outdated selected_top100.csv paths such as MY28/mdgm_rgb/...
#   by searching the current clean folder structure.
# - Uses PIL instead of cv2 for robust Unicode paths on Windows.
# - Recomputes the best threshold on validation for every model.
# - Evaluates test using the validation-selected threshold.
#
# Expected structure:
#
# ComparisonFolder/
# ├── compare_generated_models_v4_cached.py
# ├── Codigos para generar metodos/          optional; only for reference
# ├── Modelos generados/
# │   ├── best_attention_unet_mars.pth
# │   ├── best_attention_unet_mars_v2.pth
# │   ├── best_attention_unet_multiyear.pth
# │   ├── best_attention_unet_multiyear_v2.pth
# │   └── best_cbam_unet_mdad_article_style.pth
# ├── MARS/
# │   ├── images/
# │   └── masks_multiclass/
# ├── MY28/
# ├── MY31/
# └── MY32/
#
# Outputs:
# comparison_generated_models_outputs/
# ├── comparison_metrics_summary.csv
# ├── comparison_model_parameters.csv
# ├── comparison_summary_bars.png
# ├── comparison_visual_grid.png
# ├── comparison_report.md
# └── <model_name>/
#     ├── cache/
#     ├── validation_threshold_sweep.csv
#     ├── test_threshold_sweep.csv
#     ├── test_metrics.csv
#     ├── test_metrics_best_test_threshold_diagnostic.csv
#     ├── test_confusion_matrix.png
#     └── selected_visual_example.png
# ============================================================

from __future__ import annotations

import json
import math
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from matplotlib.colors import ListedColormap
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ============================================================
# 1. GLOBAL CONFIGURATION
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "Modelos generados"
OUTPUT_ROOT = PROJECT_ROOT / "comparison_generated_models_outputs"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True

# Cache behavior.
REUSE_CACHE = True
FORCE_REBUILD_CACHE = False

# Evaluation behavior.
EVAL_BATCH_SIZE_ATTENTION = 8
EVAL_BATCH_SIZE_CBAM = 2
NUM_WORKERS_EVAL = 0

# Patch selection for visual examples.
TARGET_STORM_RANGE = (0.30, 0.40)
TARGET_STORM_CENTER = 0.35

# Visualization colors: 0=no storm, 1=storm, 2=invalid/no-data.
CLASS_CMAP = ListedColormap(["#440154", "#2A9D9A", "#FDE725"])

YEARS = ["MY28", "MY31", "MY32"]
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]

MARS_TRAIN_IMAGES = [
    "346_F04_day22", "341_F04_day17", "328_F04_day04", "353_F04_day29",
    "082_D17_day19", "340_F04_day16", "106_D18_day10", "090_D17_day27",
    "357_F04_day33", "030_D15_day32", "290_F02_day31", "314_F03_day22"
]
MARS_VAL_IMAGES = ["354_F04_day30", "330_F04_day06"]
MARS_TEST_IMAGES = ["345_F04_day21", "089_D17_day26", "031_D15_day33"]


MODEL_REGISTRY = [
    {
        "model_name": "attention_unet_mars_v1",
        "architecture": "attention_bn",
        "data_source": "mars",
        "checkpoint_names": ["best_attention_unet_mars.pth"],
        "patch_h": 256,
        "patch_w": 256,
        "stride_y": 256,
        "stride_x": 256,
        "min_valid_ratio": 0.70,
        "base_channels_default": 16,
        "thresholds": np.round(np.arange(0.10, 0.91, 0.05), 2).tolist(),
        "mean_default": [0.55, 0.35, 0.25],
        "std_default": [0.20, 0.15, 0.15],
    },
    {
        "model_name": "attention_unet_mars_v2",
        "architecture": "attention_gn",
        "data_source": "mars",
        "checkpoint_names": ["best_attention_unet_mars_v2.pth"],
        "patch_h": 256,
        "patch_w": 256,
        "stride_y": 128,
        "stride_x": 128,
        "min_valid_ratio": 0.70,
        "base_channels_default": 32,
        "thresholds": np.round(np.arange(0.10, 0.91, 0.05), 2).tolist(),
        "mean_default": None,  # computed from training pixels
        "std_default": None,
    },
    {
        "model_name": "attention_unet_multiyear_v1",
        "architecture": "attention_bn",
        "data_source": "multiyear_raw",
        "checkpoint_names": ["best_attention_unet_multiyear.pth"],
        "patch_h": 256,
        "patch_w": 256,
        "stride_y": 256,
        "stride_x": 256,
        "min_valid_ratio": 0.70,
        "base_channels_default": 16,
        "thresholds": np.round(np.arange(0.10, 0.91, 0.05), 2).tolist(),
        "split_mode": "shuffle",
        "mean_default": [0.55, 0.35, 0.25],
        "std_default": [0.20, 0.15, 0.15],
    },
    {
        "model_name": "attention_unet_multiyear_v2",
        "architecture": "attention_bn",
        "data_source": "multiyear_raw",
        "checkpoint_names": ["best_attention_unet_multiyear_v2.pth"],
        "patch_h": 256,
        "patch_w": 256,
        "stride_y": 256,
        "stride_x": 256,
        "min_valid_ratio": 0.70,
        "base_channels_default": 16,
        "thresholds": np.round(np.arange(0.10, 0.91, 0.05), 2).tolist(),
        "split_mode": "shuffle",
        "mean_default": [0.55, 0.35, 0.25],
        "std_default": [0.20, 0.15, 0.15],
    },
    {
        "model_name": "cbam_unet_article_style",
        "architecture": "cbam",
        "data_source": "multiyear_cbam",
        "checkpoint_names": ["best_cbam_unet_mdad_article_style.pth"],
        "patch_h": 668,
        "patch_w": 688,
        "stride_y": 334,
        "stride_x": 334,
        "min_valid_ratio": 0.50,
        "base_channels_default": 16,
        "thresholds": np.round(np.arange(0.50, 1.00, 0.05), 2).tolist(),
        "split_mode": "sequential",
        "mean_default": None,  # computed from training images after CBAM preprocessing
        "std_default": None,
    },
]


# ============================================================
# 2. FILE AND PATH UTILITIES
# ============================================================

def log(message: str) -> None:
    print(message, flush=True)


def normalize_key(text: Any) -> str:
    return (
        str(text)
        .lower()
        .replace("_mask_class", "")
        .replace("_multiclass", "")
        .replace("_mask", "")
        .replace("_", "")
        .replace("-", "")
        .replace(".", "")
        .replace(" ", "")
    )


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(text))


def find_checkpoint(names: List[str]) -> Path:
    for name in names:
        direct = MODELS_DIR / name
        if direct.exists():
            return direct
        matches = sorted(MODELS_DIR.rglob(name)) if MODELS_DIR.exists() else []
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Checkpoint not found. Tried: {names}")


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "state_dict", "model", "net"]:
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if len(checkpoint) > 0 and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
    raise ValueError("Could not locate model state_dict in checkpoint.")


def infer_base_channels(checkpoint: Any, default: int) -> int:
    if isinstance(checkpoint, dict):
        for key in ["base_channels", "BASE_CHANNELS", "base_ch"]:
            if key in checkpoint:
                try:
                    return int(checkpoint[key])
                except Exception:
                    pass
    return int(default)


def infer_mean_std_from_checkpoint(checkpoint: Any, mean_default, std_default):
    mean = None
    std = None
    if isinstance(checkpoint, dict):
        for key in ["mean", "train_mean"]:
            if key in checkpoint:
                mean = np.asarray(checkpoint[key], dtype=np.float32)
                break
        for key in ["std", "train_std"]:
            if key in checkpoint:
                std = np.asarray(checkpoint[key], dtype=np.float32)
                break

    if mean is None and mean_default is not None:
        mean = np.asarray(mean_default, dtype=np.float32)
    if std is None and std_default is not None:
        std = np.asarray(std_default, dtype=np.float32)

    return mean, std


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("L")).astype(np.uint8)


def find_file_flexible(
    folders: List[Path],
    preferred_name: Optional[str] = None,
    tokens: Optional[List[str]] = None
) -> Optional[Path]:
    tokens = [normalize_key(t) for t in (tokens or []) if str(t) and str(t).lower() != "nan"]

    # Exact filename first.
    if preferred_name:
        for folder in folders:
            if not folder.exists():
                continue
            direct = folder / preferred_name
            if direct.exists():
                return direct
            matches = sorted(folder.rglob(preferred_name))
            if matches:
                return matches[0]

    # Flexible token matching.
    all_files = []
    for folder in folders:
        if not folder.exists():
            continue
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]:
            all_files.extend(folder.rglob(ext))

    if not tokens:
        return None

    for token in tokens:
        for file in sorted(all_files):
            key = normalize_key(file.stem)
            if token and (token in key or key in token):
                return file

    return None


def repair_pair_paths(pair_df: pd.DataFrame, root: Path) -> pd.DataFrame:
    out = pair_df.copy()

    for idx, row in out.iterrows():
        year = str(row.get("year", ""))
        year_dir = root / year

        image_text = str(row.get("image_path", ""))
        mask_text = str(row.get("mask_path", row.get("mask_png_path", "")))

        image_path = Path(image_text)
        mask_path = Path(mask_text)

        if not image_path.is_absolute():
            image_path = root / image_path
        if not mask_path.is_absolute():
            mask_path = root / mask_path

        day_key = str(row.get("day_key", ""))
        image_stem = Path(image_text).stem
        mask_stem = Path(mask_text).stem

        image_folders = [
            year_dir / "selected_top100" / "images",
            year_dir / "images",
            year_dir / "mdgm_rgb",
        ]
        mask_folders = [
            year_dir / "selected_top100" / "masks",
            year_dir / "masks",
            year_dir / "masks_multiclass",
            year_dir / "masks_png_all",
        ]

        if not image_path.exists():
            fixed = find_file_flexible(
                image_folders,
                preferred_name=Path(image_text).name,
                tokens=[day_key, image_stem]
            )
            if fixed is not None:
                image_path = fixed

        if not mask_path.exists():
            fixed = find_file_flexible(
                mask_folders,
                preferred_name=Path(mask_text).name,
                tokens=[day_key, image_path.stem, image_stem, mask_stem]
            )
            if fixed is not None:
                mask_path = fixed

        out.at[idx, "image_path"] = str(image_path)
        out.at[idx, "mask_path"] = str(mask_path)

    missing_images = out[~out["image_path"].map(lambda x: Path(str(x)).exists())]
    missing_masks = out[~out["mask_path"].map(lambda x: Path(str(x)).exists())]

    if len(missing_images) > 0:
        raise FileNotFoundError(
            "Some image paths could not be repaired. First examples:\n"
            + "\n".join(missing_images["image_path"].head(10).astype(str).tolist())
        )

    if len(missing_masks) > 0:
        raise FileNotFoundError(
            "Some mask paths could not be repaired. First examples:\n"
            + "\n".join(missing_masks["mask_path"].head(10).astype(str).tolist())
        )

    return out


# ============================================================
# 3. PREPROCESSING
# ============================================================


def normalize_patch_inventory_columns(inv: pd.DataFrame) -> pd.DataFrame:
    """
    Makes patch inventories homogeneous across models.

    CBAM inventories generated by the article-style script use:
        storm_ratio_valid

    The comparison visualization expects:
        storm_ratio

    This function creates the common column without rebuilding the inventory.
    """
    out = inv.copy()

    if "storm_ratio" not in out.columns:
        if "storm_ratio_valid" in out.columns:
            out["storm_ratio"] = out["storm_ratio_valid"]
        elif "storm_pct_valid" in out.columns:
            out["storm_ratio"] = out["storm_pct_valid"] / 100.0
        else:
            out["storm_ratio"] = 0.0

    if "positive_patch" not in out.columns:
        if "storm_pixels" in out.columns:
            out["positive_patch"] = out["storm_pixels"] > 0
        else:
            out["positive_patch"] = out["storm_ratio"] > 0

    if "valid_ratio" not in out.columns:
        out["valid_ratio"] = 1.0

    return out



def detect_black(rgb: np.ndarray, threshold: int = 5) -> np.ndarray:
    return np.all(rgb <= threshold, axis=2)


def trim_black_borders(rgb: np.ndarray, mask: np.ndarray, border_black_frac: float = 0.95):
    black = detect_black(rgb)
    h, w = black.shape

    top = 0
    while top < h - 1 and black[top, :].mean() >= border_black_frac:
        top += 1

    bottom = h - 1
    while bottom > top and black[bottom, :].mean() >= border_black_frac:
        bottom -= 1

    left = 0
    while left < w - 1 and black[:, left].mean() >= border_black_frac:
        left += 1

    right = w - 1
    while right > left and black[:, right].mean() >= border_black_frac:
        right -= 1

    return rgb[top:bottom + 1, left:right + 1], mask[top:bottom + 1, left:right + 1]


def wrap_pad_east_west(rgb: np.ndarray, mask: np.ndarray, pad: int = 36):
    if pad <= 0 or mask.shape[1] <= 2 * pad:
        return rgb, mask
    rgb_padded = np.concatenate([rgb[:, -pad:, :], rgb, rgb[:, :pad, :]], axis=1)
    mask_padded = np.concatenate([mask[:, -pad:], mask, mask[:, :pad]], axis=1)
    return rgb_padded, mask_padded


def preprocess_cbam_pair(rgb: np.ndarray, mask: np.ndarray):
    if rgb.shape[:2] != mask.shape:
        mask_img = Image.fromarray(mask.astype(np.uint8), mode="L")
        mask_img = mask_img.resize((rgb.shape[1], rgb.shape[0]), resample=Image.Resampling.NEAREST)
        mask = np.asarray(mask_img).astype(np.uint8)

    mask = mask.copy()
    mask[detect_black(rgb)] = 2
    rgb, mask = trim_black_borders(rgb, mask)
    mask[detect_black(rgb)] = 2
    rgb, mask = wrap_pad_east_west(rgb, mask, pad=36)
    return rgb, mask


class PairCache:
    def __init__(self, mode: str, max_items: int = 16):
        self.mode = mode
        self.max_items = int(max_items)
        self.cache = OrderedDict()

    def load(self, image_path: str, mask_path: str):
        key = (str(image_path), str(mask_path), self.mode)
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value

        rgb = read_rgb(Path(image_path))
        mask = read_mask(Path(mask_path))

        if self.mode == "cbam":
            rgb, mask = preprocess_cbam_pair(rgb, mask)
        else:
            if rgb.shape[:2] != mask.shape:
                mask_img = Image.fromarray(mask.astype(np.uint8), mode="L")
                mask_img = mask_img.resize((rgb.shape[1], rgb.shape[0]), resample=Image.Resampling.NEAREST)
                mask = np.asarray(mask_img).astype(np.uint8)

        self.cache[key] = (rgb, mask)
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)

        return rgb, mask


# ============================================================
# 4. DATA COLLECTION AND CACHING
# ============================================================

def build_mars_image_records(split: str) -> pd.DataFrame:
    if split == "train":
        names = MARS_TRAIN_IMAGES
    elif split == "val":
        names = MARS_VAL_IMAGES
    elif split == "test":
        names = MARS_TEST_IMAGES
    else:
        raise ValueError(split)

    rows = []
    for i, name in enumerate(names):
        rows.append({
            "year": "MARS",
            "idx": i,
            "day_key": name,
            "image_id": name,
            "image_path": str(PROJECT_ROOT / "MARS" / "images" / f"{name}.png"),
            "mask_path": str(PROJECT_ROOT / "MARS" / "masks_multiclass" / f"{name}_multiclass.png"),
            "split": split,
        })
    return pd.DataFrame(rows)


def load_year_records_from_report(year: str) -> Optional[pd.DataFrame]:
    report_path = PROJECT_ROOT / year / "reports" / "selected_top100.csv"
    if not report_path.exists():
        return None

    df = pd.read_csv(report_path)
    rows = []

    for i, row in df.iterrows():
        image_path = row.get("image_path", "")
        mask_path = row.get("mask_png_path", row.get("mask_path", ""))
        day_key = str(row.get("day_key", Path(str(image_path)).stem))
        idx_value = row.get("idx", row.get("source_index", row.get("rank", i)))
        try:
            idx_value = int(idx_value)
        except Exception:
            idx_value = i

        rows.append({
            "year": year,
            "idx": idx_value,
            "day_key": day_key,
            "image_id": f"{year}_{day_key}",
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "source_index": i,
        })

    if not rows:
        return None

    return repair_pair_paths(pd.DataFrame(rows), PROJECT_ROOT)


def load_year_records_from_folders(year: str) -> pd.DataFrame:
    year_dir = PROJECT_ROOT / year
    image_folders = [
        year_dir / "selected_top100" / "images",
        year_dir / "images",
        year_dir / "mdgm_rgb",
    ]
    mask_folders = [
        year_dir / "selected_top100" / "masks",
        year_dir / "masks",
        year_dir / "masks_multiclass",
        year_dir / "masks_png_all",
    ]

    image_files = []
    for folder in image_folders:
        if folder.exists():
            for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]:
                image_files.extend(folder.glob(ext))
            if image_files:
                break

    mask_files = []
    for folder in mask_folders:
        if folder.exists():
            for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]:
                mask_files.extend(folder.glob(ext))
            if mask_files:
                break

    if not image_files or not mask_files:
        raise FileNotFoundError(f"No image/mask files found for {year}.")

    mask_by_key = {normalize_key(p.stem): p for p in mask_files}
    rows = []

    for i, image_path in enumerate(sorted(image_files)):
        image_key = normalize_key(image_path.stem)
        mask_path = mask_by_key.get(image_key)

        if mask_path is None:
            for mk, mp in mask_by_key.items():
                if image_key in mk or mk in image_key:
                    mask_path = mp
                    break

        if mask_path is None:
            continue

        rows.append({
            "year": year,
            "idx": i,
            "day_key": image_path.stem,
            "image_id": f"{year}_{image_path.stem}",
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "source_index": i,
        })

    if not rows:
        raise RuntimeError(f"No valid pairs matched for {year}.")

    return pd.DataFrame(rows)


def collect_multiyear_pairs(cache_dir: Path) -> pd.DataFrame:
    cache_path = cache_dir / "image_pairs.csv"
    if REUSE_CACHE and cache_path.exists() and not FORCE_REBUILD_CACHE:
        log(f"Loading cached image pairs: {cache_path}")
        return repair_pair_paths(pd.read_csv(cache_path), PROJECT_ROOT)

    dfs = []
    for year in YEARS:
        log(f"Collecting image-mask pairs for {year}...")
        df = load_year_records_from_report(year)
        if df is None:
            log(f"[INFO] No usable report for {year}. Matching folders.")
            df = load_year_records_from_folders(year)
        log(f"{year}: {len(df)} image-mask pairs found.")
        dfs.append(df)

    pairs = pd.concat(dfs, ignore_index=True)
    pairs = repair_pair_paths(pairs, PROJECT_ROOT)
    pairs.to_csv(cache_path, index=False)
    return pairs


def split_multiyear_pairs(pair_df: pd.DataFrame, split_mode: str) -> pd.DataFrame:
    parts = []

    for year in YEARS:
        sub = pair_df[pair_df["year"] == year].copy().reset_index(drop=True)

        if split_mode == "shuffle":
            sub = sub.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        elif split_mode == "sequential":
            sub = sub.sort_values("idx").reset_index(drop=True)
        else:
            raise ValueError(f"Unknown split_mode: {split_mode}")

        n = len(sub)
        n_train = int(round(0.70 * n))
        n_val = int(round(0.15 * n))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)

        sub["split"] = "test"
        sub.loc[:n_train - 1, "split"] = "train"
        sub.loc[n_train:n_train + n_val - 1, "split"] = "val"

        parts.append(sub)

    return pd.concat(parts, ignore_index=True)


def build_patch_inventory_for_split(
    image_df: pd.DataFrame,
    split_name: str,
    spec: Dict[str, Any],
    loader_cache: PairCache
) -> pd.DataFrame:
    rows = []
    patch_h = int(spec["patch_h"])
    patch_w = int(spec["patch_w"])
    stride_y = int(spec["stride_y"])
    stride_x = int(spec["stride_x"])
    min_valid_ratio = float(spec["min_valid_ratio"])

    for _, item in tqdm(
        image_df[image_df["split"] == split_name].iterrows(),
        total=int((image_df["split"] == split_name).sum()),
        desc=f"Building {spec['model_name']} {split_name} patches"
    ):
        _, mask = loader_cache.load(str(item["image_path"]), str(item["mask_path"]))
        h, w = mask.shape

        if h < patch_h or w < patch_w:
            continue

        for y0 in range(0, h - patch_h + 1, stride_y):
            for x0 in range(0, w - patch_w + 1, stride_x):
                y1 = y0 + patch_h
                x1 = x0 + patch_w

                mask_patch = mask[y0:y1, x0:x1]
                valid = mask_patch != 2
                valid_ratio = float(valid.mean())

                if valid_ratio < min_valid_ratio:
                    continue

                storm_ratio = float(np.mean(mask_patch[valid] == 1)) if valid.any() else 0.0
                storm_pixels = int((mask_patch == 1).sum())

                rows.append({
                    "year": item.get("year", ""),
                    "idx": item.get("idx", -1),
                    "day_key": item.get("day_key", ""),
                    "image_id": item.get("image_id", item.get("day_key", "")),
                    "split": split_name,
                    "image_path": str(item["image_path"]),
                    "mask_path": str(item["mask_path"]),
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "valid_ratio": valid_ratio,
                    "storm_ratio": storm_ratio,
                    "storm_pixels": storm_pixels,
                    "positive_patch": storm_pixels > 0,
                })

    inv = pd.DataFrame(rows)

    if inv.empty:
        raise RuntimeError(f"No patches generated for {spec['model_name']} split={split_name}")

    return inv


def compute_train_mean_std_from_images(image_df: pd.DataFrame, loader_cache: PairCache) -> Tuple[np.ndarray, np.ndarray]:
    total_sum = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    total_n = 0

    train_df = image_df[image_df["split"] == "train"].reset_index(drop=True)

    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Computing train mean/std"):
        rgb, mask = loader_cache.load(str(row["image_path"]), str(row["mask_path"]))
        valid = mask != 2
        pixels = rgb[valid].astype(np.float32) / 255.0
        if pixels.size == 0:
            continue
        total_sum += pixels.sum(axis=0)
        total_sq += (pixels ** 2).sum(axis=0)
        total_n += pixels.shape[0]

    if total_n == 0:
        raise RuntimeError("No valid pixels found to compute mean/std.")

    mean = total_sum / total_n
    var = total_sq / total_n - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-8))

    return mean.astype(np.float32), std.astype(np.float32)


def maybe_migrate_old_cbam_cache(cache_dir: Path, model_out_dir: Path) -> bool:
    old_dir = model_out_dir / "_temp_module_outputs"
    old_patch = old_dir / "patch_inventory_all_splits.csv"
    old_split = old_dir / "image_split_manifest.csv"

    if not old_patch.exists():
        return False

    log("Migrating existing CBAM patch inventory from previous run cache.")
    all_inv = pd.read_csv(old_patch)
    all_inv = normalize_patch_inventory_columns(repair_pair_paths(all_inv, PROJECT_ROOT))

    if old_split.exists():
        split_df = pd.read_csv(old_split)
        split_df = repair_pair_paths(split_df, PROJECT_ROOT)
        split_df.to_csv(cache_dir / "image_split_manifest.csv", index=False)

    for split_name in ["train", "val", "test"]:
        sub = all_inv[all_inv["split"] == split_name].reset_index(drop=True)
        if len(sub) > 0:
            sub.to_csv(cache_dir / f"patch_inventory_{split_name}.csv", index=False)

    all_inv.to_csv(cache_dir / "patch_inventory_all.csv", index=False)
    return True


def load_or_build_context(spec: Dict[str, Any], checkpoint: Any):
    model_name = spec["model_name"]
    model_out_dir = OUTPUT_ROOT / model_name
    cache_dir = model_out_dir / "cache"
    model_out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    mode = "cbam" if spec["data_source"] == "multiyear_cbam" else "raw"
    loader_cache = PairCache(mode=mode, max_items=16)

    split_csv = cache_dir / "image_split_manifest.csv"
    train_inv_csv = cache_dir / "patch_inventory_train.csv"
    val_inv_csv = cache_dir / "patch_inventory_val.csv"
    test_inv_csv = cache_dir / "patch_inventory_test.csv"
    mean_std_json = cache_dir / "train_mean_std.json"

    cache_ready = all(p.exists() for p in [split_csv, train_inv_csv, val_inv_csv, test_inv_csv])

    # Try migrating older generated CBAM inventory before rebuilding.
    if not cache_ready and spec["data_source"] == "multiyear_cbam":
        maybe_migrate_old_cbam_cache(cache_dir, model_out_dir)
        cache_ready = all(p.exists() for p in [train_inv_csv, val_inv_csv, test_inv_csv])

    if REUSE_CACHE and cache_ready and not FORCE_REBUILD_CACHE:
        log(f"Loading cached split and patch inventories for {model_name}.")
        if split_csv.exists():
            image_df = pd.read_csv(split_csv)
            image_df = repair_pair_paths(image_df, PROJECT_ROOT)
        else:
            image_df = pd.DataFrame()

        train_inv = normalize_patch_inventory_columns(repair_pair_paths(pd.read_csv(train_inv_csv), PROJECT_ROOT))
        val_inv = normalize_patch_inventory_columns(repair_pair_paths(pd.read_csv(val_inv_csv), PROJECT_ROOT))
        test_inv = normalize_patch_inventory_columns(repair_pair_paths(pd.read_csv(test_inv_csv), PROJECT_ROOT))

    else:
        if spec["data_source"] == "mars":
            image_df = pd.concat([
                build_mars_image_records("train"),
                build_mars_image_records("val"),
                build_mars_image_records("test"),
            ], ignore_index=True)
            image_df = repair_pair_paths(image_df, PROJECT_ROOT)

        elif spec["data_source"] in ["multiyear_raw", "multiyear_cbam"]:
            pair_df = collect_multiyear_pairs(cache_dir)
            image_df = split_multiyear_pairs(pair_df, split_mode=spec.get("split_mode", "shuffle"))
            image_df = repair_pair_paths(image_df, PROJECT_ROOT)
        else:
            raise ValueError(f"Unsupported data_source: {spec['data_source']}")

        image_df.to_csv(split_csv, index=False)

        train_inv = normalize_patch_inventory_columns(build_patch_inventory_for_split(image_df, "train", spec, loader_cache))
        val_inv = normalize_patch_inventory_columns(build_patch_inventory_for_split(image_df, "val", spec, loader_cache))
        test_inv = normalize_patch_inventory_columns(build_patch_inventory_for_split(image_df, "test", spec, loader_cache))

        train_inv.to_csv(train_inv_csv, index=False)
        val_inv.to_csv(val_inv_csv, index=False)
        test_inv.to_csv(test_inv_csv, index=False)
        pd.concat([train_inv, val_inv, test_inv], ignore_index=True).to_csv(cache_dir / "patch_inventory_all.csv", index=False)

    # Mean/std
    mean, std = infer_mean_std_from_checkpoint(
        checkpoint,
        spec.get("mean_default"),
        spec.get("std_default")
    )

    if mean is None or std is None:
        if REUSE_CACHE and mean_std_json.exists() and not FORCE_REBUILD_CACHE:
            data = json.loads(mean_std_json.read_text(encoding="utf-8"))
            mean = np.asarray(data["mean"], dtype=np.float32)
            std = np.asarray(data["std"], dtype=np.float32)
            log(f"Loading cached mean/std for {model_name}.")
        else:
            # Compute on training images only.
            if split_csv.exists():
                image_df = pd.read_csv(split_csv)
                image_df = repair_pair_paths(image_df, PROJECT_ROOT)
            else:
                raise RuntimeError("Cannot compute mean/std without image split manifest.")
            mean, std = compute_train_mean_std_from_images(image_df, loader_cache)
            mean_std_json.write_text(
                json.dumps({"mean": mean.tolist(), "std": std.tolist()}, indent=2),
                encoding="utf-8"
            )

    return {
        "image_df": image_df,
        "train_inv": train_inv,
        "val_inv": val_inv,
        "test_inv": test_inv,
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "loader_mode": mode,
        "cache_dir": cache_dir,
        "model_out_dir": model_out_dir,
    }


# ============================================================
# 5. DATASET
# ============================================================

class PatchDataset(Dataset):
    def __init__(self, patch_inventory: pd.DataFrame, mean: np.ndarray, std: np.ndarray, mode: str):
        self.inv = patch_inventory.reset_index(drop=True)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.cache = PairCache(mode=mode, max_items=16)

    def __len__(self):
        return len(self.inv)

    def __getitem__(self, idx):
        row = self.inv.iloc[idx]
        rgb, mask = self.cache.load(str(row["image_path"]), str(row["mask_path"]))

        x0 = int(row["x0"])
        y0 = int(row["y0"])
        x1 = int(row["x1"])
        y1 = int(row["y1"])

        rgb_patch = rgb[y0:y1, x0:x1].astype(np.float32) / 255.0
        mask_patch = mask[y0:y1, x0:x1].astype(np.uint8)

        target = (mask_patch == 1).astype(np.float32)
        valid = (mask_patch != 2).astype(np.float32)

        rgb_patch = (rgb_patch - self.mean) / (self.std + 1e-8)
        rgb_patch = np.transpose(rgb_patch, (2, 0, 1))

        return (
            torch.tensor(rgb_patch, dtype=torch.float32),
            torch.tensor(target[None, :, :], dtype=torch.float32),
            torch.tensor(valid[None, :, :], dtype=torch.float32),
        )


# ============================================================
# 6. MODEL DEFINITIONS
# ============================================================

class ConvBlockBN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


def group_norm_layer(num_channels):
    groups = min(8, num_channels)
    return nn.GroupNorm(num_groups=groups, num_channels=num_channels)


class ConvBlockGN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            group_norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            group_norm_layer(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGateGeneric(nn.Module):
    def __init__(self, gate_channels, skip_channels, intermediate_channels, norm_type="bn"):
        super().__init__()
        norm = nn.BatchNorm2d if norm_type == "bn" else group_norm_layer
        self.gate_conv = nn.Sequential(
            nn.Conv2d(gate_channels, intermediate_channels, kernel_size=1),
            norm(intermediate_channels)
        )
        self.skip_conv = nn.Sequential(
            nn.Conv2d(skip_channels, intermediate_channels, kernel_size=1),
            norm(intermediate_channels)
        )

        if norm_type == "bn":
            self.psi = nn.Sequential(
                nn.Conv2d(intermediate_channels, 1, kernel_size=1),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )
        else:
            self.psi = nn.Sequential(
                nn.Conv2d(intermediate_channels, 1, kernel_size=1),
                nn.Sigmoid()
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        attention = self.relu(self.gate_conv(gate) + self.skip_conv(skip))
        attention = self.psi(attention)
        return skip * attention


class AttentionUNetGeneric(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=16, norm_type="bn"):
        super().__init__()

        ConvBlock = ConvBlockBN if norm_type == "bn" else ConvBlockGN
        f1 = base_channels
        f2 = base_channels * 2
        f3 = base_channels * 4
        f4 = base_channels * 8
        f5 = base_channels * 16

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1 = ConvBlock(in_channels, f1)
        self.enc2 = ConvBlock(f1, f2)
        self.enc3 = ConvBlock(f2, f3)
        self.enc4 = ConvBlock(f3, f4)
        self.bottleneck = ConvBlock(f4, f5)

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(f5, f4, kernel_size=3, padding=1)
        )
        self.att4 = AttentionGateGeneric(f4, f4, f3, norm_type=norm_type)
        self.dec4 = ConvBlock(f4 + f4, f4)

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(f4, f3, kernel_size=3, padding=1)
        )
        self.att3 = AttentionGateGeneric(f3, f3, f2, norm_type=norm_type)
        self.dec3 = ConvBlock(f3 + f3, f3)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(f3, f2, kernel_size=3, padding=1)
        )
        self.att2 = AttentionGateGeneric(f2, f2, f1, norm_type=norm_type)
        self.dec2 = ConvBlock(f2 + f2, f2)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(f2, f1, kernel_size=3, padding=1)
        )
        self.att1 = AttentionGateGeneric(f1, f1, max(f1 // 2, 1), norm_type=norm_type)
        self.dec1 = ConvBlock(f1 + f1, f1)

        self.final = nn.Conv2d(f1, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        e4_att = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4_att], dim=1))

        d3 = self.up3(d4)
        e3_att = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3_att], dim=1))

        d2 = self.up2(d3)
        e2_att = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2_att], dim=1))

        d1 = self.up1(d2)
        e1_att = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1_att], dim=1))

        return self.final(d1)


class ConvBlockCBAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        attention = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))
        return x * attention


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * attention


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x):
        return self.spatial_attention(self.channel_attention(x))


class CBAMUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=16):
        super().__init__()
        f1 = base_channels
        f2 = base_channels * 2
        f3 = base_channels * 4
        f4 = base_channels * 8
        f5 = base_channels * 16

        self.pool = nn.MaxPool2d(2, 2)

        self.enc1 = ConvBlockCBAM(in_channels, f1)
        self.enc2 = ConvBlockCBAM(f1, f2)
        self.enc3 = ConvBlockCBAM(f2, f3)
        self.enc4 = ConvBlockCBAM(f3, f4)
        self.bottleneck = ConvBlockCBAM(f4, f5)

        self.cbam1 = CBAM(f1)
        self.cbam2 = CBAM(f2)
        self.cbam3 = CBAM(f3)
        self.cbam4 = CBAM(f4)

        self.up4_conv = nn.Conv2d(f5, f4, kernel_size=1)
        self.dec4 = ConvBlockCBAM(f4 + f4, f4)

        self.up3_conv = nn.Conv2d(f4, f3, kernel_size=1)
        self.dec3 = ConvBlockCBAM(f3 + f3, f3)

        self.up2_conv = nn.Conv2d(f3, f2, kernel_size=1)
        self.dec2 = ConvBlockCBAM(f2 + f2, f2)

        self.up1_conv = nn.Conv2d(f2, f1, kernel_size=1)
        self.dec1 = ConvBlockCBAM(f1 + f1, f1)

        self.final = nn.Conv2d(f1, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = F.interpolate(b, size=e4.shape[2:], mode="bilinear", align_corners=False)
        d4 = self.up4_conv(d4)
        d4 = self.dec4(torch.cat([d4, self.cbam4(e4)], dim=1))

        d3 = F.interpolate(d4, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.up3_conv(d3)
        d3 = self.dec3(torch.cat([d3, self.cbam3(e3)], dim=1))

        d2 = F.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.up2_conv(d2)
        d2 = self.dec2(torch.cat([d2, self.cbam2(e2)], dim=1))

        d1 = F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.up1_conv(d1)
        d1 = self.dec1(torch.cat([d1, self.cbam1(e1)], dim=1))

        return self.final(d1)


def create_model(spec: Dict[str, Any], checkpoint: Any):
    base_channels = infer_base_channels(checkpoint, spec["base_channels_default"])

    if spec["architecture"] == "attention_bn":
        model = AttentionUNetGeneric(base_channels=base_channels, norm_type="bn")
    elif spec["architecture"] == "attention_gn":
        model = AttentionUNetGeneric(base_channels=base_channels, norm_type="gn")
    elif spec["architecture"] == "cbam":
        model = CBAMUNet(base_channels=base_channels)
    else:
        raise ValueError(f"Unknown architecture: {spec['architecture']}")

    state_dict = extract_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()

    return model, base_channels


# ============================================================
# 7. EVALUATION
# ============================================================

def compute_counts_for_thresholds(model, loader, thresholds: List[float]) -> pd.DataFrame:
    thresholds = [float(t) for t in thresholds]

    counts = {
        t: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for t in thresholds
    }

    model.eval()

    with torch.no_grad():
        for images, targets, valid in tqdm(loader, desc="Evaluating thresholds"):
            images = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            valid = valid.to(DEVICE, non_blocking=True)

            logits = model(images)
            probs = torch.sigmoid(logits)

            valid_bool = valid > 0.5

            for t in thresholds:
                preds = (probs >= t).float()

                counts[t]["tp"] += int(((preds == 1) & (targets == 1) & valid_bool).sum().item())
                counts[t]["fp"] += int(((preds == 1) & (targets == 0) & valid_bool).sum().item())
                counts[t]["fn"] += int(((preds == 0) & (targets == 1) & valid_bool).sum().item())
                counts[t]["tn"] += int(((preds == 0) & (targets == 0) & valid_bool).sum().item())

    rows = []
    for t in thresholds:
        tp = counts[t]["tp"]
        fp = counts[t]["fp"]
        fn = counts[t]["fn"]
        tn = counts[t]["tn"]

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

        rows.append({
            "threshold": t,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        })

    return pd.DataFrame(rows)


def select_best_threshold(val_sweep: pd.DataFrame) -> float:
    # Maximize IoU; if tied, prefer higher F1, then higher precision.
    sorted_df = val_sweep.sort_values(
        ["iou", "f1", "precision"],
        ascending=[False, False, False]
    ).reset_index(drop=True)
    return float(sorted_df.loc[0, "threshold"])


# ============================================================
# 8. VISUALIZATION
# ============================================================

def choose_visual_patch(inv: pd.DataFrame) -> pd.Series:
    lo, hi = TARGET_STORM_RANGE
    subset = inv[(inv["storm_ratio"] >= lo) & (inv["storm_ratio"] <= hi)].copy()

    if subset.empty:
        subset = inv[inv["storm_ratio"] > 0].copy()

    if subset.empty:
        return inv.iloc[0]

    subset["selection_score"] = (subset["storm_ratio"] - TARGET_STORM_CENTER).abs()
    return subset.sort_values("selection_score").iloc[0]


def predict_single_patch(model, rgb_patch: np.ndarray, mean: np.ndarray, std: np.ndarray, threshold: float):
    arr = rgb_patch.astype(np.float32) / 255.0
    arr = (arr - mean) / (std + 1e-8)
    arr = np.transpose(arr, (2, 0, 1))
    x = torch.tensor(arr[None, :, :, :], dtype=torch.float32).to(DEVICE)

    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].detach().cpu().numpy()

    pred = (prob >= threshold).astype(np.uint8)
    return prob, pred


def plot_confusion_matrix(metrics: Dict[str, Any], out_path: Path, model_name: str):
    cm = np.array([
        [int(metrics["tn"]), int(metrics["fp"])],
        [int(metrics["fn"]), int(metrics["tp"])]
    ])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="viridis")

    ax.set_title(
        f"{model_name}\n"
        f"threshold={metrics['threshold']:.2f} | IoU={metrics['iou']:.4f} | F1={metrics['f1']:.4f}",
        fontsize=14
    )
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred no storm", "Pred storm"])
    ax.set_yticklabels(["True no storm", "True storm"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="black", fontsize=12)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(
        cm,
        index=["True no storm", "True storm"],
        columns=["Pred no storm", "Pred storm"]
    ).to_csv(out_path.with_suffix(".csv"))


def plot_visual_example(
    model,
    inv: pd.DataFrame,
    mean: np.ndarray,
    std: np.ndarray,
    loader_mode: str,
    threshold: float,
    out_path: Path
):
    row = choose_visual_patch(inv)
    pair_cache = PairCache(mode=loader_mode, max_items=4)
    rgb, mask = pair_cache.load(str(row["image_path"]), str(row["mask_path"]))

    x0 = int(row["x0"])
    y0 = int(row["y0"])
    x1 = int(row["x1"])
    y1 = int(row["y1"])

    rgb_patch = rgb[y0:y1, x0:x1]
    mask_patch = mask[y0:y1, x0:x1]

    _, pred = predict_single_patch(model, rgb_patch, mean, std, threshold)
    pred_display = pred.astype(np.uint8)
    pred_display[mask_patch == 2] = 2

    gt_display = mask_patch.copy().astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    title_parts = []
    if "year" in row:
        title_parts.append(str(row["year"]))
    if "day_key" in row:
        title_parts.append(str(row["day_key"]))
    storm_ratio_value = float(row["storm_ratio"]) if "storm_ratio" in row.index else 0.0
    title_parts.append(f"storm={storm_ratio_value * 100:.1f}%")

    axes[0].imshow(rgb_patch)
    axes[0].set_title("Original RGB\n" + " | ".join(title_parts), fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(gt_display, cmap=CLASS_CMAP, vmin=0, vmax=2)
    axes[1].set_title("Ground truth\n0=no storm | 1=storm | 2=invalid", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(pred_display, cmap=CLASS_CMAP, vmin=0, vmax=2)
    axes[2].set_title(f"Prediction\nthreshold={threshold:.2f}", fontsize=11)
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(row.to_dict(), f, indent=2, default=str)


def plot_summary_bars(summary_df: pd.DataFrame, out_path: Path):
    metrics = ["iou", "f1", "precision", "recall"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()

    for ax, metric in zip(axes, metrics):
        ax.bar(summary_df["model_name"], summary_df[metric])
        ax.set_title(metric.upper())
        ax.set_ylim(0, max(0.1, float(summary_df[metric].max()) * 1.25))
        ax.tick_params(axis="x", rotation=25)
        for i, value in enumerate(summary_df[metric]):
            ax.text(i, value + 0.002, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_visual_grid(summary_df: pd.DataFrame, out_root: Path):
    rows = []
    for model_name in summary_df["model_name"]:
        img_path = out_root / model_name / "selected_visual_example.png"
        if img_path.exists():
            rows.append((model_name, plt.imread(str(img_path))))

    if not rows:
        return

    fig, axes = plt.subplots(len(rows), 1, figsize=(16, 5 * len(rows)))
    if len(rows) == 1:
        axes = [axes]

    for ax, (model_name, img) in zip(axes, rows):
        ax.imshow(img)
        ax.set_title(model_name, fontsize=14)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_root / "comparison_visual_grid.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 9. MAIN
# ============================================================

def evaluate_one_model(spec: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    model_name = spec["model_name"]
    out_dir = OUTPUT_ROOT / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    log("\n" + "=" * 72)
    log(f"Evaluating: {model_name}")

    checkpoint_path = find_checkpoint(spec["checkpoint_names"])
    log(f"Checkpoint: {checkpoint_path}")

    checkpoint = safe_torch_load(checkpoint_path)
    model, base_channels = create_model(spec, checkpoint)

    context = load_or_build_context(spec, checkpoint)

    mean = context["mean"]
    std = context["std"]
    loader_mode = context["loader_mode"]

    val_ds = PatchDataset(context["val_inv"], mean, std, mode=loader_mode)
    test_ds = PatchDataset(context["test_inv"], mean, std, mode=loader_mode)

    batch_size = EVAL_BATCH_SIZE_CBAM if spec["architecture"] == "cbam" else EVAL_BATCH_SIZE_ATTENTION

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS_EVAL,
        pin_memory=(DEVICE.type == "cuda")
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS_EVAL,
        pin_memory=(DEVICE.type == "cuda")
    )

    # Validation threshold sweep.
    val_sweep_path = out_dir / "validation_threshold_sweep.csv"
    if REUSE_CACHE and val_sweep_path.exists() and not FORCE_REBUILD_CACHE:
        log(f"Loading cached validation threshold sweep: {val_sweep_path}")
        val_sweep = pd.read_csv(val_sweep_path)
    else:
        val_sweep = compute_counts_for_thresholds(model, val_loader, spec["thresholds"])
        val_sweep.to_csv(val_sweep_path, index=False)

    best_threshold = select_best_threshold(val_sweep)
    log(f"Validation-selected threshold: {best_threshold:.2f}")

    # Test threshold sweep.
    test_sweep_path = out_dir / "test_threshold_sweep.csv"
    if REUSE_CACHE and test_sweep_path.exists() and not FORCE_REBUILD_CACHE:
        log(f"Loading cached test threshold sweep: {test_sweep_path}")
        test_sweep = pd.read_csv(test_sweep_path)
    else:
        test_sweep = compute_counts_for_thresholds(model, test_loader, spec["thresholds"])
        test_sweep.to_csv(test_sweep_path, index=False)

    # Formal metrics: test at validation-selected threshold.
    idx = (test_sweep["threshold"] - best_threshold).abs().idxmin()
    test_metrics = test_sweep.loc[idx].to_dict()
    test_metrics["threshold"] = float(test_metrics["threshold"])
    test_metrics["model_name"] = model_name
    test_metrics["checkpoint_path"] = str(checkpoint_path)

    # Diagnostic only: best threshold on test itself.
    best_test = test_sweep.sort_values(["iou", "f1", "precision"], ascending=[False, False, False]).iloc[0].to_dict()

    pd.DataFrame([test_metrics]).to_csv(out_dir / "test_metrics.csv", index=False)
    pd.DataFrame([best_test]).to_csv(out_dir / "test_metrics_best_test_threshold_diagnostic.csv", index=False)

    plot_confusion_matrix(test_metrics, out_dir / "test_confusion_matrix.png", model_name)
    plot_visual_example(
        model=model,
        inv=context["test_inv"],
        mean=mean,
        std=std,
        loader_mode=loader_mode,
        threshold=float(test_metrics["threshold"]),
        out_path=out_dir / "selected_visual_example.png"
    )

    param_info = {
        "model_name": model_name,
        "architecture": spec["architecture"],
        "data_source": spec["data_source"],
        "checkpoint": str(checkpoint_path),
        "base_channels": base_channels,
        "patch_h": spec["patch_h"],
        "patch_w": spec["patch_w"],
        "stride_y": spec["stride_y"],
        "stride_x": spec["stride_x"],
        "min_valid_ratio": spec["min_valid_ratio"],
        "mean": ",".join([f"{v:.6f}" for v in mean]),
        "std": ",".join([f"{v:.6f}" for v in std]),
        "val_selected_threshold": best_threshold,
        "best_test_threshold_diagnostic": float(best_test["threshold"]),
    }

    log(
        f"Done -> IoU={test_metrics['iou']:.4f} | "
        f"F1={test_metrics['f1']:.4f} | "
        f"Prec={test_metrics['precision']:.4f} | "
        f"Rec={test_metrics['recall']:.4f}"
    )

    return test_metrics, param_info


def main():
    log(f"Project root : {PROJECT_ROOT}")
    log(f"Models dir   : {MODELS_DIR}")
    log(f"Output dir   : {OUTPUT_ROOT}")
    log(f"Device       : {DEVICE}")
    log(f"Reuse cache  : {REUSE_CACHE}")
    log(f"Force rebuild: {FORCE_REBUILD_CACHE}")

    metrics_rows = []
    param_rows = []
    error_rows = []

    for spec in MODEL_REGISTRY:
        try:
            metrics, params = evaluate_one_model(spec)
            metrics_rows.append(metrics)
            param_rows.append(params)
        except Exception as exc:
            log(f"[ERROR] {spec['model_name']}: {exc}")
            error_rows.append({
                "model_name": spec["model_name"],
                "error": str(exc),
            })

    if metrics_rows:
        summary_df = pd.DataFrame(metrics_rows)
        preferred_cols = [
            "model_name", "threshold", "accuracy", "precision", "recall", "f1", "iou",
            "tn", "fp", "fn", "tp", "checkpoint_path"
        ]
        summary_df = summary_df[[c for c in preferred_cols if c in summary_df.columns]]
        summary_df = summary_df.sort_values("iou", ascending=False).reset_index(drop=True)
        summary_df.to_csv(OUTPUT_ROOT / "comparison_metrics_summary.csv", index=False)
        plot_summary_bars(summary_df, OUTPUT_ROOT / "comparison_summary_bars.png")
        plot_visual_grid(summary_df, OUTPUT_ROOT)

    if param_rows:
        pd.DataFrame(param_rows).to_csv(OUTPUT_ROOT / "comparison_model_parameters.csv", index=False)

    if error_rows:
        pd.DataFrame(error_rows).to_csv(OUTPUT_ROOT / "comparison_errors.csv", index=False)
    else:
        error_path = OUTPUT_ROOT / "comparison_errors.csv"
        if error_path.exists():
            error_path.unlink()

    report_path = OUTPUT_ROOT / "comparison_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Model comparison report\n\n")
        f.write(f"Device used: `{DEVICE}`\n\n")
        f.write(f"Cache reused: `{REUSE_CACHE}`\n\n")

        if param_rows:
            f.write("## Model parameters\n\n```text\n")
            f.write(pd.DataFrame(param_rows).to_string(index=False))
            f.write("\n```\n\n")

        if metrics_rows:
            f.write("## Test metrics\n\n```text\n")
            f.write(pd.DataFrame(metrics_rows)[
                [c for c in ["model_name", "threshold", "accuracy", "precision", "recall", "f1", "iou", "tn", "fp", "fn", "tp"]
                 if c in pd.DataFrame(metrics_rows).columns]
            ].to_string(index=False))
            f.write("\n```\n\n")

        if error_rows:
            f.write("## Errors\n\n```text\n")
            f.write(pd.DataFrame(error_rows).to_string(index=False))
            f.write("\n```\n")

    log("\n" + "=" * 72)
    log("Comparison completed.")
    log(f"Results saved in: {OUTPUT_ROOT}")

    if error_rows:
        log("Some models produced errors. See comparison_errors.csv")


if __name__ == "__main__":
    main()
