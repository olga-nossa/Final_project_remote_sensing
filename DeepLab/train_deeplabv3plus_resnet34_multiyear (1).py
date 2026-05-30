from __future__ import annotations

# ============================================================
# DEEPLABV3+ RESNET34 MULTI-YEAR TRAINING
# Martian Dust Storm Segmentation using MDGM RGB + MDAD masks
# ------------------------------------------------------------
# Main idea:
# - Use MY28 + MY31 + MY32 selected_top100 images.
# - Split by FULL IMAGE first: 70% train, 15% val, 15% test per year.
# - Build 512x512 patches with 50% overlap.
# - Train DeepLabV3+ with a ResNet34 ImageNet-pretrained encoder.
# - Use balanced sampling to reduce no-storm dominance.
# - Use masked BCE + precision-oriented Tversky loss.
# - Select threshold by validation IoU.
# - Evaluate both patch-level and full-image test performance.
#
# Mask convention:
# 0 = no storm
# 1 = storm
# 2 = invalid/no-data
#
# Expected structure:
# ProjectFolder/
# ├── train_deeplabv3plus_resnet34_multiyear.py
# ├── MY28/selected_top100/images/
# ├── MY28/selected_top100/masks/
# ├── MY31/selected_top100/images/
# ├── MY31/selected_top100/masks/
# ├── MY32/selected_top100/images/
# └── MY32/selected_top100/masks/
#
# Optional:
# MYXX/reports/selected_top100.csv can exist and will be used first.
#
# Required packages:
# pip install segmentation-models-pytorch timm
# ============================================================

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
from matplotlib.colors import ListedColormap
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

try:
    import segmentation_models_pytorch as smp
except ImportError as exc:
    raise ImportError(
        "\nMissing package: segmentation_models_pytorch\n\n"
        "Install it with:\n"
        "    pip install segmentation-models-pytorch timm\n\n"
        "Then run this script again.\n"
    ) from exc


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT_DIR = Path(__file__).resolve().parent
YEARS = ["MY28", "MY31", "MY32"]

RUN_NAME = "deeplabv3plus_resnet34_multiyear_p512_s256"
OUTPUT_DIR = ROOT_DIR / f"{RUN_NAME}_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = OUTPUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# Data split
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
SPLIT_MODE = "shuffle"  # "shuffle" recommended.

# Patch settings
PATCH_SIZE = 512
STRIDE = 256
MIN_VALID_RATIO = 0.70

# Model
ARCHITECTURE = "DeepLabV3Plus"
ENCODER_NAME = "resnet34"
ENCODER_WEIGHTS = "imagenet"
IN_CHANNELS = 3
CLASSES = 1

# Training
BATCH_SIZE = 16                 # RTX 4070 Laptop 8 GB: start with 2. Try 3/4 if VRAM allows.
ACCUM_STEPS = 1                # effective batch = BATCH_SIZE * ACCUM_STEPS
NUM_EPOCHS = 35
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 6                # if Windows/DataLoader issues, set to 0
USE_AMP = True

# Sampling
SAMPLES_PER_EPOCH = 8000       # controls epoch duration; increase if training is stable
NO_STORM_WEIGHT = 0.25
LOW_STORM_WEIGHT = 1.0
MEDIUM_STORM_WEIGHT = 3.0
HIGH_STORM_WEIGHT = 5.0

# Loss
POS_WEIGHT = 4.0               # lower than 8/23 to reduce oversegmentation
LOSS_BCE_WEIGHT = 0.50
LOSS_TVERSKY_WEIGHT = 0.50
TVERSKY_ALPHA = 0.70           # penalizes false positives more
TVERSKY_BETA = 0.30

# Validation / threshold
THRESHOLDS = np.round(np.arange(0.10, 0.91, 0.05), 2).tolist()
EARLY_STOP_PATIENCE = 8
LR_PATIENCE = 3
LR_FACTOR = 0.5

# Full-image evaluation
FULL_IMAGE_TILE_SIZE = 512
FULL_IMAGE_TILE_STRIDE = 256
N_TEST_VISUALIZATIONS = 5

# Cache behavior
REUSE_CACHE = True
FORCE_REBUILD_CACHE = False

# Output files
BEST_MODEL_PATH = OUTPUT_DIR / "best_deeplabv3plus_resnet34_multiyear.pth"
HISTORY_CSV = OUTPUT_DIR / "training_history.csv"
SPLIT_CSV = CACHE_DIR / "image_split_manifest.csv"
PATCH_CSV = CACHE_DIR / "patch_inventory_all_splits.csv"
TRAIN_PATCH_CSV = CACHE_DIR / "patch_inventory_train.csv"
VAL_PATCH_CSV = CACHE_DIR / "patch_inventory_val.csv"
TEST_PATCH_CSV = CACHE_DIR / "patch_inventory_test.csv"
TRAIN_STATS_JSON = CACHE_DIR / "train_mean_std.json"

CLASS_CMAP = ListedColormap(["#440154", "#2A9D9A", "#FDE725"])


# ============================================================
# 2. BASIC UTILITIES
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
        .replace("_class", "")
        .replace("_storm", "")
        .replace("_", "")
        .replace("-", "")
        .replace(".", "")
        .replace(" ", "")
        .replace("/", "")
        .replace("\\", "")
    )


def safe_filename(text: Any) -> str:
    text = str(text)
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("L"), dtype=np.uint8)


def resize_mask_to_rgb(mask: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    if mask.shape == rgb.shape[:2]:
        return mask
    mask_img = Image.fromarray(mask.astype(np.uint8), mode="L")
    mask_img = mask_img.resize((rgb.shape[1], rgb.shape[0]), resample=Image.Resampling.NEAREST)
    return np.asarray(mask_img, dtype=np.uint8)


def get_autocast_context():
    if DEVICE.type == "cuda" and USE_AMP:
        return torch.amp.autocast("cuda")
    return torch.amp.autocast("cpu", enabled=False)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


# ============================================================
# 3. DATA PAIRING
# ============================================================

def find_existing_dir(candidates: List[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists() and path.is_dir():
            return path
    return None


def find_file_by_name_or_stem(folders: List[Path], preferred_name: str, stem_tokens: List[str]) -> Optional[Path]:
    for folder in folders:
        if not folder.exists():
            continue
        direct = folder / preferred_name
        if direct.exists():
            return direct
        matches = sorted(folder.rglob(preferred_name))
        if matches:
            return matches[0]

    tokens = [normalize_key(t) for t in stem_tokens if str(t) and str(t).lower() != "nan"]
    if not tokens:
        return None

    all_files = []
    for folder in folders:
        if not folder.exists():
            continue
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]:
            all_files.extend(folder.rglob(ext))

    for token in tokens:
        for file in sorted(all_files):
            key = normalize_key(file.stem)
            if token and (token in key or key in token):
                return file

    return None


def repair_pair_paths(pair_df: pd.DataFrame) -> pd.DataFrame:
    out = pair_df.copy()

    for idx, row in out.iterrows():
        year = str(row["year"])
        year_dir = ROOT_DIR / year

        image_text = str(row.get("image_path", ""))
        mask_text = str(row.get("mask_path", row.get("mask_png_path", "")))

        image_path = Path(image_text)
        mask_path = Path(mask_text)

        if not image_path.is_absolute():
            image_path = ROOT_DIR / image_path
        if not mask_path.is_absolute():
            mask_path = ROOT_DIR / mask_path

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
            fixed = find_file_by_name_or_stem(
                image_folders,
                preferred_name=Path(image_text).name,
                stem_tokens=[day_key, image_stem],
            )
            if fixed is not None:
                image_path = fixed

        if not mask_path.exists():
            fixed = find_file_by_name_or_stem(
                mask_folders,
                preferred_name=Path(mask_text).name,
                stem_tokens=[day_key, image_path.stem, image_stem, mask_stem],
            )
            if fixed is not None:
                mask_path = fixed

        out.at[idx, "image_path"] = str(image_path)
        out.at[idx, "mask_path"] = str(mask_path)

    missing_images = out[~out["image_path"].map(lambda p: Path(str(p)).exists())]
    missing_masks = out[~out["mask_path"].map(lambda p: Path(str(p)).exists())]

    if len(missing_images) > 0:
        raise FileNotFoundError(
            "Some image paths could not be repaired. First examples:\n"
            + "\n".join(missing_images["image_path"].head(5).astype(str).tolist())
        )
    if len(missing_masks) > 0:
        raise FileNotFoundError(
            "Some mask paths could not be repaired. First examples:\n"
            + "\n".join(missing_masks["mask_path"].head(5).astype(str).tolist())
        )

    return out


def collect_pairs_from_report(year: str) -> Optional[pd.DataFrame]:
    report_path = ROOT_DIR / year / "reports" / "selected_top100.csv"
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
            "source": "report",
            "source_index": i,
        })

    if not rows:
        return None

    repaired = repair_pair_paths(pd.DataFrame(rows))
    return repaired if len(repaired) > 0 else None


def collect_pairs_from_folders(year: str) -> pd.DataFrame:
    year_dir = ROOT_DIR / year

    image_dir = find_existing_dir([
        year_dir / "selected_top100" / "images",
        year_dir / "images",
        year_dir / "mdgm_rgb",
    ])
    mask_dir = find_existing_dir([
        year_dir / "selected_top100" / "masks",
        year_dir / "masks",
        year_dir / "masks_multiclass",
        year_dir / "masks_png_all",
    ])

    if image_dir is None or mask_dir is None:
        raise FileNotFoundError(f"Could not find image or mask folders for {year}.")

    image_files = []
    mask_files = []

    for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]:
        image_files.extend(sorted(image_dir.glob(ext)))
        mask_files.extend(sorted(mask_dir.glob(ext)))

    if not image_files:
        raise FileNotFoundError(f"No RGB images found for {year}.")
    if not mask_files:
        raise FileNotFoundError(f"No masks found for {year}.")

    mask_by_key = {normalize_key(p.stem): p for p in mask_files}
    rows = []

    for i, img_path in enumerate(sorted(image_files)):
        img_key = normalize_key(img_path.stem)
        mask_path = mask_by_key.get(img_key)

        if mask_path is None:
            for mk, mp in mask_by_key.items():
                if img_key in mk or mk in img_key:
                    mask_path = mp
                    break

        if mask_path is None:
            continue

        rows.append({
            "year": year,
            "idx": i,
            "day_key": img_path.stem,
            "image_id": f"{year}_{img_path.stem}",
            "image_path": str(img_path),
            "mask_path": str(mask_path),
            "source": "folder",
            "source_index": i,
        })

    if not rows:
        raise RuntimeError(f"No image-mask pairs matched for {year}.")

    return pd.DataFrame(rows)


def collect_all_pairs() -> pd.DataFrame:
    cache_path = CACHE_DIR / "image_pairs.csv"

    if REUSE_CACHE and cache_path.exists() and not FORCE_REBUILD_CACHE:
        log(f"Loading cached image pairs: {cache_path}")
        return repair_pair_paths(pd.read_csv(cache_path))

    all_years = []
    for year in YEARS:
        log(f"Collecting image-mask pairs for {year}...")
        df = collect_pairs_from_report(year)
        if df is not None:
            log(f"[INFO] Using {year}/reports/selected_top100.csv")
        else:
            log(f"[INFO] No valid report for {year}; matching folders.")
            df = collect_pairs_from_folders(year)

        log(f"{year}: {len(df)} pairs found.")
        all_years.append(df)

    pair_df = pd.concat(all_years, ignore_index=True)
    pair_df = repair_pair_paths(pair_df)
    pair_df.to_csv(cache_path, index=False)
    return pair_df


def split_pairs_by_year(pair_df: pd.DataFrame) -> pd.DataFrame:
    if REUSE_CACHE and SPLIT_CSV.exists() and not FORCE_REBUILD_CACHE:
        log(f"Loading cached image split: {SPLIT_CSV}")
        return repair_pair_paths(pd.read_csv(SPLIT_CSV))

    parts = []

    for year in YEARS:
        sub = pair_df[pair_df["year"] == year].copy().reset_index(drop=True)

        if SPLIT_MODE == "shuffle":
            sub = sub.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        elif SPLIT_MODE == "sequential":
            sub = sub.sort_values("idx").reset_index(drop=True)
        else:
            raise ValueError(f"Unknown SPLIT_MODE: {SPLIT_MODE}")

        n = len(sub)
        n_train = int(round(TRAIN_FRAC * n))
        n_val = int(round(VAL_FRAC * n))

        # Keep exact consistency.
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)

        sub["split"] = "test"
        sub.loc[:n_train - 1, "split"] = "train"
        sub.loc[n_train:n_train + n_val - 1, "split"] = "val"

        log(f"{year} split -> train={(sub['split']=='train').sum()}, val={(sub['split']=='val').sum()}, test={(sub['split']=='test').sum()}")
        parts.append(sub)

    split_df = pd.concat(parts, ignore_index=True)
    split_df.to_csv(SPLIT_CSV, index=False)
    return split_df


# ============================================================
# 4. PATCH INVENTORY
# ============================================================

class PairCache:
    def __init__(self, max_items: int = 12):
        self.max_items = max_items
        self.cache = OrderedDict()

    def load(self, image_path: str, mask_path: str) -> Tuple[np.ndarray, np.ndarray]:
        key = (image_path, mask_path)

        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value

        rgb = read_rgb(Path(image_path))
        mask = read_mask(Path(mask_path))
        mask = resize_mask_to_rgb(mask, rgb)

        self.cache[key] = (rgb, mask)
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)

        return rgb, mask


def patch_category(storm_ratio: float) -> str:
    if storm_ratio == 0.0:
        return "no_storm"
    if storm_ratio < 0.05:
        return "low_storm"
    if storm_ratio < 0.20:
        return "medium_storm"
    return "high_storm"


def get_patch_starts(length: int, patch_size: int, stride: int) -> List[int]:
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    if starts[-1] != length - patch_size:
        starts.append(length - patch_size)
    return starts


def build_patch_inventory_for_split(split_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    cache = PairCache(max_items=8)
    rows = []

    subset = split_df[split_df["split"] == split_name].reset_index(drop=True)

    for _, item in tqdm(subset.iterrows(), total=len(subset), desc=f"Building {split_name} patches"):
        rgb, mask = cache.load(str(item["image_path"]), str(item["mask_path"]))
        h, w = mask.shape

        if h < PATCH_SIZE or w < PATCH_SIZE:
            continue

        ys = get_patch_starts(h, PATCH_SIZE, STRIDE)
        xs = get_patch_starts(w, PATCH_SIZE, STRIDE)

        for y0 in ys:
            for x0 in xs:
                y1 = y0 + PATCH_SIZE
                x1 = x0 + PATCH_SIZE

                mask_patch = mask[y0:y1, x0:x1]
                valid = mask_patch != 2
                valid_ratio = float(valid.mean())

                if valid_ratio < MIN_VALID_RATIO:
                    continue

                storm_ratio = float(np.mean(mask_patch[valid] == 1)) if valid.any() else 0.0
                storm_pixels = int((mask_patch == 1).sum())

                rows.append({
                    "year": item["year"],
                    "idx": item.get("idx", -1),
                    "day_key": item["day_key"],
                    "image_id": item.get("image_id", f"{item['year']}_{item['day_key']}"),
                    "split": split_name,
                    "image_path": str(item["image_path"]),
                    "mask_path": str(item["mask_path"]),
                    "x0": int(x0),
                    "y0": int(y0),
                    "x1": int(x1),
                    "y1": int(y1),
                    "valid_ratio": valid_ratio,
                    "storm_ratio": storm_ratio,
                    "storm_pixels": storm_pixels,
                    "category": patch_category(storm_ratio),
                    "positive_patch": storm_pixels > 0,
                })

    inv = pd.DataFrame(rows)
    if inv.empty:
        raise RuntimeError(f"No patches generated for split={split_name}")

    return inv


def load_or_build_patch_inventories(split_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = [TRAIN_PATCH_CSV, VAL_PATCH_CSV, TEST_PATCH_CSV, PATCH_CSV]

    if REUSE_CACHE and all(p.exists() for p in required) and not FORCE_REBUILD_CACHE:
        log("Loading cached patch inventories.")
        train_inv = repair_pair_paths(pd.read_csv(TRAIN_PATCH_CSV))
        val_inv = repair_pair_paths(pd.read_csv(VAL_PATCH_CSV))
        test_inv = repair_pair_paths(pd.read_csv(TEST_PATCH_CSV))
        return train_inv, val_inv, test_inv

    train_inv = build_patch_inventory_for_split(split_df, "train")
    val_inv = build_patch_inventory_for_split(split_df, "val")
    test_inv = build_patch_inventory_for_split(split_df, "test")

    train_inv.to_csv(TRAIN_PATCH_CSV, index=False)
    val_inv.to_csv(VAL_PATCH_CSV, index=False)
    test_inv.to_csv(TEST_PATCH_CSV, index=False)

    all_inv = pd.concat([train_inv, val_inv, test_inv], ignore_index=True)
    all_inv.to_csv(PATCH_CSV, index=False)

    return train_inv, val_inv, test_inv


# ============================================================
# 5. NORMALIZATION
# ============================================================

def compute_train_mean_std(split_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    if REUSE_CACHE and TRAIN_STATS_JSON.exists() and not FORCE_REBUILD_CACHE:
        data = json.loads(TRAIN_STATS_JSON.read_text(encoding="utf-8"))
        return np.asarray(data["mean"], dtype=np.float32), np.asarray(data["std"], dtype=np.float32)

    total_sum = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    total_n = 0
    cache = PairCache(max_items=4)

    train_df = split_df[split_df["split"] == "train"].reset_index(drop=True)

    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Computing train mean/std"):
        rgb, mask = cache.load(str(row["image_path"]), str(row["mask_path"]))
        valid = mask != 2
        pixels = rgb[valid].astype(np.float32) / 255.0

        if pixels.size == 0:
            continue

        total_sum += pixels.sum(axis=0)
        total_sq += (pixels ** 2).sum(axis=0)
        total_n += pixels.shape[0]

    if total_n == 0:
        raise RuntimeError("No valid pixels found for mean/std calculation.")

    mean = total_sum / total_n
    var = total_sq / total_n - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-8))

    mean = mean.astype(np.float32)
    std = np.maximum(std.astype(np.float32), 1e-6)

    TRAIN_STATS_JSON.write_text(
        json.dumps({"mean": mean.tolist(), "std": std.tolist()}, indent=2),
        encoding="utf-8"
    )

    return mean, std


# ============================================================
# 6. DATASET AND AUGMENTATION
# ============================================================

def random_augment(rgb: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Horizontal flip
    if random.random() < 0.5:
        rgb = np.fliplr(rgb).copy()
        mask = np.fliplr(mask).copy()

    # Vertical flip
    if random.random() < 0.3:
        rgb = np.flipud(rgb).copy()
        mask = np.flipud(mask).copy()

    # 90-degree rotations
    if random.random() < 0.5:
        k = random.randint(0, 3)
        if k > 0:
            rgb = np.rot90(rgb, k).copy()
            mask = np.rot90(mask, k).copy()

    # Mild brightness/contrast augmentation
    if random.random() < 0.6:
        rgb_f = rgb.astype(np.float32)
        contrast = random.uniform(0.85, 1.15)
        brightness = random.uniform(-18.0, 18.0)
        rgb_f = (rgb_f - 127.5) * contrast + 127.5 + brightness
        rgb = np.clip(rgb_f, 0, 255).astype(np.uint8)

    return rgb, mask


class StormPatchDataset(Dataset):
    def __init__(
        self,
        inventory: pd.DataFrame,
        mean: np.ndarray,
        std: np.ndarray,
        augment: bool = False,
        cache_items: int = 12,
    ):
        self.inv = inventory.reset_index(drop=True)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.augment = augment
        self.cache = PairCache(max_items=cache_items)

    def __len__(self):
        return len(self.inv)

    def __getitem__(self, idx):
        row = self.inv.iloc[idx]
        rgb, mask = self.cache.load(str(row["image_path"]), str(row["mask_path"]))

        x0 = int(row["x0"])
        y0 = int(row["y0"])
        x1 = int(row["x1"])
        y1 = int(row["y1"])

        rgb_patch = rgb[y0:y1, x0:x1].copy()
        mask_patch = mask[y0:y1, x0:x1].copy()

        if self.augment:
            rgb_patch, mask_patch = random_augment(rgb_patch, mask_patch)

        image = rgb_patch.astype(np.float32) / 255.0
        image = (image - self.mean) / (self.std + 1e-8)
        image = np.transpose(image, (2, 0, 1))

        target = (mask_patch == 1).astype(np.float32)
        valid = (mask_patch != 2).astype(np.float32)

        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(target[None, :, :], dtype=torch.float32),
            torch.tensor(valid[None, :, :], dtype=torch.float32),
        )


def build_train_sampler(train_inv: pd.DataFrame) -> WeightedRandomSampler:
    category_weights = {
        "no_storm": NO_STORM_WEIGHT,
        "low_storm": LOW_STORM_WEIGHT,
        "medium_storm": MEDIUM_STORM_WEIGHT,
        "high_storm": HIGH_STORM_WEIGHT,
    }

    weights = train_inv["category"].map(category_weights).fillna(1.0).astype(np.float64).values
    weights = torch.as_tensor(weights, dtype=torch.double)

    return WeightedRandomSampler(
        weights=weights,
        num_samples=SAMPLES_PER_EPOCH,
        replacement=True,
    )


# ============================================================
# 7. LOSS AND METRICS
# ============================================================

class MaskedBCETverskyLoss(nn.Module):
    def __init__(
        self,
        pos_weight: float = POS_WEIGHT,
        bce_weight: float = LOSS_BCE_WEIGHT,
        tversky_weight: float = LOSS_TVERSKY_WEIGHT,
        alpha: float = TVERSKY_ALPHA,
        beta: float = TVERSKY_BETA,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.register_buffer("pos_weight_tensor", torch.tensor([pos_weight], dtype=torch.float32))
        self.bce_weight = bce_weight
        self.tversky_weight = tversky_weight
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits, targets, valid):
        valid = valid.float()
        targets = targets.float()

        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=self.pos_weight_tensor.to(logits.device),
        )
        bce = (bce * valid).sum() / (valid.sum() + self.eps)

        probs = torch.sigmoid(logits)
        probs = probs * valid
        targets = targets * valid

        tp = (probs * targets).sum()
        fp = (probs * (1.0 - targets) * valid).sum()
        fn = ((1.0 - probs) * targets).sum()

        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        tversky_loss = 1.0 - tversky

        return self.bce_weight * bce + self.tversky_weight * tversky_loss


@torch.no_grad()
def compute_metrics_from_counts(tp, fp, fn, tn) -> Dict[str, float]:
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


@torch.no_grad()
def threshold_sweep(model, loader, thresholds: List[float]) -> pd.DataFrame:
    model.eval()

    counts = {
        float(t): {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for t in thresholds
    }

    for images, targets, valid in tqdm(loader, desc="Threshold sweep", leave=False):
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        valid = valid.to(DEVICE, non_blocking=True)

        logits = model(images)
        probs = torch.sigmoid(logits)
        valid_bool = valid > 0.5

        for t in thresholds:
            t = float(t)
            pred = probs >= t

            counts[t]["tp"] += int(((pred == 1) & (targets == 1) & valid_bool).sum().item())
            counts[t]["fp"] += int(((pred == 1) & (targets == 0) & valid_bool).sum().item())
            counts[t]["fn"] += int(((pred == 0) & (targets == 1) & valid_bool).sum().item())
            counts[t]["tn"] += int(((pred == 0) & (targets == 0) & valid_bool).sum().item())

    rows = []
    for t, c in counts.items():
        rows.append({"threshold": t, **compute_metrics_from_counts(**c)})

    return pd.DataFrame(rows)


@torch.no_grad()
def evaluate_loss(model, loader, criterion) -> float:
    model.eval()
    losses = []

    for images, targets, valid in tqdm(loader, desc="Validation loss", leave=False):
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        valid = valid.to(DEVICE, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets, valid)
        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else float("nan")


def select_best_threshold(sweep_df: pd.DataFrame) -> float:
    row = sweep_df.sort_values(["iou", "f1", "precision"], ascending=[False, False, False]).iloc[0]
    return float(row["threshold"])


# ============================================================
# 8. FULL-IMAGE INFERENCE AND VISUALIZATION
# ============================================================

def get_tile_starts(length: int, tile: int, stride: int) -> List[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


@torch.no_grad()
def predict_full_image_tiled(
    model,
    rgb: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    tile_size: int = FULL_IMAGE_TILE_SIZE,
    stride: int = FULL_IMAGE_TILE_STRIDE,
) -> np.ndarray:
    h, w, _ = rgb.shape
    ys = get_tile_starts(h, tile_size, stride)
    xs = get_tile_starts(w, tile_size, stride)

    prob_sum = np.zeros((h, w), dtype=np.float32)
    prob_count = np.zeros((h, w), dtype=np.float32)

    model.eval()

    for y0 in ys:
        for x0 in xs:
            patch = rgb[y0:y0 + tile_size, x0:x0 + tile_size].astype(np.float32) / 255.0
            patch = (patch - mean) / (std + 1e-8)
            patch = np.transpose(patch, (2, 0, 1))

            x = torch.tensor(patch[None, ...], dtype=torch.float32, device=DEVICE)

            with get_autocast_context():
                prob = torch.sigmoid(model(x))[0, 0].detach().float().cpu().numpy()

            prob_sum[y0:y0 + tile_size, x0:x0 + tile_size] += prob
            prob_count[y0:y0 + tile_size, x0:x0 + tile_size] += 1.0

    return prob_sum / np.maximum(prob_count, 1e-8)


def counts_from_prob_and_mask(prob_map: np.ndarray, mask: np.ndarray, threshold: float) -> Dict[str, int]:
    valid = mask != 2
    pred = prob_map >= threshold
    gt = mask == 1

    return {
        "tp": int(np.logical_and(pred, gt & valid).sum()),
        "fp": int(np.logical_and(pred, (~gt) & valid).sum()),
        "fn": int(np.logical_and(~pred, gt & valid).sum()),
        "tn": int(np.logical_and(~pred, (~gt) & valid).sum()),
    }


def plot_confusion_matrix(metrics: Dict[str, Any], out_path: Path, title: str) -> None:
    cm = np.array([
        [int(metrics["tn"]), int(metrics["fp"])],
        [int(metrics["fn"]), int(metrics["tp"])],
    ])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="viridis")
    ax.set_title(
        f"{title}\nthreshold={metrics['threshold']:.2f} | IoU={metrics['iou']:.4f} | F1={metrics['f1']:.4f}",
        fontsize=13,
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
        columns=["Pred no storm", "Pred storm"],
    ).to_csv(out_path.with_suffix(".csv"))


def plot_patch_predictions(model, dataset, inventory, mean, std, threshold: float, out_path: Path, n: int = 5) -> None:
    inv = inventory.copy()
    positives = inv[inv["storm_ratio"] > 0].copy()
    if len(positives) == 0:
        selected = inv.sample(n=min(n, len(inv)), random_state=SEED)
    else:
        positives["score"] = (positives["storm_ratio"] - 0.25).abs()
        selected = positives.sort_values("score").head(n)

    fig, axes = plt.subplots(len(selected), 3, figsize=(13, 4.2 * len(selected)))
    if len(selected) == 1:
        axes = np.array([axes])

    cache = PairCache(max_items=4)
    model.eval()

    for i, (_, row) in enumerate(selected.iterrows()):
        rgb, mask = cache.load(str(row["image_path"]), str(row["mask_path"]))

        x0, y0, x1, y1 = map(int, [row["x0"], row["y0"], row["x1"], row["y1"]])
        rgb_patch = rgb[y0:y1, x0:x1]
        mask_patch = mask[y0:y1, x0:x1]

        arr = rgb_patch.astype(np.float32) / 255.0
        arr = (arr - mean) / (std + 1e-8)
        arr = np.transpose(arr, (2, 0, 1))
        x = torch.tensor(arr[None, ...], dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            with get_autocast_context():
                prob = torch.sigmoid(model(x))[0, 0].detach().float().cpu().numpy()

        pred = (prob >= threshold).astype(np.uint8)
        pred_display = pred.copy()
        pred_display[mask_patch == 2] = 2

        axes[i, 0].imshow(rgb_patch)
        axes[i, 0].set_title(f"RGB | {row['year']} | {row['day_key']}\nstorm={row['storm_ratio']*100:.1f}%")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask_patch, cmap=CLASS_CMAP, vmin=0, vmax=2)
        axes[i, 1].set_title("Ground truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred_display, cmap=CLASS_CMAP, vmin=0, vmax=2)
        axes[i, 2].set_title(f"Prediction | threshold={threshold:.2f}")
        axes[i, 2].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def full_image_evaluation(model, split_df: pd.DataFrame, mean, std, threshold: float, split_name: str) -> Dict[str, Any]:
    image_df = split_df[split_df["split"] == split_name].copy().reset_index(drop=True)

    counts_total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    per_image_rows = []

    prob_dir = OUTPUT_DIR / f"full_image_prob_maps_{split_name}"
    prob_dir.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(image_df.iterrows(), total=len(image_df), desc=f"Full-image evaluation [{split_name}]"):
        rgb = read_rgb(Path(row["image_path"]))
        mask = read_mask(Path(row["mask_path"]))
        mask = resize_mask_to_rgb(mask, rgb)

        prob_path = prob_dir / f"{safe_filename(row['year'])}_{safe_filename(row['day_key'])}_prob.npy"

        if REUSE_CACHE and prob_path.exists() and not FORCE_REBUILD_CACHE:
            prob = np.load(prob_path)
        else:
            prob = predict_full_image_tiled(model, rgb, mean, std)
            np.save(prob_path, prob)

        counts = counts_from_prob_and_mask(prob, mask, threshold)
        metrics = compute_metrics_from_counts(**counts)

        per_image_rows.append({
            "year": row["year"],
            "day_key": row["day_key"],
            "storm_ratio_valid": float(np.mean(mask[mask != 2] == 1)) if np.any(mask != 2) else 0.0,
            **metrics,
        })

        for k in counts_total:
            counts_total[k] += counts[k]

    full_metrics = compute_metrics_from_counts(**counts_total)
    full_metrics["threshold"] = threshold

    per_image_df = pd.DataFrame(per_image_rows)
    per_image_df.to_csv(OUTPUT_DIR / f"full_image_{split_name}_metrics_per_image.csv", index=False)
    pd.DataFrame([full_metrics]).to_csv(OUTPUT_DIR / f"full_image_{split_name}_metrics.csv", index=False)

    plot_confusion_matrix(
        full_metrics,
        OUTPUT_DIR / f"full_image_{split_name}_confusion_matrix.png",
        title=f"Full-image {split_name} evaluation",
    )

    return full_metrics


def plot_full_image_visualizations(model, split_df, mean, std, threshold: float, out_path: Path, split_name: str = "test") -> None:
    image_df = split_df[split_df["split"] == split_name].copy().reset_index(drop=True)

    # Prefer images with storm around 10-30%, but fall back to top storm images.
    rows = []
    for _, row in image_df.iterrows():
        mask = read_mask(Path(row["mask_path"]))
        rgb = read_rgb(Path(row["image_path"]))
        mask = resize_mask_to_rgb(mask, rgb)
        valid = mask != 2
        storm_ratio = float(np.mean(mask[valid] == 1)) if np.any(valid) else 0.0
        rows.append({**row.to_dict(), "storm_ratio_valid": storm_ratio})

    cand = pd.DataFrame(rows)
    mid = cand[(cand["storm_ratio_valid"] >= 0.08) & (cand["storm_ratio_valid"] <= 0.35)].copy()
    if len(mid) > 0:
        mid["score"] = (mid["storm_ratio_valid"] - 0.18).abs()
        selected = mid.sort_values("score").head(N_TEST_VISUALIZATIONS)
    else:
        selected = cand.sort_values("storm_ratio_valid", ascending=False).head(N_TEST_VISUALIZATIONS)

    n = len(selected)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 3, figsize=(15, 4.5 * n))
    if n == 1:
        axes = np.array([axes])

    for i, (_, row) in enumerate(selected.iterrows()):
        rgb = read_rgb(Path(row["image_path"]))
        mask = read_mask(Path(row["mask_path"]))
        mask = resize_mask_to_rgb(mask, rgb)

        prob = predict_full_image_tiled(model, rgb, mean, std)
        pred = (prob >= threshold).astype(np.uint8)
        pred_display = pred.copy()
        pred_display[mask == 2] = 2

        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"RGB | {row['year']} | {row['day_key']}\nstorm={row['storm_ratio_valid']*100:.1f}%")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask, cmap=CLASS_CMAP, vmin=0, vmax=2)
        axes[i, 1].set_title("Ground truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred_display, cmap=CLASS_CMAP, vmin=0, vmax=2)
        axes[i, 2].set_title(f"Prediction | threshold={threshold:.2f}")
        axes[i, 2].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 9. MODEL
# ============================================================

def build_model():
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=IN_CHANNELS,
        classes=CLASSES,
        activation=None,
    )
    return model


# ============================================================
# 10. TRAINING LOOP
# ============================================================

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    train_inv,
    val_inv,
    test_inv,
    split_df,
    mean,
    std,
):
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda" and USE_AMP))

    history = []
    best_val_iou = -1.0
    best_threshold = 0.5
    epochs_without_improvement = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_losses = []
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} - training")
        for step, (images, targets, valid) in enumerate(pbar, start=1):
            images = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            valid = valid.to(DEVICE, non_blocking=True)

            with get_autocast_context():
                logits = model(images)
                loss = criterion(logits, targets, valid)
                loss_scaled = loss / ACCUM_STEPS

            scaler.scale(loss_scaled).backward()

            if step % ACCUM_STEPS == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_losses.append(float(loss.item()))
            pbar.set_postfix(loss=np.mean(train_losses))

        train_loss = float(np.mean(train_losses))
        val_loss = evaluate_loss(model, val_loader, criterion)

        val_sweep = threshold_sweep(model, val_loader, THRESHOLDS)
        val_sweep.to_csv(OUTPUT_DIR / f"validation_threshold_sweep_epoch_{epoch:03d}.csv", index=False)

        best_row = val_sweep.sort_values(["iou", "f1", "precision"], ascending=[False, False, False]).iloc[0]
        val_threshold = float(best_row["threshold"])
        val_iou = float(best_row["iou"])
        val_f1 = float(best_row["f1"])
        val_precision = float(best_row["precision"])
        val_recall = float(best_row["recall"])

        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_threshold": val_threshold,
            "val_accuracy": float(best_row["accuracy"]),
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1,
            "val_iou": val_iou,
            "lr": current_lr,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(HISTORY_CSV, index=False)

        log(
            f"Epoch [{epoch:02d}/{NUM_EPOCHS}] | "
            f"T.Loss={train_loss:.4f} | V.Loss={val_loss:.4f} | "
            f"Thr={val_threshold:.2f} | IoU={val_iou:.4f} | F1={val_f1:.4f} | "
            f"Prec={val_precision:.4f} | Rec={val_recall:.4f} | LR={current_lr:.2e}"
        )

        scheduler.step(val_iou)

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_threshold = val_threshold
            epochs_without_improvement = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "architecture": ARCHITECTURE,
                "encoder_name": ENCODER_NAME,
                "encoder_weights": ENCODER_WEIGHTS,
                "mean": mean.tolist(),
                "std": std.tolist(),
                "best_threshold": best_threshold,
                "best_val_iou": best_val_iou,
                "epoch": epoch,
                "config": {
                    "patch_size": PATCH_SIZE,
                    "stride": STRIDE,
                    "min_valid_ratio": MIN_VALID_RATIO,
                    "batch_size": BATCH_SIZE,
                    "accum_steps": ACCUM_STEPS,
                    "samples_per_epoch": SAMPLES_PER_EPOCH,
                    "loss": "masked_bce_plus_tversky",
                    "pos_weight": POS_WEIGHT,
                    "tversky_alpha": TVERSKY_ALPHA,
                    "tversky_beta": TVERSKY_BETA,
                },
            }
            torch.save(checkpoint, BEST_MODEL_PATH)
            log(f"-> New best model saved: Val IoU={best_val_iou:.4f}, threshold={best_threshold:.2f}")
        else:
            epochs_without_improvement += 1
            log(f"-> No improvement. Patience: {epochs_without_improvement}/{EARLY_STOP_PATIENCE}")

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            log("Early stopping activated.")
            break

    return best_threshold, best_val_iou


# ============================================================
# 11. MAIN
# ============================================================

def main():
    log(f"Project root : {ROOT_DIR}")
    log(f"Output dir   : {OUTPUT_DIR}")
    log(f"Device       : {DEVICE}")
    if DEVICE.type == "cuda":
        log(f"GPU          : {torch.cuda.get_device_name(0)}")
    log(f"Model        : {ARCHITECTURE} | encoder={ENCODER_NAME} | weights={ENCODER_WEIGHTS}")

    config = {
        "YEARS": YEARS,
        "RUN_NAME": RUN_NAME,
        "PATCH_SIZE": PATCH_SIZE,
        "STRIDE": STRIDE,
        "MIN_VALID_RATIO": MIN_VALID_RATIO,
        "ARCHITECTURE": ARCHITECTURE,
        "ENCODER_NAME": ENCODER_NAME,
        "ENCODER_WEIGHTS": ENCODER_WEIGHTS,
        "BATCH_SIZE": BATCH_SIZE,
        "ACCUM_STEPS": ACCUM_STEPS,
        "NUM_EPOCHS": NUM_EPOCHS,
        "LEARNING_RATE": LEARNING_RATE,
        "SAMPLES_PER_EPOCH": SAMPLES_PER_EPOCH,
        "POS_WEIGHT": POS_WEIGHT,
        "LOSS_BCE_WEIGHT": LOSS_BCE_WEIGHT,
        "LOSS_TVERSKY_WEIGHT": LOSS_TVERSKY_WEIGHT,
        "TVERSKY_ALPHA": TVERSKY_ALPHA,
        "TVERSKY_BETA": TVERSKY_BETA,
        "THRESHOLDS": THRESHOLDS,
    }
    save_json(OUTPUT_DIR / "run_config.json", config)

    # 1. Data pairing and split.
    pair_df = collect_all_pairs()
    split_df = split_pairs_by_year(pair_df)
    split_df.to_csv(OUTPUT_DIR / "image_split_manifest.csv", index=False)

    # 2. Patch inventories.
    train_inv, val_inv, test_inv = load_or_build_patch_inventories(split_df)

    log("\nPatch inventory summary:")
    for name, inv in [("train", train_inv), ("val", val_inv), ("test", test_inv)]:
        log(f"\n{name.upper()} patches: {len(inv)}")
        log(inv["category"].value_counts().to_string())
        log(f"Positive patch fraction: {inv['positive_patch'].mean():.4f}")
        log(f"Storm pixel fraction over valid pixels: {inv['storm_ratio'].mean():.4f}")

    # 3. Normalization from training images.
    mean, std = compute_train_mean_std(split_df)
    log(f"\nTrain mean: {mean}")
    log(f"Train std : {std}")

    # 4. Datasets and DataLoaders.
    train_ds = StormPatchDataset(train_inv, mean, std, augment=True, cache_items=8)
    val_ds = StormPatchDataset(val_inv, mean, std, augment=False, cache_items=8)
    test_ds = StormPatchDataset(test_inv, mean, std, augment=False, cache_items=8)

    train_sampler = build_train_sampler(train_inv)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    # 5. Model / loss / optimizer.
    model = build_model().to(DEVICE)
    criterion = MaskedBCETverskyLoss().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
    )

    # 6. Train.
    best_threshold, best_val_iou = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_inv=train_inv,
        val_inv=val_inv,
        test_inv=test_inv,
        split_df=split_df,
        mean=mean,
        std=std,
    )

    # 7. Load best model and evaluate.
    log("\nLoading best checkpoint for final evaluation...")
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = build_model().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    best_threshold = float(checkpoint.get("best_threshold", best_threshold))
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)

    # Patch-level test.
    test_sweep = threshold_sweep(model, test_loader, THRESHOLDS)
    test_sweep.to_csv(OUTPUT_DIR / "test_patch_threshold_sweep.csv", index=False)

    final_idx = (test_sweep["threshold"] - best_threshold).abs().idxmin()
    patch_metrics = test_sweep.loc[final_idx].to_dict()
    patch_metrics["threshold"] = float(patch_metrics["threshold"])
    pd.DataFrame([patch_metrics]).to_csv(OUTPUT_DIR / "test_patch_metrics.csv", index=False)

    plot_confusion_matrix(
        patch_metrics,
        OUTPUT_DIR / "test_patch_confusion_matrix.png",
        title="Patch-level test",
    )

    plot_patch_predictions(
        model=model,
        dataset=test_ds,
        inventory=test_inv,
        mean=mean,
        std=std,
        threshold=best_threshold,
        out_path=OUTPUT_DIR / "test_patch_predictions.png",
        n=5,
    )

    # Full-image validation and test.
    log("\nRunning full-image validation/test evaluation...")
    full_val_metrics = full_image_evaluation(model, split_df, mean, std, best_threshold, split_name="val")
    full_test_metrics = full_image_evaluation(model, split_df, mean, std, best_threshold, split_name="test")

    plot_full_image_visualizations(
        model,
        split_df,
        mean,
        std,
        threshold=best_threshold,
        out_path=OUTPUT_DIR / "full_image_test_predictions.png",
        split_name="test",
    )

    # Final summary.
    summary = {
        "best_model_path": str(BEST_MODEL_PATH),
        "best_val_iou_patch": float(best_val_iou),
        "validation_selected_threshold": float(best_threshold),
        "patch_test_metrics": patch_metrics,
        "full_image_val_metrics": full_val_metrics,
        "full_image_test_metrics": full_test_metrics,
    }
    save_json(OUTPUT_DIR / "final_summary.json", summary)

    log("\n================ FINAL RESULTS ================")
    log(f"Best validation IoU        : {best_val_iou:.4f}")
    log(f"Validation-selected threshold: {best_threshold:.2f}")
    log("\nPatch-level test:")
    log(pd.DataFrame([patch_metrics]).to_string(index=False))
    log("\nFull-image validation:")
    log(pd.DataFrame([full_val_metrics]).to_string(index=False))
    log("\nFull-image test:")
    log(pd.DataFrame([full_test_metrics]).to_string(index=False))
    log(f"\nAll outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
