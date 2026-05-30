# ============================================================
# ATTENTION U-NET MULTI-YEAR TRAINING - OPTIMIZED V2
# Martian Dust Storm Segmentation using MDGM/MDAD masks
#
# This version keeps the original Attention U-Net baseline, but adapts
# it to MY28 + MY31 + MY32 and adds:
#
# - pos_weight = 8.0
# - samples_per_epoch = 10000
# - batch_size = 8
# - num_epochs = 35
# - validation threshold sweep
# - best model selected by best validation IoU across thresholds
# - final test evaluated with the best validation threshold
# - DataLoader optimizations
# - small LRU cache for image/mask loading
# - separate output folder per run to avoid overwriting old results
#
# Expected clean folder structure:
#
# ProjectFolder/
# ├── Attention_UNet_MultiYear_Optimized_V2.py
# ├── MY28/
# │   ├── selected_top100/images/
# │   ├── selected_top100/masks/
# │   └── reports/selected_top100.csv       optional but recommended
# ├── MY31/
# │   ├── selected_top100/images/
# │   ├── selected_top100/masks/
# │   └── reports/selected_top100.csv
# └── MY32/
#     ├── selected_top100/images/
#     ├── selected_top100/masks/
#     └── reports/selected_top100.csv
#
# Mask convention:
# 0 = no storm
# 1 = storm
# 2 = no-data / invalid pixel
# ============================================================

import random
import shutil
from pathlib import Path
from datetime import datetime
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# ============================================================
# 1. General configuration
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT_DIR = Path(__file__).resolve().parent

YEARS = ["MY28", "MY31", "MY32"]

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]

# Split by full images first. Patches are generated only after the split.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# If selected_top100.csv is sorted by storm percentage, sequential split can bias the sets.
SPLIT_MODE = "shuffle"   # options: "shuffle", "sequential"

RUN_NAME = "attention_unet_multiyear_v2_pw8_s10000"

# If False, a timestamp is appended when the output folder already exists.
# This prevents overwriting previous runs.
OVERWRITE_OUTPUTS = False


def make_output_dir(root_dir, run_name, overwrite=False):
    base_dir = root_dir / f"{run_name}_outputs"

    if overwrite:
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir

    if not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root_dir / f"{run_name}_{timestamp}_outputs"
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


OUTPUT_DIR = make_output_dir(ROOT_DIR, RUN_NAME, overwrite=OVERWRITE_OUTPUTS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


# ============================================================
# 2. Training parameters
# ============================================================

patch_size = 256
stride = 256
min_valid_ratio = 0.70

batch_size = 8
num_epochs = 35
learning_rate = 1e-4

base_channels = 16
pos_weight = 8.0

# The model samples this number of training patches per epoch.
# With batch_size=8, this gives about 1250 training iterations per epoch.
samples_per_epoch = 10000

num_workers = 2

# Cache per DataLoader worker. Increase slightly if RAM allows.
train_cache_items = 8
eval_cache_items = 8

# Threshold sweep. The best threshold is selected using validation IoU.
threshold_values = np.round(np.arange(0.10, 0.91, 0.05), 2).tolist()

best_model_path = OUTPUT_DIR / "best_attention_unet_multiyear_v2.pth"


# ============================================================
# 3. Utility functions
# ============================================================

def normalize_key(text):
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


def find_existing_dir(candidates):
    for path in candidates:
        if path.exists() and path.is_dir():
            return path
    return None


def read_rgb(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Image not found or unreadable: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found or unreadable: {path}")
    return mask


def resolve_path(path_text):
    path = Path(str(path_text))
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def get_autocast_context():
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=(device.type == "cuda")
    )


def make_grad_scaler():
    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=(device.type == "cuda")
        )
    except Exception:
        return torch.cuda.amp.GradScaler(
            enabled=(device.type == "cuda")
        )


def make_dataloader(dataset, batch_size, shuffle=False, sampler=None, drop_last=False):
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=drop_last
    )

    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    return DataLoader(**kwargs)


# ============================================================
# 4. Multi-year image/mask discovery
# ============================================================

def collect_pairs_from_report(year):
    """
    Preferred method when reports/selected_top100.csv exists.
    The pairing script normally writes image_path and mask_png_path relative to ROOT_DIR.
    """

    report_path = ROOT_DIR / year / "reports" / "selected_top100.csv"

    if not report_path.exists():
        return None

    df = pd.read_csv(report_path)

    if "image_path" not in df.columns or "mask_png_path" not in df.columns:
        return None

    rows = []

    for i, row in df.iterrows():
        img_path = resolve_path(row["image_path"])
        mask_path = resolve_path(row["mask_png_path"])

        if not img_path.exists() or not mask_path.exists():
            continue

        day_key = str(row.get("day_key", img_path.stem))

        rows.append({
            "year": year,
            "image_id": f"{year}_{day_key}",
            "day_key": day_key,
            "image_path": str(img_path),
            "mask_path": str(mask_path),
            "source_index": int(row.get("rank", i)) if str(row.get("rank", i)).isdigit() else i
        })

    if len(rows) == 0:
        return None

    return pd.DataFrame(rows)


def collect_pairs_from_folders(year):
    """
    Fallback when selected_top100.csv is not available.
    It tries to match images and masks by normalized stem.
    """

    year_dir = ROOT_DIR / year

    image_dir = find_existing_dir([
        year_dir / "selected_top100" / "images",
        year_dir / "images",
        year_dir / "mdgm_rgb"
    ])

    mask_dir = find_existing_dir([
        year_dir / "selected_top100" / "masks",
        year_dir / "masks",
        year_dir / "masks_multiclass",
        year_dir / "masks_png_all"
    ])

    if image_dir is None or mask_dir is None:
        raise FileNotFoundError(
            f"Could not find image/mask folders for {year}.\n"
            f"Expected for example:\n"
            f"{year}/selected_top100/images\n"
            f"{year}/selected_top100/masks"
        )

    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(sorted(image_dir.glob(f"*{ext}")))

    masks = []
    for ext in IMAGE_EXTENSIONS:
        masks.extend(sorted(mask_dir.glob(f"*{ext}")))

    if len(images) == 0 or len(masks) == 0:
        raise RuntimeError(f"No images or masks found for {year}.")

    mask_dict = {}
    for mask_path in masks:
        key = normalize_key(mask_path.stem)
        mask_dict[key] = mask_path

    rows = []

    for i, img_path in enumerate(images):
        img_key = normalize_key(img_path.stem)
        mask_path = None

        if img_key in mask_dict:
            mask_path = mask_dict[img_key]
        else:
            for mk, mp in mask_dict.items():
                if img_key in mk or mk in img_key:
                    mask_path = mp
                    break

        if mask_path is None:
            print(f"[WARN] No mask found for image: {img_path.name}")
            continue

        rows.append({
            "year": year,
            "image_id": f"{year}_{img_path.stem}",
            "day_key": img_path.stem,
            "image_path": str(img_path),
            "mask_path": str(mask_path),
            "source_index": i
        })

    if len(rows) == 0:
        raise RuntimeError(f"No valid image-mask pairs could be matched for {year}.")

    return pd.DataFrame(rows)


def collect_all_image_pairs():
    all_rows = []

    for year in YEARS:
        print(f"\nCollecting image-mask pairs for {year}...")

        df = collect_pairs_from_report(year)

        if df is None:
            print(f"[INFO] Report not available or incomplete for {year}. Using folder matching.")
            df = collect_pairs_from_folders(year)
        else:
            print(f"[INFO] Using report file for {year}.")

        print(f"{year}: {len(df)} image-mask pairs found.")
        all_rows.append(df)

    pair_df = pd.concat(all_rows, ignore_index=True)

    if len(pair_df) == 0:
        raise RuntimeError("No image-mask pairs were found.")

    pair_df.to_csv(OUTPUT_DIR / "all_image_pairs.csv", index=False)

    return pair_df


def split_pairs_by_year(pair_df):
    """
    Applies 70/15/15 split independently to each Mars year.
    """

    split_rows = []

    for year in YEARS:
        sub = pair_df[pair_df["year"] == year].copy().reset_index(drop=True)

        if SPLIT_MODE == "shuffle":
            sub = sub.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        elif SPLIT_MODE == "sequential":
            sub = sub.sort_values("source_index").reset_index(drop=True)
        else:
            raise ValueError("SPLIT_MODE must be 'shuffle' or 'sequential'.")

        n = len(sub)
        n_train = int(round(TRAIN_FRAC * n))
        n_val = int(round(VAL_FRAC * n))

        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)

        sub["split"] = "test"
        sub.loc[:n_train - 1, "split"] = "train"
        sub.loc[n_train:n_train + n_val - 1, "split"] = "val"
        sub.loc[n_train + n_val:, "split"] = "test"

        print(
            f"{year} split -> "
            f"train={int((sub['split'] == 'train').sum())}, "
            f"val={int((sub['split'] == 'val').sum())}, "
            f"test={int((sub['split'] == 'test').sum())}"
        )

        split_rows.append(sub)

    split_df = pd.concat(split_rows, ignore_index=True)
    split_df.to_csv(OUTPUT_DIR / "image_split_manifest.csv", index=False)

    return split_df


# ============================================================
# 5. Image and mask loading
# ============================================================

def load_image_and_mask_from_paths(image_path, mask_path):
    img = read_rgb(Path(image_path))
    mask = read_mask(Path(mask_path))

    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(
            mask,
            (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    return img, mask


# ============================================================
# 6. Patch inventory construction
# ============================================================

def patch_category(storm_ratio):
    if storm_ratio == 0:
        return "no_storm"
    elif storm_ratio < 0.05:
        return "low_storm"
    elif storm_ratio < 0.20:
        return "medium_storm"
    else:
        return "high_storm"


def build_patch_inventory(split_df, split_name):
    rows = []
    split_images = split_df[split_df["split"] == split_name].reset_index(drop=True)

    for _, item in tqdm(
        split_images.iterrows(),
        total=len(split_images),
        desc=f"Building {split_name} patches"
    ):
        _, mask = load_image_and_mask_from_paths(
            item["image_path"],
            item["mask_path"]
        )

        h, w = mask.shape

        for y0 in range(0, h - patch_size + 1, stride):
            for x0 in range(0, w - patch_size + 1, stride):

                y1 = y0 + patch_size
                x1 = x0 + patch_size

                mask_patch = mask[y0:y1, x0:x1]

                valid = mask_patch != 2
                valid_ratio = valid.mean()

                if valid_ratio < min_valid_ratio:
                    continue

                storm_ratio = np.mean(mask_patch[valid] == 1)

                rows.append({
                    "year": item["year"],
                    "image_id": item["image_id"],
                    "day_key": item["day_key"],
                    "image_path": item["image_path"],
                    "mask_path": item["mask_path"],
                    "split": split_name,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "valid_ratio": valid_ratio,
                    "storm_ratio": storm_ratio,
                    "positive_patch": storm_ratio > 0,
                    "category": patch_category(storm_ratio)
                })

    inventory = pd.DataFrame(rows)

    if len(inventory) == 0:
        raise RuntimeError(f"No valid patches were found for split: {split_name}")

    inventory.to_csv(OUTPUT_DIR / f"patch_inventory_{split_name}.csv", index=False)

    return inventory


def print_patch_summary(name, inventory):
    print(f"\n{name} patches: {len(inventory)}")
    print("Patch categories:")
    print(inventory["category"].value_counts())

    positive_patch_fraction = float(inventory["positive_patch"].mean())
    weighted_storm_pixel_fraction = float(
        (inventory["storm_ratio"] * inventory["valid_ratio"]).sum()
        / (inventory["valid_ratio"].sum() + 1e-8)
    )

    print(f"Positive patch fraction: {positive_patch_fraction:.4f}")
    print(f"Approx. storm pixel fraction over valid pixels: {weighted_storm_pixel_fraction:.4f}")


# ============================================================
# 7. PyTorch Dataset with LRU cache
# ============================================================

class MarsStormDataset(Dataset):
    def __init__(self, patch_inventory, augment=False, max_cache_items=8):
        self.patch_inventory = patch_inventory.reset_index(drop=True)
        self.augment = augment
        self.max_cache_items = int(max_cache_items)
        self.cache = OrderedDict()

        # Same approximate normalization used in the original baseline.
        self.mean = np.array([0.55, 0.35, 0.25], dtype=np.float32)
        self.std = np.array([0.20, 0.15, 0.15], dtype=np.float32)

    def __len__(self):
        return len(self.patch_inventory)

    def _load_cached(self, image_path, mask_path):
        key = (str(image_path), str(mask_path))

        if key in self.cache:
            img, mask = self.cache.pop(key)
            self.cache[key] = (img, mask)
            return img, mask

        img, mask = load_image_and_mask_from_paths(image_path, mask_path)

        self.cache[key] = (img, mask)

        if self.max_cache_items > 0:
            while len(self.cache) > self.max_cache_items:
                self.cache.popitem(last=False)
        else:
            self.cache.clear()

        return img, mask

    def __getitem__(self, idx):
        row = self.patch_inventory.iloc[idx]

        x0 = int(row["x0"])
        y0 = int(row["y0"])
        x1 = int(row["x1"])
        y1 = int(row["y1"])

        img, mask = self._load_cached(
            row["image_path"],
            row["mask_path"]
        )

        img_patch = img[y0:y1, x0:x1].astype(np.float32) / 255.0
        img_patch = (img_patch - self.mean) / (self.std + 1e-8)

        mask_patch = mask[y0:y1, x0:x1]

        target = (mask_patch == 1).astype(np.float32)
        valid = (mask_patch != 2).astype(np.float32)

        if self.augment:
            img_patch, target, valid = self._augment(img_patch, target, valid)

        img_patch = np.transpose(img_patch, (2, 0, 1))
        target = target[None, :, :]
        valid = valid[None, :, :]

        return (
            torch.tensor(img_patch, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(valid, dtype=torch.float32)
        )

    def _augment(self, img, target, valid):
        if np.random.rand() < 0.5:
            img = np.flip(img, axis=1).copy()
            target = np.flip(target, axis=1).copy()
            valid = np.flip(valid, axis=1).copy()

        if np.random.rand() < 0.5:
            img = np.flip(img, axis=0).copy()
            target = np.flip(target, axis=0).copy()
            valid = np.flip(valid, axis=0).copy()

        k = np.random.randint(0, 4)

        img = np.rot90(img, k, axes=(0, 1)).copy()
        target = np.rot90(target, k, axes=(0, 1)).copy()
        valid = np.rot90(valid, k, axes=(0, 1)).copy()

        return img, target, valid


# ============================================================
# 8. Attention U-Net architecture
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    def __init__(self, gate_channels, skip_channels, intermediate_channels):
        super().__init__()

        self.gate_conv = nn.Sequential(
            nn.Conv2d(gate_channels, intermediate_channels, kernel_size=1),
            nn.BatchNorm2d(intermediate_channels)
        )

        self.skip_conv = nn.Sequential(
            nn.Conv2d(skip_channels, intermediate_channels, kernel_size=1),
            nn.BatchNorm2d(intermediate_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(intermediate_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        attention = self.relu(self.gate_conv(gate) + self.skip_conv(skip))
        attention = self.psi(attention)

        return skip * attention


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=16):
        super().__init__()

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
        self.att4 = AttentionGate(f4, f4, f3)
        self.dec4 = ConvBlock(f4 + f4, f4)

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(f4, f3, kernel_size=3, padding=1)
        )
        self.att3 = AttentionGate(f3, f3, f2)
        self.dec3 = ConvBlock(f3 + f3, f3)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(f3, f2, kernel_size=3, padding=1)
        )
        self.att2 = AttentionGate(f2, f2, f1)
        self.dec2 = ConvBlock(f2 + f2, f2)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(f2, f1, kernel_size=3, padding=1)
        )
        self.att1 = AttentionGate(f1, f1, max(f1 // 2, 1))
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


# ============================================================
# 9. Loss function: BCE + Dice
# ============================================================

class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=8.0, bce_weight=1.0, dice_weight=1.0):
        super().__init__()

        self.register_buffer(
            "pos_weight",
            torch.tensor([pos_weight], dtype=torch.float32)
        )

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets, valid):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none"
        )

        bce = (bce * valid).sum() / (valid.sum() + 1e-8)

        probs = torch.sigmoid(logits) * valid
        targets = targets * valid

        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()

        smooth = 1.0
        dice = 1.0 - (2.0 * intersection + smooth) / (union + smooth)

        return self.bce_weight * bce + self.dice_weight * dice


# ============================================================
# 10. Metrics and threshold sweep
# ============================================================

def counts_to_metrics(tp, fp, fn, tn=0):
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8) if (tp + fp + fn + tn) > 0 else 0.0

    return precision, recall, f1, iou, accuracy


def evaluate_threshold_sweep(model, loader, criterion, thresholds):
    model.eval()

    thresholds = [float(t) for t in thresholds]
    stats = {
        t: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for t in thresholds
    }

    total_loss = 0.0

    with torch.no_grad():
        for images, targets, valid in tqdm(loader, desc="Evaluating threshold sweep"):

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)

            with get_autocast_context():
                logits = model(images)
                loss = criterion(logits, targets, valid)

            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            valid_bool = valid == 1
            target_bool = (targets == 1) & valid_bool

            for t in thresholds:
                pred_bool = (probs >= t) & valid_bool

                tp = (pred_bool & target_bool).sum().item()
                fp = (pred_bool & (~target_bool) & valid_bool).sum().item()
                fn = ((~pred_bool) & target_bool).sum().item()
                tn = ((~pred_bool) & (~target_bool) & valid_bool).sum().item()

                stats[t]["tp"] += tp
                stats[t]["fp"] += fp
                stats[t]["fn"] += fn
                stats[t]["tn"] += tn

    avg_loss = total_loss / len(loader)

    rows = []

    for t in thresholds:
        tp = stats[t]["tp"]
        fp = stats[t]["fp"]
        fn = stats[t]["fn"]
        tn = stats[t]["tn"]

        precision, recall, f1, iou, accuracy = counts_to_metrics(tp, fp, fn, tn)

        rows.append({
            "threshold": t,
            "loss": avg_loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "accuracy": accuracy
        })

    result_df = pd.DataFrame(rows)

    best_row = (
        result_df
        .sort_values(["iou", "f1", "precision"], ascending=False)
        .iloc[0]
        .to_dict()
    )

    return result_df, best_row


# ============================================================
# 11. Visualization
# ============================================================

def predict_patch(model, img_patch, mean, std, threshold):
    img_norm = img_patch.astype(np.float32) / 255.0
    img_norm = (img_norm - mean) / (std + 1e-8)
    img_norm = np.transpose(img_norm, (2, 0, 1))

    img_tensor = torch.tensor(
        img_norm[None, :, :, :],
        dtype=torch.float32
    ).to(device)

    model.eval()

    with torch.no_grad():
        with get_autocast_context():
            logits = model(img_tensor)
            prob = torch.sigmoid(logits)

    prob = prob.squeeze().detach().cpu().numpy()
    pred = (prob >= threshold).astype(np.uint8)

    return prob, pred


def plot_test_patch_predictions(model, test_inventory, mean, std, threshold, n_examples=5):
    candidates = test_inventory[
        (test_inventory["storm_ratio"] >= 0.10) &
        (test_inventory["storm_ratio"] <= 0.50) &
        (test_inventory["valid_ratio"] >= 0.95)
    ].copy()

    if len(candidates) == 0:
        candidates = test_inventory[test_inventory["storm_ratio"] > 0].copy()

    if len(candidates) == 0:
        candidates = test_inventory.copy()

    candidates["selection_score"] = abs(candidates["storm_ratio"] - 0.30)
    selected = candidates.sort_values("selection_score").head(n_examples).reset_index(drop=True)

    fig, axes = plt.subplots(
        nrows=len(selected),
        ncols=4,
        figsize=(17, 4.2 * len(selected))
    )

    if len(selected) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, row in selected.iterrows():
        img, mask = load_image_and_mask_from_paths(
            row["image_path"],
            row["mask_path"]
        )

        x0 = int(row["x0"])
        y0 = int(row["y0"])
        x1 = int(row["x1"])
        y1 = int(row["y1"])

        img_patch = img[y0:y1, x0:x1]
        mask_patch = mask[y0:y1, x0:x1]

        target = (mask_patch == 1).astype(np.uint8)
        valid = (mask_patch != 2).astype(np.uint8)

        prob, pred = predict_patch(
            model=model,
            img_patch=img_patch,
            mean=mean,
            std=std,
            threshold=threshold
        )

        pred[valid == 0] = 0

        gt_display = target.copy()
        gt_display[valid == 0] = 2

        pred_display = pred.copy()
        pred_display[valid == 0] = 2

        axes[i, 0].imshow(img_patch)
        axes[i, 0].set_title(
            f"Original patch\n"
            f"{row['year']} | {row['day_key']}\n"
            f"storm={row['storm_ratio']:.1%}"
        )
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt_display, vmin=0, vmax=2)
        axes[i, 1].set_title("Ground truth")
        axes[i, 1].axis("off")

        im = axes[i, 2].imshow(prob, vmin=0, vmax=1)
        axes[i, 2].set_title("Probability map")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(pred_display, vmin=0, vmax=2)
        axes[i, 3].set_title(f"Prediction | threshold={threshold:.2f}")
        axes[i, 3].axis("off")

    fig.tight_layout()

    out_path = OUTPUT_DIR / f"test_5_patch_predictions_thr_{str(round(threshold, 2)).replace('.', 'p')}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Test visualization saved as: {out_path}")


def plot_confusion_matrix(metrics, threshold):
    cm = np.array([
        [metrics["tn"], metrics["fp"]],
        [metrics["fn"], metrics["tp"]]
    ], dtype=np.int64)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(cm)

    ax.set_title(
        f"Test Confusion Matrix | threshold={threshold:.2f}\n"
        f"IoU={metrics['iou']:.4f} | F1={metrics['f1']:.4f}"
    )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred no storm", "Pred storm"])
    ax.set_yticklabels(["True no storm", "True storm"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center")

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    out_path = OUTPUT_DIR / f"test_confusion_matrix_thr_{str(round(threshold, 2)).replace('.', 'p')}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(
        cm,
        index=["True no storm", "True storm"],
        columns=["Pred no storm", "Pred storm"]
    ).to_csv(OUTPUT_DIR / f"test_confusion_matrix_thr_{str(round(threshold, 2)).replace('.', 'p')}.csv")

    print(f"Confusion matrix saved as: {out_path}")


def plot_threshold_curve(df, split_name):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["threshold"], df["iou"], marker="o", label="IoU")
    ax.plot(df["threshold"], df["f1"], marker="o", label="F1")
    ax.plot(df["threshold"], df["precision"], marker="o", label="Precision")
    ax.plot(df["threshold"], df["recall"], marker="o", label="Recall")

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric value")
    ax.set_title(f"{split_name} threshold sweep")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out_path = OUTPUT_DIR / f"{split_name.lower()}_threshold_sweep.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 12. Main execution
# ============================================================

if __name__ == "__main__":

    print("============================================================")
    print("Attention U-Net Multi-Year Training - Optimized V2")
    print("============================================================")
    print(f"Root folder: {ROOT_DIR}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Years: {YEARS}")
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    try:
        shutil.copy2(__file__, OUTPUT_DIR / Path(__file__).name)
    except Exception:
        pass

    # Save run configuration
    run_config = {
        "seed": SEED,
        "years": ",".join(YEARS),
        "train_frac": TRAIN_FRAC,
        "val_frac": VAL_FRAC,
        "test_frac": TEST_FRAC,
        "split_mode": SPLIT_MODE,
        "patch_size": patch_size,
        "stride": stride,
        "min_valid_ratio": min_valid_ratio,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "base_channels": base_channels,
        "pos_weight": pos_weight,
        "samples_per_epoch": samples_per_epoch,
        "num_workers": num_workers,
        "threshold_values": ",".join([str(t) for t in threshold_values]),
        "train_cache_items": train_cache_items,
        "eval_cache_items": eval_cache_items
    }

    pd.DataFrame([run_config]).to_csv(OUTPUT_DIR / "run_config.csv", index=False)

    # --------------------------------------------------------
    # Image-level split
    # --------------------------------------------------------

    pair_df = collect_all_image_pairs()
    split_df = split_pairs_by_year(pair_df)

    # --------------------------------------------------------
    # Patch-level inventories
    # --------------------------------------------------------

    print("\nBuilding patch inventories...")

    train_patch_inventory = build_patch_inventory(split_df, "train")
    val_patch_inventory = build_patch_inventory(split_df, "val")
    test_patch_inventory = build_patch_inventory(split_df, "test")

    all_patches = pd.concat(
        [train_patch_inventory, val_patch_inventory, test_patch_inventory],
        ignore_index=True
    )
    all_patches.to_csv(OUTPUT_DIR / "patch_inventory_all.csv", index=False)

    print_patch_summary("Train", train_patch_inventory)
    print_patch_summary("Validation", val_patch_inventory)
    print_patch_summary("Test", test_patch_inventory)

    # --------------------------------------------------------
    # Datasets and loaders
    # --------------------------------------------------------

    train_dataset = MarsStormDataset(
        train_patch_inventory,
        augment=True,
        max_cache_items=train_cache_items
    )

    val_dataset = MarsStormDataset(
        val_patch_inventory,
        augment=False,
        max_cache_items=eval_cache_items
    )

    test_dataset = MarsStormDataset(
        test_patch_inventory,
        augment=False,
        max_cache_items=eval_cache_items
    )

    # Keep the original sampler logic for now.
    category_weights = {
        "no_storm": 0.5,
        "low_storm": 2.0,
        "medium_storm": 3.0,
        "high_storm": 4.0
    }

    sample_weights = (
        train_patch_inventory["category"]
        .map(category_weights)
        .fillna(1.0)
        .values
    )

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=samples_per_epoch,
        replacement=True
    )

    train_loader = make_dataloader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler
    )

    val_loader = make_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = make_dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # --------------------------------------------------------
    # Model, loss, optimizer
    # --------------------------------------------------------

    model = AttentionUNet(
        in_channels=3,
        out_channels=1,
        base_channels=base_channels
    ).to(device)

    criterion = BCEDiceLoss(
        pos_weight=pos_weight
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4
    )

    scaler = make_grad_scaler()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_iou = -1.0
    best_threshold = None
    history = []

    print("\nStarting training...")

    for epoch in range(num_epochs):

        model.train()
        train_loss = 0.0

        for images, targets, valid in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs} - Training"
        ):

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with get_autocast_context():
                logits = model(images)
                loss = criterion(logits, targets, valid)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation threshold sweep
        val_sweep_df, best_val_row = evaluate_threshold_sweep(
            model,
            val_loader,
            criterion,
            threshold_values
        )

        val_sweep_df.insert(0, "epoch", epoch + 1)
        val_sweep_df.to_csv(OUTPUT_DIR / f"val_threshold_sweep_epoch_{epoch + 1:03d}.csv", index=False)

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(best_val_row["iou"])
        new_lr = optimizer.param_groups[0]["lr"]

        history_row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": best_val_row["loss"],
            "best_val_threshold": best_val_row["threshold"],
            "val_precision": best_val_row["precision"],
            "val_recall": best_val_row["recall"],
            "val_f1": best_val_row["f1"],
            "val_iou": best_val_row["iou"],
            "val_accuracy": best_val_row["accuracy"],
            "val_tp": best_val_row["tp"],
            "val_fp": best_val_row["fp"],
            "val_fn": best_val_row["fn"],
            "val_tn": best_val_row["tn"],
            "learning_rate": new_lr
        }

        history.append(history_row)
        pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history.csv", index=False)

        print(
            f"Epoch [{epoch + 1:02d}/{num_epochs}] | "
            f"T.Loss: {train_loss:.4f} | "
            f"V.Loss: {best_val_row['loss']:.4f} | "
            f"BestThr: {best_val_row['threshold']:.2f} | "
            f"Prec: {best_val_row['precision']:.4f} | "
            f"Rec: {best_val_row['recall']:.4f} | "
            f"F1: {best_val_row['f1']:.4f} | "
            f"IoU: {best_val_row['iou']:.4f} | "
            f"LR: {new_lr:.2e}"
        )

        if new_lr < old_lr:
            print(f"-> Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e}")

        if best_val_row["iou"] > best_val_iou:
            best_val_iou = float(best_val_row["iou"])
            best_threshold = float(best_val_row["threshold"])

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "best_val_iou": best_val_iou,
                    "best_threshold": best_threshold,
                    "best_val_metrics": best_val_row,
                    "base_channels": base_channels,
                    "patch_size": patch_size,
                    "stride": stride,
                    "mean": train_dataset.mean.tolist(),
                    "std": train_dataset.std.tolist(),
                    "years": YEARS,
                    "pos_weight": pos_weight,
                    "samples_per_epoch": samples_per_epoch,
                    "threshold_values": threshold_values
                },
                best_model_path
            )

            print(
                f"-> New best model saved. "
                f"Val IoU: {best_val_iou:.4f} | "
                f"Threshold: {best_threshold:.2f}"
            )

        if (epoch + 1) % 5 == 0:
            hist_df = pd.DataFrame(history)
            plot_threshold_curve(val_sweep_df, f"Val_epoch_{epoch + 1:03d}")

    print("\nTraining completed.")
    print(f"Best validation IoU: {best_val_iou:.4f}")
    print(f"Best validation threshold: {best_threshold:.2f}")
    print(f"Best model saved at: {best_model_path}")

    # --------------------------------------------------------
    # Final test evaluation with best checkpoint
    # --------------------------------------------------------

    print("\nLoading best checkpoint for final test evaluation...")

    checkpoint = safe_torch_load(best_model_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    best_threshold = float(checkpoint["best_threshold"])

    test_sweep_df, best_test_row_from_sweep = evaluate_threshold_sweep(
        model,
        test_loader,
        criterion,
        threshold_values
    )

    test_sweep_df.to_csv(OUTPUT_DIR / "test_threshold_sweep.csv", index=False)
    plot_threshold_curve(test_sweep_df, "Test")

    # The formal test report uses the validation-selected threshold.
    test_at_val_threshold = test_sweep_df[
        np.isclose(test_sweep_df["threshold"], best_threshold)
    ]

    if len(test_at_val_threshold) == 0:
        # Fallback in case of floating-point formatting mismatch.
        idx = (test_sweep_df["threshold"] - best_threshold).abs().idxmin()
        test_metrics = test_sweep_df.loc[idx].to_dict()
    else:
        test_metrics = test_at_val_threshold.iloc[0].to_dict()

    pd.DataFrame([test_metrics]).to_csv(OUTPUT_DIR / "test_metrics_at_best_val_threshold.csv", index=False)
    pd.DataFrame([best_test_row_from_sweep]).to_csv(OUTPUT_DIR / "test_metrics_best_test_threshold_diagnostic.csv", index=False)

    print("\n========== FINAL TEST RESULTS ==========")
    print(f"Validation-selected threshold : {best_threshold:.2f}")
    print(f"Loss                          : {test_metrics['loss']:.4f}")
    print(f"Accuracy                      : {test_metrics['accuracy']:.4f}")
    print(f"Precision                     : {test_metrics['precision']:.4f}")
    print(f"Recall                        : {test_metrics['recall']:.4f}")
    print(f"F1-score                      : {test_metrics['f1']:.4f}")
    print(f"IoU                           : {test_metrics['iou']:.4f}")
    print(f"TN                            : {int(test_metrics['tn'])}")
    print(f"FP                            : {int(test_metrics['fp'])}")
    print(f"FN                            : {int(test_metrics['fn'])}")
    print(f"TP                            : {int(test_metrics['tp'])}")

    print("\nDiagnostic only: best threshold on test itself")
    print(
        f"Best test threshold: {best_test_row_from_sweep['threshold']:.2f} | "
        f"IoU={best_test_row_from_sweep['iou']:.4f} | "
        f"F1={best_test_row_from_sweep['f1']:.4f} | "
        f"Prec={best_test_row_from_sweep['precision']:.4f} | "
        f"Rec={best_test_row_from_sweep['recall']:.4f}"
    )

    plot_confusion_matrix(test_metrics, best_threshold)

    plot_test_patch_predictions(
        model=model,
        test_inventory=test_patch_inventory,
        mean=train_dataset.mean,
        std=train_dataset.std,
        threshold=best_threshold,
        n_examples=5
    )

    print("\nCompleted.")
    print(f"All outputs saved in: {OUTPUT_DIR}")
