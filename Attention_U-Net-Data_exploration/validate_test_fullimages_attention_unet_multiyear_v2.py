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
from matplotlib.colors import ListedColormap
from PIL import Image
from tqdm import tqdm


# ============================================================
# FULL-IMAGE VALIDATION + TEST FOR best_attention_unet_multiyear_v2.pth
# ------------------------------------------------------------
# What this script does:
# 1) Loads the already trained checkpoint best_attention_unet_multiyear_v2.pth
# 2) Scans MY28, MY31 and MY32 full images and masks
# 3) Selects 2 validation images per year and 1 test image per year
#    with storm percentage between 10% and 20% whenever possible
# 4) Excludes the user-provided MY32 images from both validation and test
# 5) Runs full-image tiled inference (no retraining)
# 6) Finds the best threshold using ONLY the 6 validation images
# 7) Evaluates the 3 test images using that validation-selected threshold
# 8) Saves metrics, confusion matrix, CSVs and a final 3x3 visualization
#
# Folder structure expected next to this script:
# - Modelos generados/best_attention_unet_multiyear_v2.pth
# - MY28/
# - MY31/
# - MY32/
#   where each year contains selected_top100/images and selected_top100/masks
#   or equivalent fallback folders.
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "Modelos generados"
OUTPUT_DIR = PROJECT_ROOT / "fullimage_attention_unet_multiyear_v2_eval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = ["MY28", "MY31", "MY32"]
TARGET_MIN_STORM = 0.10
TARGET_MAX_STORM = 0.20
TARGET_CENTER = 0.15

# Attention U-Net v2 settings (same family as your multi-year V2 model).
PATCH_SIZE = 256
TILE_SIZE = 256
TILE_STRIDE = 128   # overlap for smoother full-image inference
BASE_CHANNELS_DEFAULT = 16
MEAN_DEFAULT = np.array([0.55, 0.35, 0.25], dtype=np.float32)
STD_DEFAULT = np.array([0.20, 0.15, 0.15], dtype=np.float32)
THRESHOLDS = np.round(np.arange(0.10, 0.91, 0.05), 2).tolist()

CLASS_CMAP = ListedColormap(["#440154", "#2A9D9A", "#FDE725"])

EXCLUDED_MY32 = {
    "346_F04_day22", "341_F04_day17", "328_F04_day04", "353_F04_day29",
    "082_D17_day19", "340_F04_day16", "106_D18_day10", "090_D17_day27",
    "357_F04_day33", "030_D15_day32", "290_F02_day31", "314_F03_day22",
}


def log(msg: str):
    print(msg, flush=True)


def normalize_key(text: Any) -> str:
    return (
        str(text)
        .lower()
        .replace("_mask_class", "")
        .replace("_multiclass", "")
        .replace("_mask", "")
        .replace("_class", "")
        .replace("_", "")
        .replace("-", "")
        .replace(".", "")
        .replace(" ", "")
    )


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("L"), dtype=np.uint8)


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
        if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
    raise ValueError("No state_dict was found in the checkpoint.")


def infer_base_channels(checkpoint: Any, default: int = BASE_CHANNELS_DEFAULT) -> int:
    if isinstance(checkpoint, dict):
        for key in ["base_channels", "BASE_CHANNELS", "base_ch"]:
            if key in checkpoint:
                try:
                    return int(checkpoint[key])
                except Exception:
                    pass
    return int(default)


def infer_mean_std(checkpoint: Any):
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
    if mean is None:
        mean = MEAN_DEFAULT.copy()
    if std is None:
        std = STD_DEFAULT.copy()
    return mean, std


# ============================================================
# MODEL DEFINITION (Attention U-Net with BatchNorm)
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
            nn.ReLU(inplace=True),
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
# DATA DISCOVERY
# ============================================================

def find_first_existing(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def load_year_records(year: str) -> pd.DataFrame:
    year_dir = PROJECT_ROOT / year
    img_dir = find_first_existing([
        year_dir / "selected_top100" / "images",
        year_dir / "images",
        year_dir / "mdgm_rgb",
    ])
    mask_dir = find_first_existing([
        year_dir / "selected_top100" / "masks",
        year_dir / "masks",
        year_dir / "masks_multiclass",
        year_dir / "masks_png_all",
    ])

    if img_dir is None or mask_dir is None:
        raise FileNotFoundError(f"Could not find image or mask folders for {year}.")

    image_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]:
        image_files.extend(sorted(img_dir.glob(ext)))

    mask_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]:
        mask_files.extend(sorted(mask_dir.glob(ext)))

    mask_by_key = {normalize_key(p.stem): p for p in mask_files}
    rows = []

    for img_path in image_files:
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
            "day_key": img_path.stem,
            "image_path": str(img_path),
            "mask_path": str(mask_path),
        })

    if not rows:
        raise RuntimeError(f"No valid image-mask matches were found for {year}.")

    return pd.DataFrame(rows)


def compute_full_image_storm_ratios(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in tqdm(records.iterrows(), total=len(records), desc="Computing full-image storm ratios"):
        rgb = read_rgb(Path(row["image_path"]))
        mask = read_mask(Path(row["mask_path"]))
        if rgb.shape[:2] != mask.shape:
            mask_img = Image.fromarray(mask, mode="L")
            mask_img = mask_img.resize((rgb.shape[1], rgb.shape[0]), resample=Image.Resampling.NEAREST)
            mask = np.asarray(mask_img, dtype=np.uint8)

        valid = mask != 2
        storm_ratio = float(np.mean(mask[valid] == 1)) if valid.any() else 0.0
        valid_ratio = float(valid.mean())
        rows.append({
            **row.to_dict(),
            "height": int(mask.shape[0]),
            "width": int(mask.shape[1]),
            "valid_ratio": valid_ratio,
            "storm_ratio": storm_ratio,
        })
    return pd.DataFrame(rows)


def select_validation_and_test_images(all_records: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    val_rows = []
    test_rows = []

    for year in YEARS:
        sub = all_records[all_records["year"] == year].copy().reset_index(drop=True)

        if year == "MY32":
            sub = sub[~sub["day_key"].isin(EXCLUDED_MY32)].copy().reset_index(drop=True)

        if sub.empty:
            raise RuntimeError(f"No eligible images left for {year} after exclusions.")

        in_range = sub[(sub["storm_ratio"] >= TARGET_MIN_STORM) & (sub["storm_ratio"] <= TARGET_MAX_STORM)].copy()

        if len(in_range) >= 3:
            cand = in_range.copy()
        else:
            log(f"[WARN] {year} has only {len(in_range)} images in the 10%-20% range. Falling back to closest images.")
            cand = sub.copy()

        cand["selection_score"] = (cand["storm_ratio"] - TARGET_CENTER).abs()
        cand = cand.sort_values(["selection_score", "storm_ratio"]).reset_index(drop=True)

        if len(cand) < 3:
            raise RuntimeError(f"Not enough candidate images to select 2 validation + 1 test for {year}.")

        chosen = cand.iloc[:3].copy().reset_index(drop=True)
        chosen.loc[0:1, "subset"] = "validation"
        chosen.loc[2, "subset"] = "test"

        val_rows.append(chosen.iloc[0:2].copy())
        test_rows.append(chosen.iloc[[2]].copy())

    val_df = pd.concat(val_rows, ignore_index=True)
    test_df = pd.concat(test_rows, ignore_index=True)
    return val_df, test_df


# ============================================================
# FULL-IMAGE TILED INFERENCE
# ============================================================

def get_tile_starts(length: int, tile: int, stride: int) -> List[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


@torch.no_grad()
def predict_full_image_tiled(model: nn.Module, rgb: np.ndarray, mean: np.ndarray, std: np.ndarray,
                             tile_size: int = TILE_SIZE, stride: int = TILE_STRIDE) -> np.ndarray:
    h, w, _ = rgb.shape
    ys = get_tile_starts(h, tile_size, stride)
    xs = get_tile_starts(w, tile_size, stride)

    prob_sum = np.zeros((h, w), dtype=np.float32)
    prob_count = np.zeros((h, w), dtype=np.float32)

    model.eval()

    for y0 in ys:
        for x0 in xs:
            y1 = y0 + tile_size
            x1 = x0 + tile_size
            patch = rgb[y0:y1, x0:x1].astype(np.float32) / 255.0
            patch = (patch - mean) / (std + 1e-8)
            patch = np.transpose(patch, (2, 0, 1))
            x = torch.tensor(patch[None, ...], dtype=torch.float32, device=DEVICE)
            logits = model(x)
            probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

            prob_sum[y0:y1, x0:x1] += probs
            prob_count[y0:y1, x0:x1] += 1.0

    prob_map = prob_sum / np.maximum(prob_count, 1e-8)
    return prob_map


# ============================================================
# METRICS
# ============================================================

def counts_from_prob_and_mask(prob_map: np.ndarray, mask: np.ndarray, threshold: float) -> Dict[str, int]:
    valid = mask != 2
    pred = prob_map >= threshold
    gt = mask == 1

    tp = int(np.logical_and(pred, gt & valid).sum())
    fp = int(np.logical_and(pred, (~gt) & valid).sum())
    fn = int(np.logical_and(~pred, gt & valid).sum())
    tn = int(np.logical_and(~pred, (~gt) & valid).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
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


def aggregate_counts(counts_list: List[Dict[str, int]]) -> Dict[str, int]:
    total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for c in counts_list:
        for k in total:
            total[k] += int(c[k])
    return total


def threshold_sweep_full_images(items: List[Dict[str, Any]], thresholds: List[float]) -> pd.DataFrame:
    rows = []
    for th in thresholds:
        counts_list = [counts_from_prob_and_mask(item["prob_map"], item["mask"], float(th)) for item in items]
        totals = aggregate_counts(counts_list)
        metrics = metrics_from_counts(**totals)
        rows.append({"threshold": float(th), **metrics})
    return pd.DataFrame(rows)


# ============================================================
# PLOTTING
# ============================================================

def plot_confusion_matrix(metrics: Dict[str, Any], out_png: Path):
    cm = np.array([
        [metrics["tn"], metrics["fp"]],
        [metrics["fn"], metrics["tp"]],
    ], dtype=np.int64)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="viridis")
    ax.set_title(
        f"Test confusion matrix | threshold={metrics['threshold']:.2f}\n"
        f"IoU={metrics['iou']:.4f} | F1={metrics['f1']:.4f}",
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
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(cm, index=["True no storm", "True storm"], columns=["Pred no storm", "Pred storm"]).to_csv(
        out_png.with_suffix(".csv")
    )


def plot_test_triplets(test_items: List[Dict[str, Any]], threshold: float, out_png: Path):
    n = len(test_items)
    fig, axes = plt.subplots(n, 3, figsize=(14, 5 * n))
    if n == 1:
        axes = np.array([axes])

    for i, item in enumerate(test_items):
        rgb = item["rgb"]
        mask = item["mask"]
        prob = item["prob_map"]
        pred = (prob >= threshold).astype(np.uint8)
        pred_display = pred.copy()
        pred_display[mask == 2] = 2

        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(
            f"Original RGB\n{item['year']} | {item['day_key']}\nstorm={item['storm_ratio'] * 100:.1f}%",
            fontsize=11
        )
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask, cmap=CLASS_CMAP, vmin=0, vmax=2)
        axes[i, 1].set_title("Ground truth\n0=no storm | 1=storm | 2=invalid", fontsize=11)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred_display, cmap=CLASS_CMAP, vmin=0, vmax=2)
        axes[i, 2].set_title(f"Prediction\nthreshold={threshold:.2f}", fontsize=11)
        axes[i, 2].axis("off")

    fig.tight_layout()
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    log(f"Project root : {PROJECT_ROOT}")
    log(f"Output dir   : {OUTPUT_DIR}")
    log(f"Device       : {DEVICE}")

    checkpoint_path = MODELS_DIR / "best_attention_unet_multiyear_v2.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load full-image candidate records.
    per_year = []
    for year in YEARS:
        log(f"Scanning {year}...")
        df = load_year_records(year)
        log(f"{year}: {len(df)} matched image-mask pairs found.")
        per_year.append(df)

    all_records = pd.concat(per_year, ignore_index=True)
    all_records = compute_full_image_storm_ratios(all_records)
    all_records.to_csv(OUTPUT_DIR / "all_fullimage_candidates.csv", index=False)

    # Select validation/test images.
    val_df, test_df = select_validation_and_test_images(all_records)
    val_df.to_csv(OUTPUT_DIR / "selected_validation_images.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "selected_test_images.csv", index=False)

    log("\nSelected validation images:")
    log(val_df[["year", "day_key", "storm_ratio", "valid_ratio"]].to_string(index=False))
    log("\nSelected test images:")
    log(test_df[["year", "day_key", "storm_ratio", "valid_ratio"]].to_string(index=False))

    # Load model.
    checkpoint = safe_torch_load(checkpoint_path)
    base_channels = infer_base_channels(checkpoint, BASE_CHANNELS_DEFAULT)
    mean, std = infer_mean_std(checkpoint)
    model = AttentionUNet(in_channels=3, out_channels=1, base_channels=base_channels).to(DEVICE)
    model.load_state_dict(extract_state_dict(checkpoint))
    model.eval()

    # Run full-image inference for validation.
    val_items = []
    for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Validation full-image inference"):
        rgb = read_rgb(Path(row["image_path"]))
        mask = read_mask(Path(row["mask_path"]))
        if rgb.shape[:2] != mask.shape:
            mask_img = Image.fromarray(mask, mode="L")
            mask_img = mask_img.resize((rgb.shape[1], rgb.shape[0]), resample=Image.Resampling.NEAREST)
            mask = np.asarray(mask_img, dtype=np.uint8)
        prob_map = predict_full_image_tiled(model, rgb, mean, std, tile_size=TILE_SIZE, stride=TILE_STRIDE)
        val_items.append({
            **row.to_dict(),
            "rgb": rgb,
            "mask": mask,
            "prob_map": prob_map,
        })

    val_sweep = threshold_sweep_full_images(val_items, THRESHOLDS)
    val_sweep.to_csv(OUTPUT_DIR / "validation_threshold_sweep.csv", index=False)
    best_idx = val_sweep.sort_values(["iou", "f1", "precision"], ascending=[False, False, False]).index[0]
    best_threshold = float(val_sweep.loc[best_idx, "threshold"])

    # Run full-image inference for test.
    test_items = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Test full-image inference"):
        rgb = read_rgb(Path(row["image_path"]))
        mask = read_mask(Path(row["mask_path"]))
        if rgb.shape[:2] != mask.shape:
            mask_img = Image.fromarray(mask, mode="L")
            mask_img = mask_img.resize((rgb.shape[1], rgb.shape[0]), resample=Image.Resampling.NEAREST)
            mask = np.asarray(mask_img, dtype=np.uint8)
        prob_map = predict_full_image_tiled(model, rgb, mean, std, tile_size=TILE_SIZE, stride=TILE_STRIDE)
        test_items.append({
            **row.to_dict(),
            "rgb": rgb,
            "mask": mask,
            "prob_map": prob_map,
        })

    test_sweep = threshold_sweep_full_images(test_items, THRESHOLDS)
    test_sweep.to_csv(OUTPUT_DIR / "test_threshold_sweep.csv", index=False)

    # Final formal metrics: test evaluated at the validation-selected threshold.
    final_idx = (test_sweep["threshold"] - best_threshold).abs().idxmin()
    final_metrics = test_sweep.loc[final_idx].to_dict()
    final_metrics["threshold"] = float(final_metrics["threshold"])
    final_metrics["base_channels"] = int(base_channels)
    final_metrics["tile_size"] = int(TILE_SIZE)
    final_metrics["tile_stride"] = int(TILE_STRIDE)
    pd.DataFrame([final_metrics]).to_csv(OUTPUT_DIR / "test_metrics.csv", index=False)

    # Also save per-image test metrics at the chosen threshold.
    per_image_rows = []
    for item in test_items:
        counts = counts_from_prob_and_mask(item["prob_map"], item["mask"], best_threshold)
        metrics = metrics_from_counts(**counts)
        per_image_rows.append({
            "year": item["year"],
            "day_key": item["day_key"],
            "storm_ratio": item["storm_ratio"],
            **metrics,
        })
    pd.DataFrame(per_image_rows).to_csv(OUTPUT_DIR / "test_metrics_per_image.csv", index=False)

    plot_confusion_matrix(final_metrics, OUTPUT_DIR / "test_confusion_matrix.png")
    plot_test_triplets(test_items, best_threshold, OUTPUT_DIR / "test_three_images_rgb_mask_prediction.png")

    summary = {
        "checkpoint": str(checkpoint_path),
        "base_channels": int(base_channels),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "validation_selected_threshold": float(best_threshold),
        "selected_validation_images": val_df[["year", "day_key", "storm_ratio"]].to_dict(orient="records"),
        "selected_test_images": test_df[["year", "day_key", "storm_ratio"]].to_dict(orient="records"),
        "final_test_metrics": final_metrics,
    }
    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log("\n================ FINAL RESULTS ================")
    log(f"Validation-selected threshold: {best_threshold:.2f}")
    log(pd.DataFrame([final_metrics]).to_string(index=False))
    log(f"\nOutputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
