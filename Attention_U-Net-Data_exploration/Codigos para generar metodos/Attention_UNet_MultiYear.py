# ============================================================
# ATTENTION U-NET MULTI-YEAR TRAINING FOR MARTIAN DUST STORM SEGMENTATION
# Adapted from the original single-year Attention U-Net baseline.
#
# Expected clean structure:
#   ProjectFolder/
#   ├── Attention_UNet_MultiYear.py
#   ├── MY28/selected_top100/images/ and MY28/selected_top100/masks/
#   ├── MY31/selected_top100/images/ and MY31/selected_top100/masks/
#   └── MY32/selected_top100/images/ and MY32/selected_top100/masks/
#
# If MYXX/reports/selected_top100.csv exists, it is used first.
# Otherwise, the script matches image and mask files by filename.
# ============================================================

from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ============================================================
# 1. Configuration
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT_DIR = Path(__file__).resolve().parent
YEARS = ["MY28", "MY31", "MY32"]
OUTPUT_DIR = ROOT_DIR / "attention_unet_multiyear_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]

# Article-style split percentage: 70 / 15 / 15.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
SPLIT_MODE = "shuffle"  # "shuffle" recommended if selected_top100.csv is sorted by storm percentage.

# Original best-baseline settings.
patch_size = 256
stride = 256
min_valid_ratio = 0.70

batch_size = 4
num_epochs = 25
learning_rate = 1e-4
base_channels = 16
pos_weight = 3.0
samples_per_epoch = 5000
num_workers = 2

test_threshold = 0.60
best_model_path = OUTPUT_DIR / "best_attention_unet_multiyear.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 2. Data discovery
# ============================================================

def normalize_key(text: str) -> str:
    return (
        str(text).lower()
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


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Image not found or unreadable: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found or unreadable: {path}")
    return mask


def collect_pairs_from_report(year: str):
    report_path = ROOT_DIR / year / "reports" / "selected_top100.csv"
    if not report_path.exists():
        return None

    df = pd.read_csv(report_path)
    if "image_path" not in df.columns or "mask_png_path" not in df.columns:
        return None

    rows = []
    for i, row in df.iterrows():
        img_path = ROOT_DIR / str(row["image_path"])
        mask_path = ROOT_DIR / str(row["mask_png_path"])
        if not img_path.exists() or not mask_path.exists():
            continue
        day_key = str(row.get("day_key", img_path.stem))
        rows.append({
            "year": year,
            "image_id": f"{year}_{day_key}",
            "day_key": day_key,
            "image_path": str(img_path),
            "mask_path": str(mask_path),
            "source_index": i,
        })
    return pd.DataFrame(rows) if rows else None


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
        raise FileNotFoundError(
            f"Could not find image/mask folders for {year}. Expected, for example, "
            f"{year}/selected_top100/images and {year}/selected_top100/masks."
        )

    images, masks = [], []
    for ext in IMAGE_EXTENSIONS:
        images.extend(sorted(image_dir.glob(f"*{ext}")))
        masks.extend(sorted(mask_dir.glob(f"*{ext}")))

    mask_dict = {normalize_key(p.stem): p for p in masks}
    rows = []

    for i, img_path in enumerate(images):
        img_key = normalize_key(img_path.stem)
        mask_path = mask_dict.get(img_key)

        if mask_path is None:
            for mk, mp in mask_dict.items():
                if img_key in mk or mk in img_key:
                    mask_path = mp
                    break

        if mask_path is None:
            print(f"[WARN] No mask found for {img_path.name}")
            continue

        rows.append({
            "year": year,
            "image_id": f"{year}_{img_path.stem}",
            "day_key": img_path.stem,
            "image_path": str(img_path),
            "mask_path": str(mask_path),
            "source_index": i,
        })

    if not rows:
        raise RuntimeError(f"No valid image-mask pairs matched for {year}.")
    return pd.DataFrame(rows)


def collect_all_image_pairs() -> pd.DataFrame:
    dfs = []
    for year in YEARS:
        print(f"\nCollecting pairs for {year}...")
        df = collect_pairs_from_report(year)
        if df is None:
            print(f"[INFO] No valid report found for {year}. Matching folders instead.")
            df = collect_pairs_from_folders(year)
        else:
            print(f"[INFO] Using {year}/reports/selected_top100.csv")
        print(f"{year}: {len(df)} pairs")
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    out.to_csv(OUTPUT_DIR / "all_image_pairs.csv", index=False)
    return out


def split_pairs_by_year(pair_df: pd.DataFrame) -> pd.DataFrame:
    parts = []
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

        print(
            f"{year}: train={(sub['split']=='train').sum()}, "
            f"val={(sub['split']=='val').sum()}, test={(sub['split']=='test').sum()}"
        )
        parts.append(sub)

    out = pd.concat(parts, ignore_index=True)
    out.to_csv(OUTPUT_DIR / "image_split_manifest.csv", index=False)
    return out


def load_image_and_mask_from_paths(image_path: str, mask_path: str):
    img = read_rgb(Path(image_path))
    mask = read_mask(Path(mask_path))
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    return img, mask

# ============================================================
# 3. Patch inventory
# ============================================================

def patch_category(storm_ratio: float) -> str:
    if storm_ratio == 0:
        return "no_storm"
    if storm_ratio < 0.05:
        return "low_storm"
    if storm_ratio < 0.20:
        return "medium_storm"
    return "high_storm"


def build_patch_inventory(split_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    rows = []
    imgs = split_df[split_df["split"] == split_name].reset_index(drop=True)

    for _, item in tqdm(imgs.iterrows(), total=len(imgs), desc=f"Building {split_name} patches"):
        _, mask = load_image_and_mask_from_paths(item["image_path"], item["mask_path"])
        h, w = mask.shape

        for y0 in range(0, h - patch_size + 1, stride):
            for x0 in range(0, w - patch_size + 1, stride):
                y1, x1 = y0 + patch_size, x0 + patch_size
                patch = mask[y0:y1, x0:x1]
                valid = patch != 2
                valid_ratio = valid.mean()
                if valid_ratio < min_valid_ratio:
                    continue
                storm_ratio = float(np.mean(patch[valid] == 1))
                rows.append({
                    "year": item["year"],
                    "image_id": item["image_id"],
                    "day_key": item["day_key"],
                    "image_path": item["image_path"],
                    "mask_path": item["mask_path"],
                    "split": split_name,
                    "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                    "valid_ratio": float(valid_ratio),
                    "storm_ratio": storm_ratio,
                    "category": patch_category(storm_ratio),
                })

    inv = pd.DataFrame(rows)
    if inv.empty:
        raise RuntimeError(f"No valid patches found for split: {split_name}")
    inv.to_csv(OUTPUT_DIR / f"patch_inventory_{split_name}.csv", index=False)
    return inv

# ============================================================
# 4. Dataset
# ============================================================

class MarsStormDataset(Dataset):
    def __init__(self, patch_inventory: pd.DataFrame, augment: bool = False):
        self.patch_inventory = patch_inventory.reset_index(drop=True)
        self.augment = augment
        self.mean = np.array([0.55, 0.35, 0.25], dtype=np.float32)
        self.std = np.array([0.20, 0.15, 0.15], dtype=np.float32)

    def __len__(self):
        return len(self.patch_inventory)

    def __getitem__(self, idx):
        row = self.patch_inventory.iloc[idx]
        x0, y0, x1, y1 = int(row.x0), int(row.y0), int(row.x1), int(row.y1)

        img, mask = load_image_and_mask_from_paths(row.image_path, row.mask_path)
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
            torch.tensor(valid, dtype=torch.float32),
        )

    def _augment(self, img, target, valid):
        if np.random.rand() < 0.5:
            img, target, valid = np.flip(img, 1).copy(), np.flip(target, 1).copy(), np.flip(valid, 1).copy()
        if np.random.rand() < 0.5:
            img, target, valid = np.flip(img, 0).copy(), np.flip(target, 0).copy(), np.flip(valid, 0).copy()
        k = np.random.randint(0, 4)
        img = np.rot90(img, k, axes=(0, 1)).copy()
        target = np.rot90(target, k, axes=(0, 1)).copy()
        valid = np.rot90(valid, k, axes=(0, 1)).copy()
        return img, target, valid

# ============================================================
# 5. Attention U-Net model
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
        self.gate_conv = nn.Sequential(nn.Conv2d(gate_channels, intermediate_channels, 1), nn.BatchNorm2d(intermediate_channels))
        self.skip_conv = nn.Sequential(nn.Conv2d(skip_channels, intermediate_channels, 1), nn.BatchNorm2d(intermediate_channels))
        self.psi = nn.Sequential(nn.Conv2d(intermediate_channels, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        att = self.relu(self.gate_conv(gate) + self.skip_conv(skip))
        return skip * self.psi(att)


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=16):
        super().__init__()
        f1, f2, f3, f4, f5 = base_channels, base_channels*2, base_channels*4, base_channels*8, base_channels*16
        self.pool = nn.MaxPool2d(2, 2)
        self.enc1 = ConvBlock(in_channels, f1)
        self.enc2 = ConvBlock(f1, f2)
        self.enc3 = ConvBlock(f2, f3)
        self.enc4 = ConvBlock(f3, f4)
        self.bottleneck = ConvBlock(f4, f5)

        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True), nn.Conv2d(f5, f4, 3, padding=1))
        self.att4 = AttentionGate(f4, f4, f3)
        self.dec4 = ConvBlock(f4 + f4, f4)
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True), nn.Conv2d(f4, f3, 3, padding=1))
        self.att3 = AttentionGate(f3, f3, f2)
        self.dec3 = ConvBlock(f3 + f3, f3)
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True), nn.Conv2d(f3, f2, 3, padding=1))
        self.att2 = AttentionGate(f2, f2, f1)
        self.dec2 = ConvBlock(f2 + f2, f2)
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True), nn.Conv2d(f2, f1, 3, padding=1))
        self.att1 = AttentionGate(f1, f1, max(f1 // 2, 1))
        self.dec1 = ConvBlock(f1 + f1, f1)
        self.final = nn.Conv2d(f1, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, self.att4(d4, e4)], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, self.att3(d3, e3)], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, self.att2(d2, e2)], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, self.att1(d1, e1)], dim=1))
        return self.final(d1)

# ============================================================
# 6. Loss and metrics
# ============================================================

class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=3.0, bce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets, valid):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction="none")
        bce = (bce * valid).sum() / (valid.sum() + 1e-8)
        probs = torch.sigmoid(logits) * valid
        targets = targets * valid
        inter = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        dice = 1.0 - (2.0 * inter + 1.0) / (union + 1.0)
        return self.bce_weight * bce + self.dice_weight * dice


def compute_batch_counts(logits, targets, valid, threshold=0.5):
    preds = (torch.sigmoid(logits) >= threshold).float()
    preds = preds[valid == 1]
    targets = targets[valid == 1]
    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    tn = ((preds == 0) & (targets == 0)).sum().item()
    return tp, fp, fn, tn


def counts_to_metrics(tp, fp, fn, tn=0):
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    return precision, recall, f1, iou, accuracy


def evaluate_model(model, loader, criterion, threshold=0.5):
    model.eval()
    total_loss = 0.0
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for images, targets, valid in tqdm(loader, desc=f"Eval threshold={threshold:.2f}"):
            images, targets, valid = images.to(device), targets.to(device), valid.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, targets, valid)
            total_loss += loss.item()
            btp, bfp, bfn, btn = compute_batch_counts(logits, targets, valid, threshold)
            tp += btp; fp += bfp; fn += bfn; tn += btn
    precision, recall, f1, iou, accuracy = counts_to_metrics(tp, fp, fn, tn)
    return {
        "loss": total_loss / len(loader),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1, "iou": iou, "accuracy": accuracy,
    }

# ============================================================
# 7. Visualization
# ============================================================

def predict_patch(model, img_patch, mean, std, threshold):
    x = img_patch.astype(np.float32) / 255.0
    x = (x - mean) / (std + 1e-8)
    x = np.transpose(x, (2, 0, 1))[None]
    x = torch.tensor(x, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            prob = torch.sigmoid(model(x)).squeeze().cpu().numpy()
    pred = (prob >= threshold).astype(np.uint8)
    return prob, pred


def plot_test_patch_predictions(model, test_inventory, mean, std, threshold=test_threshold, n_examples=5):
    cand = test_inventory[(test_inventory.storm_ratio >= 0.10) & (test_inventory.storm_ratio <= 0.50) & (test_inventory.valid_ratio >= 0.95)].copy()
    if cand.empty:
        cand = test_inventory[test_inventory.storm_ratio > 0].copy()
    if cand.empty:
        cand = test_inventory.copy()
    cand["score"] = abs(cand.storm_ratio - 0.30)
    selected = cand.sort_values("score").head(n_examples).reset_index(drop=True)

    fig, axes = plt.subplots(len(selected), 3, figsize=(13, 4.2 * len(selected)))
    if len(selected) == 1:
        axes = np.expand_dims(axes, 0)

    for i, row in selected.iterrows():
        img, mask = load_image_and_mask_from_paths(row.image_path, row.mask_path)
        x0, y0, x1, y1 = int(row.x0), int(row.y0), int(row.x1), int(row.y1)
        imgp = img[y0:y1, x0:x1]
        maskp = mask[y0:y1, x0:x1]
        target = (maskp == 1).astype(np.uint8)
        valid = (maskp != 2).astype(np.uint8)
        _, pred = predict_patch(model, imgp, mean, std, threshold)
        pred[valid == 0] = 0
        gt_display = target.copy(); gt_display[valid == 0] = 2
        pred_display = pred.copy(); pred_display[valid == 0] = 2

        axes[i, 0].imshow(imgp); axes[i, 0].set_title(f"Original\n{row.year} | {row.day_key}\nstorm={row.storm_ratio:.1%}"); axes[i, 0].axis("off")
        axes[i, 1].imshow(gt_display, vmin=0, vmax=2); axes[i, 1].set_title("Ground truth"); axes[i, 1].axis("off")
        axes[i, 2].imshow(pred_display, vmin=0, vmax=2); axes[i, 2].set_title(f"Prediction | threshold={threshold:.2f}"); axes[i, 2].axis("off")

    fig.tight_layout()
    out = OUTPUT_DIR / "test_5_patch_predictions.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualization saved: {out}")


def plot_confusion_matrix(metrics):
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]], dtype=np.int64)
    pd.DataFrame(cm, index=["True no storm", "True storm"], columns=["Pred no storm", "Pred storm"]).to_csv(OUTPUT_DIR / "test_confusion_matrix.csv")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.set_title(f"Test Confusion Matrix | threshold={test_threshold:.2f}\nIoU={metrics['iou']:.4f} | F1={metrics['f1']:.4f}")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred no storm", "Pred storm"]); ax.set_yticklabels(["True no storm", "True storm"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center")
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = OUTPUT_DIR / "test_confusion_matrix.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix saved: {out}")

# ============================================================
# 8. Main execution
# ============================================================

if __name__ == "__main__":
    print("============================================================")
    print("Attention U-Net Multi-Year Training")
    print("============================================================")
    print(f"Root folder: {ROOT_DIR}")
    print(f"Years: {YEARS}")
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    pair_df = collect_all_image_pairs()
    split_df = split_pairs_by_year(pair_df)

    print("\nBuilding patch inventories...")
    train_inv = build_patch_inventory(split_df, "train")
    val_inv = build_patch_inventory(split_df, "val")
    test_inv = build_patch_inventory(split_df, "test")
    pd.concat([train_inv, val_inv, test_inv], ignore_index=True).to_csv(OUTPUT_DIR / "patch_inventory_all.csv", index=False)

    print("\nPatch summary:")
    print(f"Train patches: {len(train_inv)}")
    print(f"Val patches  : {len(val_inv)}")
    print(f"Test patches : {len(test_inv)}")
    print("\nTrain categories:")
    print(train_inv.category.value_counts())
    print("\nValidation categories:")
    print(val_inv.category.value_counts())
    print("\nTest categories:")
    print(test_inv.category.value_counts())

    train_ds = MarsStormDataset(train_inv, augment=True)
    val_ds = MarsStormDataset(val_inv, augment=False)
    test_ds = MarsStormDataset(test_inv, augment=False)

    category_weights = {"no_storm": 0.5, "low_storm": 2.0, "medium_storm": 3.0, "high_storm": 4.0}
    sample_weights = train_inv.category.map(category_weights).fillna(1.0).values
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=min(samples_per_epoch, len(train_inv)),
        replacement=True,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))

    model = AttentionUNet(in_channels=3, out_channels=1, base_channels=base_channels).to(device)
    criterion = BCEDiceLoss(pos_weight=pos_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_iou = 0.0
    history = []

    print("\nStarting training...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, targets, valid in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            images, targets, valid = images.to(device), targets.to(device), valid.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, targets, valid)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_metrics = evaluate_model(model, val_loader, criterion, threshold=0.5)

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_metrics["iou"])
        new_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_iou": val_metrics["iou"],
            "val_accuracy": val_metrics["accuracy"],
            "learning_rate": new_lr,
        })
        pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history.csv", index=False)

        print(
            f"Epoch [{epoch + 1:02d}/{num_epochs}] | "
            f"T.Loss: {train_loss:.4f} | V.Loss: {val_metrics['loss']:.4f} | "
            f"Prec: {val_metrics['precision']:.4f} | Rec: {val_metrics['recall']:.4f} | "
            f"F1: {val_metrics['f1']:.4f} | IoU: {val_metrics['iou']:.4f} | LR: {new_lr:.2e}"
        )

        if new_lr < old_lr:
            print(f"-> Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e}")

        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best_val_iou": best_val_iou,
                "base_channels": base_channels,
                "patch_size": patch_size,
                "stride": stride,
                "mean": train_ds.mean.tolist(),
                "std": train_ds.std.tolist(),
                "years": YEARS,
                "test_threshold": test_threshold,
            }, best_model_path)
            print(f"-> New best model saved. Val IoU: {best_val_iou:.4f}")

    print("\nTraining completed.")
    print(f"Best validation IoU: {best_val_iou:.4f}")
    print(f"Best model saved at: {best_model_path}")

    print("\nLoading best checkpoint for final test evaluation...")
    try:
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(best_model_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_metrics = evaluate_model(model, test_loader, criterion, threshold=test_threshold)
    pd.DataFrame([test_metrics]).to_csv(OUTPUT_DIR / "test_metrics.csv", index=False)

    print("\n========== FINAL TEST RESULTS ==========")
    print(f"Threshold : {test_threshold:.2f}")
    print(f"Loss      : {test_metrics['loss']:.4f}")
    print(f"Accuracy  : {test_metrics['accuracy']:.4f}")
    print(f"Precision : {test_metrics['precision']:.4f}")
    print(f"Recall    : {test_metrics['recall']:.4f}")
    print(f"F1-score  : {test_metrics['f1']:.4f}")
    print(f"IoU       : {test_metrics['iou']:.4f}")
    print(f"TN        : {test_metrics['tn']}")
    print(f"FP        : {test_metrics['fp']}")
    print(f"FN        : {test_metrics['fn']}")
    print(f"TP        : {test_metrics['tp']}")

    plot_confusion_matrix(test_metrics)
    plot_test_patch_predictions(model, test_inv, train_ds.mean, train_ds.std, threshold=test_threshold, n_examples=5)

    print("\nCompleted.")
    print(f"All outputs saved in: {OUTPUT_DIR}")
