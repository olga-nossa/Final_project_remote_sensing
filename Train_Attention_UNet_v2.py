# ============================================================
# ATTENTION U-NET V2 TRAINING FOR MARTIAN DUST STORM SEGMENTATION
# Improved version: stride overlap, train normalization, GroupNorm,
# BCE + Tversky loss, early stopping, validation threshold selection,
# final test evaluation and visualization.
# ============================================================

import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# ============================================================
# 1. General configuration
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")


# ============================================================
# 2. Dataset paths and image split
# ============================================================

image_folder = Path("MARS/images")
mask_folder = Path("MARS/masks_multiclass")

train_images = [
    "346_F04_day22", "341_F04_day17", "328_F04_day04", "353_F04_day29",
    "082_D17_day19", "340_F04_day16", "106_D18_day10", "090_D17_day27",
    "357_F04_day33", "030_D15_day32", "290_F02_day31", "314_F03_day22"
]

val_images = [
    "354_F04_day30",
    "330_F04_day06"
]

test_images = [
    "345_F04_day21",
    "089_D17_day26",
    "031_D15_day33"
]


# ============================================================
# 3. Training parameters
# ============================================================

patch_size = 256
stride = 128
min_valid_ratio = 0.70

batch_size = 2
num_epochs = 50
learning_rate = 1e-4

base_channels = 32
pos_weight = 3.0

samples_per_epoch = 1200
early_stopping_patience = 10

best_model_path = "best_attention_unet_mars_v2.pth"
history_csv_path = "training_history_attention_unet_v2.csv"
test_figure_path = "test_predictions_attention_unet_v2.png"


# ============================================================
# 4. Image and mask loading
# ============================================================

def load_image_and_mask(name):
    """
    Loads RGB image and multiclass mask.

    Mask convention:
    0 = no storm
    1 = storm
    2 = no-data / invalid pixel
    """

    img_path = image_folder / f"{name}.png"
    mask_path = mask_folder / f"{name}_multiclass.png"

    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    return img, mask


# ============================================================
# 5. Train normalization
# ============================================================

def compute_train_mean_std(image_list):
    """
    Computes RGB mean and std using only valid pixels from the training set.
    This avoids using approximate normalization values.
    """

    all_pixels = []

    for name in image_list:
        img, mask = load_image_and_mask(name)

        img = img.astype(np.float32) / 255.0
        valid = mask != 2

        valid_pixels = img[valid]

        if len(valid_pixels) > 0:
            all_pixels.append(valid_pixels)

    if len(all_pixels) == 0:
        raise RuntimeError("No valid pixels found to compute mean/std.")

    all_pixels = np.concatenate(all_pixels, axis=0)

    mean = all_pixels.mean(axis=0).astype(np.float32)
    std = all_pixels.std(axis=0).astype(np.float32)

    std = np.maximum(std, 1e-6)

    return mean, std


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


def build_patch_inventory(image_list):
    """
    Builds a dataframe with patch coordinates, valid-pixel ratio,
    storm ratio and storm category.
    """

    rows = []

    for name in image_list:
        _, mask = load_image_and_mask(name)

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
                    "image": name,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "valid_ratio": valid_ratio,
                    "storm_ratio": storm_ratio,
                    "category": patch_category(storm_ratio)
                })

    inventory = pd.DataFrame(rows)

    if len(inventory) == 0:
        raise RuntimeError(
            "No valid patches found. Check patch_size, stride, "
            "min_valid_ratio, image paths and mask values."
        )

    return inventory


# ============================================================
# 7. Dataset
# ============================================================

class MarsStormDataset(Dataset):
    def __init__(self, patch_inventory, mean, std, augment=False):
        self.patch_inventory = patch_inventory.reset_index(drop=True)
        self.mean = mean
        self.std = std
        self.augment = augment

    def __len__(self):
        return len(self.patch_inventory)

    def __getitem__(self, idx):
        row = self.patch_inventory.iloc[idx]

        name = row["image"]
        x0 = int(row["x0"])
        y0 = int(row["y0"])
        x1 = int(row["x1"])
        y1 = int(row["y1"])

        img, mask = load_image_and_mask(name)

        img_patch = img[y0:y1, x0:x1].astype(np.float32) / 255.0
        mask_patch = mask[y0:y1, x0:x1]

        target = (mask_patch == 1).astype(np.float32)
        valid = (mask_patch != 2).astype(np.float32)

        if self.augment:
            img_patch, target, valid = self._augment(img_patch, target, valid)

        img_patch = (img_patch - self.mean) / (self.std + 1e-8)

        img_patch = np.transpose(img_patch, (2, 0, 1))
        target = target[None, :, :]
        valid = valid[None, :, :]

        return (
            torch.tensor(img_patch, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(valid, dtype=torch.float32)
        )

    def _augment(self, img, target, valid):
        """
        Spatial and mild photometric augmentation.
        Photometric augmentation is applied before normalization.
        """

        if np.random.rand() < 0.5:
            img = np.flip(img, axis=1).copy()
            target = np.flip(target, axis=1).copy()
            valid = np.flip(valid, axis=1).copy()

        if np.random.rand() < 0.5:
            img = np.flip(img, axis=0).copy()
            target = np.flip(target, axis=0).copy()
            valid = np.flip(valid, axis=0).copy()

        if np.random.rand() < 0.5:
            brightness = np.random.uniform(0.90, 1.10)
            contrast = np.random.uniform(0.90, 1.10)

            img_mean = img.mean(axis=(0, 1), keepdims=True)
            img = (img - img_mean) * contrast + img_mean
            img = img * brightness
            img = np.clip(img, 0.0, 1.0)

        return img, target, valid


# ============================================================
# 8. Attention U-Net with GroupNorm
# ============================================================

def norm_layer(num_channels):
    groups = min(8, num_channels)
    return nn.GroupNorm(num_groups=groups, num_channels=num_channels)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    def __init__(self, gate_channels, skip_channels, intermediate_channels):
        super().__init__()

        self.gate_conv = nn.Sequential(
            nn.Conv2d(gate_channels, intermediate_channels, kernel_size=1),
            norm_layer(intermediate_channels)
        )

        self.skip_conv = nn.Sequential(
            nn.Conv2d(skip_channels, intermediate_channels, kernel_size=1),
            norm_layer(intermediate_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(intermediate_channels, 1, kernel_size=1),
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
# 9. Loss: BCE + Tversky
# ============================================================

class BCETverskyLoss(nn.Module):
    def __init__(
        self,
        pos_weight=3.0,
        alpha=0.3,
        beta=0.7,
        bce_weight=1.0,
        tversky_weight=1.0
    ):
        super().__init__()

        self.register_buffer(
            "pos_weight",
            torch.tensor([pos_weight], dtype=torch.float32)
        )

        self.alpha = alpha
        self.beta = beta
        self.bce_weight = bce_weight
        self.tversky_weight = tversky_weight

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

        tp = (probs * targets).sum()
        fp = (probs * (1.0 - targets) * valid).sum()
        fn = ((1.0 - probs) * targets).sum()

        smooth = 1.0
        tversky = (tp + smooth) / (
            tp + self.alpha * fp + self.beta * fn + smooth
        )

        tversky_loss = 1.0 - tversky

        return self.bce_weight * bce + self.tversky_weight * tversky_loss


# ============================================================
# 10. Metrics
# ============================================================

def compute_batch_counts(logits, targets, valid, threshold=0.5):
    preds = (torch.sigmoid(logits) >= threshold).float()

    preds = preds[valid == 1]
    targets = targets[valid == 1]

    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()

    return tp, fp, fn


def counts_to_metrics(tp, fp, fn):
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    return precision, recall, f1, iou


def evaluate_model(model, loader, criterion, threshold=0.5):
    model.eval()

    total_loss = 0.0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.no_grad():
        for images, targets, valid in loader:
            images = images.to(device)
            targets = targets.to(device)
            valid = valid.to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda")
            ):
                logits = model(images)
                loss = criterion(logits, targets, valid)

            total_loss += loss.item()

            tp, fp, fn = compute_batch_counts(
                logits,
                targets,
                valid,
                threshold=threshold
            )

            total_tp += tp
            total_fp += fp
            total_fn += fn

    avg_loss = total_loss / len(loader)

    precision, recall, f1, iou = counts_to_metrics(
        total_tp,
        total_fp,
        total_fn
    )

    return {
        "loss": avg_loss,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou
    }


def threshold_sweep(model, loader, criterion):
    thresholds = np.arange(0.1, 0.91, 0.1)
    rows = []

    for threshold in thresholds:
        metrics = evaluate_model(
            model,
            loader,
            criterion,
            threshold=float(threshold)
        )

        rows.append({
            "threshold": float(threshold),
            "loss": metrics["loss"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "iou": metrics["iou"]
        })

    return pd.DataFrame(rows)


# ============================================================
# 11. Full-image prediction and visualization
# ============================================================

def predict_full_image_tiled(model, image_name, mean, std, threshold=0.5):
    img, mask = load_image_and_mask(image_name)

    h, w = mask.shape

    prob_sum = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    model.eval()

    with torch.no_grad():
        for y0 in range(0, h - patch_size + 1, stride):
            for x0 in range(0, w - patch_size + 1, stride):

                y1 = y0 + patch_size
                x1 = x0 + patch_size

                img_patch = img[y0:y1, x0:x1].astype(np.float32) / 255.0
                img_patch = (img_patch - mean) / (std + 1e-8)

                img_patch = np.transpose(img_patch, (2, 0, 1))

                img_tensor = torch.tensor(
                    img_patch[None, :, :, :],
                    dtype=torch.float32
                ).to(device)

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=(device.type == "cuda")
                ):
                    logits = model(img_tensor)
                    probs = torch.sigmoid(logits)

                probs = probs.squeeze().detach().cpu().numpy()

                prob_sum[y0:y1, x0:x1] += probs
                count_map[y0:y1, x0:x1] += 1.0

    prob_map = prob_sum / np.maximum(count_map, 1e-8)

    valid_mask = mask != 2
    target_mask = (mask == 1).astype(np.uint8)
    pred_mask = (prob_map >= threshold).astype(np.uint8)

    pred_mask[~valid_mask] = 0

    return img, mask, target_mask, prob_map, pred_mask


def remove_small_components(pred_mask, min_area=40):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        pred_mask.astype(np.uint8),
        connectivity=8
    )

    cleaned = np.zeros_like(pred_mask, dtype=np.uint8)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area >= min_area:
            cleaned[labels == label] = 1

    return cleaned


def plot_test_predictions(model, image_names, mean, std, threshold=0.5):
    n = len(image_names)

    fig, axes = plt.subplots(
        nrows=n,
        ncols=3,
        figsize=(15, 5 * n)
    )

    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, image_name in enumerate(image_names):
        img, mask, target_mask, prob_map, pred_mask = predict_full_image_tiled(
            model,
            image_name,
            mean,
            std,
            threshold=threshold
        )

        pred_mask = remove_small_components(pred_mask, min_area=40)

        valid_mask = mask != 2

        gt_display = target_mask.copy()
        gt_display[~valid_mask] = 2

        pred_display = pred_mask.copy()
        pred_display[~valid_mask] = 2

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Original image\n{image_name}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt_display, vmin=0, vmax=2)
        axes[i, 1].set_title("Ground-truth mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred_display, vmin=0, vmax=2)
        axes[i, 2].set_title(f"Predicted mask\nthreshold = {threshold:.2f}")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(test_figure_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nTest prediction figure saved as: {test_figure_path}")


# ============================================================
# 12. Main execution
# ============================================================

if __name__ == "__main__":

    print("\nComputing training normalization...")
    train_mean, train_std = compute_train_mean_std(train_images)

    print(f"Train mean: {train_mean}")
    print(f"Train std : {train_std}")

    print("\nBuilding patch inventories...")
    train_inventory = build_patch_inventory(train_images)
    val_inventory = build_patch_inventory(val_images)
    test_inventory = build_patch_inventory(test_images)

    print("\nTrain patch categories:")
    print(train_inventory["category"].value_counts())

    print("\nValidation patch categories:")
    print(val_inventory["category"].value_counts())

    print("\nTest patch categories:")
    print(test_inventory["category"].value_counts())

    train_dataset = MarsStormDataset(
        train_inventory,
        mean=train_mean,
        std=train_std,
        augment=True
    )

    val_dataset = MarsStormDataset(
        val_inventory,
        mean=train_mean,
        std=train_std,
        augment=False
    )

    test_dataset = MarsStormDataset(
        test_inventory,
        mean=train_mean,
        std=train_std,
        augment=False
    )

    category_weights = {
        "no_storm": 0.5,
        "low_storm": 2.0,
        "medium_storm": 3.0,
        "high_storm": 4.0
    }

    sample_weights = (
        train_inventory["category"]
        .map(category_weights)
        .fillna(1.0)
        .values
    )

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=min(samples_per_epoch, len(train_inventory)),
        replacement=True
    )

    num_workers = 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )

    model = AttentionUNet(
        in_channels=3,
        out_channels=1,
        base_channels=base_channels
    ).to(device)

    criterion = BCETverskyLoss(
        pos_weight=pos_weight,
        alpha=0.3,
        beta=0.7
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

    scaler = torch.cuda.amp.GradScaler(
        enabled=(device.type == "cuda")
    )

    best_val_iou = 0.0
    epochs_without_improvement = 0
    history = []

    print("\nStarting training...")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, targets, valid in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            valid = valid.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda")
            ):
                logits = model(images)
                loss = criterion(logits, targets, valid)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        val_metrics = evaluate_model(
            model,
            val_loader,
            criterion,
            threshold=0.5
        )

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
            "learning_rate": new_lr
        })

        print(
            f"Epoch [{epoch + 1:02d}/{num_epochs}] | "
            f"T.Loss: {train_loss:.4f} | "
            f"V.Loss: {val_metrics['loss']:.4f} | "
            f"Prec: {val_metrics['precision']:.4f} | "
            f"Rec: {val_metrics['recall']:.4f} | "
            f"F1: {val_metrics['f1']:.4f} | "
            f"IoU: {val_metrics['iou']:.4f} | "
            f"LR: {new_lr:.2e}"
        )

        if new_lr < old_lr:
            print(f"-> Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e}")

        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "best_val_iou": best_val_iou,
                    "base_channels": base_channels,
                    "patch_size": patch_size,
                    "stride": stride,
                    "mean": train_mean,
                    "std": train_std
                },
                best_model_path
            )

            print(f"-> New best model saved. Val IoU: {best_val_iou:.4f}")

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print("\nEarly stopping activated.")
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(history_csv_path, index=False)
    print(f"\nTraining history saved as: {history_csv_path}")

    print("\nLoading best model for final evaluation...")

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Best saved epoch: {checkpoint['epoch']}")
    print(f"Best validation IoU: {checkpoint['best_val_iou']:.4f}")

    print("\nSelecting best threshold using validation set...")

    val_sweep_df = threshold_sweep(
        model,
        val_loader,
        criterion
    )

    print("\n========== VALIDATION THRESHOLD SWEEP ==========")
    print(val_sweep_df.round(4).to_string(index=False))

    best_threshold_row = val_sweep_df.loc[val_sweep_df["iou"].idxmax()]
    best_threshold = float(best_threshold_row["threshold"])

    print("\nBest threshold selected from validation:")
    print(f"Threshold : {best_threshold:.2f}")
    print(f"Val IoU   : {best_threshold_row['iou']:.4f}")
    print(f"Val F1    : {best_threshold_row['f1']:.4f}")

    final_val_metrics = evaluate_model(
        model,
        val_loader,
        criterion,
        threshold=best_threshold
    )

    final_test_metrics = evaluate_model(
        model,
        test_loader,
        criterion,
        threshold=best_threshold
    )

    print("\n========== FINAL VALIDATION RESULTS ==========")
    print(f"Threshold : {best_threshold:.2f}")
    print(f"Loss      : {final_val_metrics['loss']:.4f}")
    print(f"Precision : {final_val_metrics['precision']:.4f}")
    print(f"Recall    : {final_val_metrics['recall']:.4f}")
    print(f"F1-score  : {final_val_metrics['f1']:.4f}")
    print(f"IoU       : {final_val_metrics['iou']:.4f}")

    print("\n============== FINAL TEST RESULTS =============")
    print(f"Threshold : {best_threshold:.2f}")
    print(f"Loss      : {final_test_metrics['loss']:.4f}")
    print(f"Precision : {final_test_metrics['precision']:.4f}")
    print(f"Recall    : {final_test_metrics['recall']:.4f}")
    print(f"F1-score  : {final_test_metrics['f1']:.4f}")
    print(f"IoU       : {final_test_metrics['iou']:.4f}")

    print("\nGenerating test prediction figure...")

    plot_test_predictions(
        model,
        test_images,
        mean=train_mean,
        std=train_std,
        threshold=best_threshold
    )

    print("\nCompleted.")
    print(f"Best model saved as: {best_model_path}")