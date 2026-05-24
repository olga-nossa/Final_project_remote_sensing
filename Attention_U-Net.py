# ============================================================
# ATTENTION U-NET TRAINING FOR MARTIAN DUST STORM SEGMENTATION
# Optimizado para GPU y estabilidad de entrenamiento
# ============================================================

import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ------------------------------------------------------------
# 1. Imports and general configuration
# ------------------------------------------------------------

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    # Activa la búsqueda del mejor algoritmo convolucional para tu hardware
    torch.backends.cudnn.benchmark = True
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# ------------------------------------------------------------
# 2. Dataset paths and image split
# ------------------------------------------------------------

image_folder = Path("MARS/images")
mask_folder = Path("MARS/masks_multiclass")

train_images = [
    "346_F04_day22", "341_F04_day17", "328_F04_day04", "353_F04_day29",
    "082_D17_day19", "340_F04_day16", "106_D18_day10", "090_D17_day27",
    "357_F04_day33", "030_D15_day32", "290_F02_day31", "314_F03_day22"
]

val_images = ["354_F04_day30", "330_F04_day06"]
test_images = ["345_F04_day21", "089_D17_day26", "031_D15_day33"]

# ------------------------------------------------------------
# 3. Training parameters
# ------------------------------------------------------------

patch_size = 256
stride = 256
min_valid_ratio = 0.70

batch_size = 4          
num_epochs = 25
learning_rate = 1e-4

base_channels = 16      
pos_weight = 3.0        

samples_per_epoch = 800 

# ------------------------------------------------------------
# 4. Image and mask loading
# ------------------------------------------------------------

def load_image_and_mask(name):
    """
    Loads RGB image and multiclass mask.
    0 = no storm, 1 = storm, 2 = no-data / invalid pixel
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

# ------------------------------------------------------------
# 5. Patch inventory construction
# ------------------------------------------------------------

def patch_category(storm_ratio):
    if storm_ratio == 0: return "no_storm"
    elif storm_ratio < 0.05: return "low_storm"
    elif storm_ratio < 0.20: return "medium_storm"
    else: return "high_storm"

def build_patch_inventory(image_list):
    rows = []
    for name in image_list:
        _, mask = load_image_and_mask(name)
        h, w = mask.shape

        for y0 in range(0, h - patch_size + 1, stride):
            for x0 in range(0, w - patch_size + 1, stride):
                y1, x1 = y0 + patch_size, x0 + patch_size
                mask_patch = mask[y0:y1, x0:x1]

                valid = mask_patch != 2
                valid_ratio = valid.mean()

                if valid_ratio < min_valid_ratio: continue

                storm_ratio = np.mean(mask_patch[valid] == 1)
                rows.append({
                    "image": name, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                    "valid_ratio": valid_ratio, "storm_ratio": storm_ratio,
                    "category": patch_category(storm_ratio)
                })
    return pd.DataFrame(rows)

# ------------------------------------------------------------
# 6. PyTorch Dataset
# ------------------------------------------------------------

class MarsStormDataset(Dataset):
    def __init__(self, patch_inventory, augment=False):
        self.patch_inventory = patch_inventory.reset_index(drop=True)
        self.augment = augment
        
        # Z-score Normalization (Valores aproximados para Marte: sesgo rojo)
        # Nota: Idealmente recalcula estos valores con tu dataset completo.
        self.mean = np.array([0.55, 0.35, 0.25], dtype=np.float32)
        self.std = np.array([0.20, 0.15, 0.15], dtype=np.float32)

    def __len__(self):
        return len(self.patch_inventory)

    def __getitem__(self, idx):
        row = self.patch_inventory.iloc[idx]
        name = row["image"]
        x0, y0, x1, y1 = int(row["x0"]), int(row["y0"]), int(row["x1"]), int(row["y1"])

        # Se carga al vuelo para evitar fugas masivas de memoria RAM (OOM)
        img, mask = load_image_and_mask(name)

        img_patch = img[y0:y1, x0:x1].astype(np.float32) / 255.0
        # Aplicar Z-score normalization para estabilizar características marcianas
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
            img, target, valid = np.flip(img, 1).copy(), np.flip(target, 1).copy(), np.flip(valid, 1).copy()
        if np.random.rand() < 0.5:
            img, target, valid = np.flip(img, 0).copy(), np.flip(target, 0).copy(), np.flip(valid, 0).copy()
        k = np.random.randint(0, 4)
        return np.rot90(img, k, (0,1)).copy(), np.rot90(target, k, (0,1)).copy(), np.rot90(valid, k, (0,1)).copy()

# ------------------------------------------------------------
# 8. Attention U-Net architecture
# ------------------------------------------------------------

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
        self.gate_conv = nn.Sequential(nn.Conv2d(gate_channels, intermediate_channels, kernel_size=1), nn.BatchNorm2d(intermediate_channels))
        self.skip_conv = nn.Sequential(nn.Conv2d(skip_channels, intermediate_channels, kernel_size=1), nn.BatchNorm2d(intermediate_channels))
        self.psi = nn.Sequential(nn.Conv2d(intermediate_channels, 1, kernel_size=1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        attention = self.relu(self.gate_conv(gate) + self.skip_conv(skip))
        return skip * self.psi(attention)

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=16):
        super().__init__()
        f1, f2, f3, f4, f5 = base_channels, base_channels*2, base_channels*4, base_channels*8, base_channels*16

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc1 = ConvBlock(in_channels, f1)
        self.enc2 = ConvBlock(f1, f2)
        self.enc3 = ConvBlock(f2, f3)
        self.enc4 = ConvBlock(f3, f4)
        self.bottleneck = ConvBlock(f4, f5)

        # Reemplazo de ConvTranspose2d por Interpolación Bilineal (Evita checkerboard artifacts)
        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(f5, f4, kernel_size=3, padding=1))
        self.att4 = AttentionGate(f4, f4, f3)
        self.dec4 = ConvBlock(f4 + f4, f4)

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(f4, f3, kernel_size=3, padding=1))
        self.att3 = AttentionGate(f3, f3, f2)
        self.dec3 = ConvBlock(f3 + f3, f3)

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(f3, f2, kernel_size=3, padding=1))
        self.att2 = AttentionGate(f2, f2, f1)
        self.dec2 = ConvBlock(f2 + f2, f2)

        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(f2, f1, kernel_size=3, padding=1))
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
        e4 = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        e3 = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        e2 = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        e1 = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final(d1)

# ------------------------------------------------------------
# 9. Loss function: BCE + Dice
# ------------------------------------------------------------

class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=3.0, bce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets, valid):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        bce = (bce * valid).sum() / (valid.sum() + 1e-8)

        probs = torch.sigmoid(logits) * valid
        targets = targets * valid

        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()

        # Corrección de estabilidad numérica en Dice Loss (smooth=1.0)
        smooth = 1.0
        dice = 1.0 - (2.0 * intersection + smooth) / (union + smooth)

        return self.bce_weight * bce + self.dice_weight * dice

# ------------------------------------------------------------
# 10. Metrics for validation
# ------------------------------------------------------------

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
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return precision, recall, f1, iou

# ============================================================
# MAIN EXECUTION BLOCK (Evita Fork Bombs en Windows)
# ============================================================
if __name__ == '__main__':
    print("Construyendo inventarios...")
    train_patch_inventory = build_patch_inventory(train_images)
    val_patch_inventory = build_patch_inventory(val_images)

    train_dataset = MarsStormDataset(train_patch_inventory, augment=True)
    val_dataset = MarsStormDataset(val_patch_inventory, augment=False)

    category_weights = {"no_storm": 0.5, "low_storm": 2.0, "medium_storm": 3.0, "high_storm": 4.0}
    sample_weights = train_patch_inventory["category"].map(category_weights).fillna(1.0).values

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=min(samples_per_epoch, len(train_patch_inventory)),
        replacement=True
    )

    # num_workers optimizados a 4 para mantener la GPU alimentada
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = AttentionUNet(in_channels=3, out_channels=1, base_channels=base_channels).to(device)
    criterion = BCEDiceLoss(pos_weight=pos_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Scheduler: Reduce el LR si el IoU se estanca
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    # Scaler para Precisión Mixta Automática (Ahorra VRAM y acelera el entrenamiento)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_iou = 0.0
    best_model_path = "best_attention_unet_mars.pth"

    print("\nIniciando entrenamiento...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, targets, valid in train_loader:
            images, targets, valid = images.to(device), targets.to(device), valid.to(device)
            optimizer.zero_grad()

            # Autocast: Ejecuta las operaciones seguras en 16 bits
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, targets, valid)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validación
        model.eval()
        val_loss, total_tp, total_fp, total_fn = 0.0, 0, 0, 0

        with torch.no_grad():
            for images, targets, valid in val_loader:
                images, targets, valid = images.to(device), targets.to(device), valid.to(device)
                
                # Validación también se beneficia del Mixed Precision
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    logits = model(images)
                    loss = criterion(logits, targets, valid)

                val_loss += loss.item()
                tp, fp, fn = compute_batch_counts(logits, targets, valid)
                total_tp += tp
                total_fp += fp
                total_fn += fn

        val_loss /= len(val_loader)
        val_prec, val_rec, val_f1, val_iou = counts_to_metrics(total_tp, total_fp, total_fn)

        print(f"Epoch [{epoch + 1:02d}/{num_epochs}] | T.Loss: {train_loss:.4f} | V.Loss: {val_loss:.4f} | IoU: {val_iou:.4f}")

        # Actualiza el scheduler basado en el IoU de validación
        scheduler.step(val_iou)

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), best_model_path)
            print(f"-> Nuevo mejor modelo guardado (IoU: {best_val_iou:.4f})")

    print(f"\nEntrenamiento completado. Mejor IoU: {best_val_iou:.4f}")