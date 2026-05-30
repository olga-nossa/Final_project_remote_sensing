# Martian Dust Storm Segmentation Repository

This repository contains the codes and data organization used for the segmentation of Martian dust storms from MARCI Mars Daily Global Maps (MDGMs).

## Repository Structure

### 1. `Attention_UNet_models/`

This folder contains the codes used to train, compare, and evaluate different U-Net-based models. Several architectures were tested, and the Attention U-Net model was selected as the best-performing model among them.

Before running any code in this folder, the corresponding dataset must be downloaded from the following link:

**Attention U-Net data:** https://drive.google.com/drive/folders/1GH81_s1E4vshUw4yvgNbHCwcAvAotsoi?usp=drive_link

After downloading the data, place the files in the expected data directory indicated in the scripts.

---

### 2. `DeepLab_training/`

This folder contains the codes used to train and validate the DeepLabv3 model. This model was implemented as an additional deep learning alternative to evaluate whether a more advanced segmentation architecture could improve the results obtained with the U-Net-based models.

Before running any code in this folder, the corresponding dataset must be downloaded from the following link:

**DeepLab data:** https://drive.google.com/drive/folders/1GH81_s1E4vshUw4yvgNbHCwcAvAotsoi?usp=drive_link

After downloading the data, place the files in the expected data directory indicated in the scripts.

---

### 3. `Random_Forest_data/`

This folder contains the files and data used for the Random Forest models. These models were used as classical machine learning baselines for the pixel-level classification of storm and no-storm regions.

---

## Important Note

The image files and their corresponding mask pairings are not stored directly in this repository because their total size exceeds 10 GB. For this reason, the datasets required for the Attention U-Net and DeepLabv3 models are provided through external cloud storage links.

Please download the corresponding data before executing any training, validation, or preprocessing script.