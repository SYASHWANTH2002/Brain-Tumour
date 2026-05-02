import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from dataset import BraTSDataset
from model import AttentionCNNTransformerUNet

# =====================
# CONFIG
# =====================
DATA_ROOT = "data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
CKPT = "model_final.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)


# =====================
# Dice function (WT)
# =====================
def dice_score(pred, gt, eps=1e-6):
    pred = pred.float()
    gt = gt.float()
    intersection = (pred * gt).sum()
    return (2 * intersection + eps) / (pred.sum() + gt.sum() + eps)


# =====================
# Patient-level eval
# =====================
@torch.no_grad()
def evaluate_patient_level():
    # Load model
    model = AttentionCNNTransformerUNet().to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()

    # Dataset (slice-based, but grouped later)
    ds = BraTSDataset(DATA_ROOT)
    print(f"[INFO] Total slices: {len(ds)}")

    patient_preds = defaultdict(list)
    patient_gts = defaultdict(list)

    print("Collecting slices...")
    for i in tqdm(range(len(ds))):
        x, y, pid = ds[i]

        x = x.unsqueeze(0).to(DEVICE)   # [1, 4, H, W]
        y = y.to(DEVICE)                # [H, W]

        logits = model(x)
        pred = torch.argmax(logits, dim=1).squeeze(0)

        patient_preds[pid].append(pred)
        patient_gts[pid].append(y)

    # =====================
    # Compute patient Dice
    # =====================
    dice_scores = []

    for pid in patient_preds:
        pred_vol = torch.stack(patient_preds[pid])
        gt_vol = torch.stack(patient_gts[pid])

        d = dice_score(pred_vol, gt_vol)
        dice_scores.append(d.item())

    mean_dice = np.mean(dice_scores)
    std_dice = np.std(dice_scores)

    print("\n==============================")
    print(f"Patient-level Dice WT : {mean_dice:.4f}")
    print(f"STD Dice WT           : {std_dice:.4f}")
    print("==============================\n")


# =====================
# MAIN
# =====================
if __name__ == "__main__":
    evaluate_patient_level()
