import os
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import AttentionCNNTransformerUNet

# -----------------------
# CONFIG
# -----------------------
DATA_ROOT = "data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
CKPT = "model_final.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MC_PASSES = 10


# -----------------------
# Dice (Binary WT)
# -----------------------
def dice_binary(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt   = (gt > 0).astype(np.uint8)

    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)

    if union == 0:
        return 1.0

    return (2.0 * intersection) / union


# -----------------------
# Dataset (with patient + slice id)
# -----------------------
class BraTSPatientDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        patients = sorted(os.listdir(root_dir))
        for pid in patients:
            pdir = os.path.join(root_dir, pid)
            if not os.path.isdir(pdir):
                continue

            files = {}
            for f in os.listdir(pdir):
                if f.endswith(".nii"):
                    files[f.split("_")[-1].replace(".nii", "")] = os.path.join(pdir, f)

            if "seg" not in files:
                continue

            seg = nib.load(files["seg"]).get_fdata()
            for z in range(seg.shape[2]):
                if np.sum(seg[:, :, z]) > 0:
                    self.samples.append((pid, files, z))

        print(f"[INFO] Loaded {len(self.samples)} tumor slices")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, files, z = self.samples[idx]

        def load_slice(k):
            vol = nib.load(files[k]).get_fdata()
            s = vol[:, :, z]
            s = (s - s.mean()) / (s.std() + 1e-5)
            return s

        x = np.stack([
            load_slice("flair"),
            load_slice("t1"),
            load_slice("t1ce"),
            load_slice("t2")
        ], axis=0)

        seg = nib.load(files["seg"]).get_fdata()[:, :, z]
        seg = (seg > 0).astype(np.uint8)

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(seg, dtype=torch.long),
            pid,
            z
        )


# -----------------------
# Enable MC Dropout
# -----------------------
def enable_mc_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


# -----------------------
# MAIN EVAL
# -----------------------
@torch.no_grad()
def evaluate_patient_uncertainty():
    print("Using device:", DEVICE)

    model = AttentionCNNTransformerUNet().to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()
    enable_mc_dropout(model)

    ds = BraTSPatientDataset(DATA_ROOT)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    patient_preds = {}
    patient_gts = {}
    patient_uncert = {}

    print("[INFO] Total slices:", len(ds))
    print("Collecting slices...")

    for x, y, pid, _ in tqdm(loader):
        x = x.to(DEVICE)

        probs_mc = []
        for _ in range(MC_PASSES):
            logits = model(x)
            probs = F.softmax(logits, dim=1)[:, 1]  # WT prob
            probs_mc.append(probs.cpu().numpy())

        probs_mc = np.stack(probs_mc, axis=0)
        mean_prob = probs_mc.mean(axis=0)[0]
        entropy = -np.mean(
            mean_prob * np.log(mean_prob + 1e-8) +
            (1 - mean_prob) * np.log(1 - mean_prob + 1e-8)
        )

        pred = (mean_prob > 0.5).astype(np.uint8)
        gt = y.numpy()[0]

        pid = pid[0]
        patient_preds.setdefault(pid, []).append(pred)
        patient_gts.setdefault(pid, []).append(gt)
        patient_uncert.setdefault(pid, []).append(entropy)

    # -----------------------
    # Patient-level Dice
    # -----------------------
    dice_scores = []
    uncert_scores = []

    for pid in patient_preds:
        pred_vol = np.stack(patient_preds[pid])
        gt_vol = np.stack(patient_gts[pid])

        d = dice_binary(pred_vol, gt_vol)
        u = np.mean(patient_uncert[pid])

        dice_scores.append(d)
        uncert_scores.append(u)

    print("\n==============================")
    print(f"Uncertainty-Aware Patient Dice WT : {np.mean(dice_scores):.4f}")
    print(f"Dice STD (patients)              : {np.std(dice_scores):.4f}")
    print(f"Prediction Uncertainty (avg)     : {np.mean(uncert_scores):.6f}")
    print("==============================\n")


# -----------------------
if __name__ == "__main__":
    evaluate_patient_uncertainty()
