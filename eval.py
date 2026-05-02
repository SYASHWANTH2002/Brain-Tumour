import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BraTSDataset
from model import AttentionCNNTransformerUNet


# ---------------------------
# Dice function (binary WT)
# ---------------------------
def dice_binary(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt   = (gt > 0).astype(np.uint8)

    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)

    if union == 0:
        return 1.0

    return (2.0 * intersection) / union


# ---------------------------
# Evaluation
# ---------------------------
@torch.no_grad()
def evaluate():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # -------- paths --------
    DATA_ROOT = "data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    CKPT = "model_final.pth"

    # -------- model --------
    model = AttentionCNNTransformerUNet(
        in_channels=4,
        num_classes=2
    ).to(DEVICE)

    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()

    # -------- dataset --------
    dataset = BraTSDataset(DATA_ROOT)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    print(f"[INFO] Total slices: {len(dataset)}")

    dice_scores = []

    # -------- loop --------
    for batch in tqdm(loader, desc="Evaluating"):
        x = batch[0].to(DEVICE)   # image
        y = batch[1].to(DEVICE)   # mask

        logits = model(x)
        pred = torch.argmax(logits, dim=1)

        pred_np = pred.squeeze().cpu().numpy()
        gt_np   = y.squeeze().cpu().numpy()

        d = dice_binary(pred_np, gt_np)
        dice_scores.append(d)

    dice_scores = np.array(dice_scores)

    print("\n==============================")
    print(f"FINAL Dice WT : {dice_scores.mean():.4f}")
    print(f"STD Dice WT   : {dice_scores.std():.4f}")
    print("==============================\n")


if __name__ == "__main__":
    evaluate()
