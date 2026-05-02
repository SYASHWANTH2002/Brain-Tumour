import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BraTSDataset
from model import AttentionCNNTransformerUNet
from losses import DiceCELoss


def train_epoch(loader, model, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for x, y in tqdm(loader, leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def val_epoch(loader, model, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()

    return total_loss / len(loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    TRAIN_PATH = "data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"

    train_ds = BraTSDataset(TRAIN_PATH)

    train_loader = DataLoader(
        train_ds,
        batch_size=2,          # SAFE for 6GB VRAM
        shuffle=True,
        num_workers=0,         # IMPORTANT for Windows
        pin_memory=True
    )

    model = AttentionCNNTransformerUNet().to(device)
    criterion = DiceCELoss().to(device)
    print("Loss weights device:", criterion.weights.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    epochs = 30

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(train_loader, model, optimizer, criterion, device)
        print(f"Epoch [{epoch}/{epochs}] | Train Loss: {train_loss:.4f}")

    torch.save(model.state_dict(), "model_final.pth")
    print("✅ Model saved as model_final.pth")


if __name__ == "__main__":
    main()
