import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCELoss(nn.Module):
    def __init__(self):
        super().__init__()

        # Class weights (background, tumor)
        weights = torch.tensor([0.1, 0.9], dtype=torch.float32)
        self.register_buffer("weights", weights)

        self.ce = nn.CrossEntropyLoss(weight=self.weights)

    def dice_loss(self, logits, targets, eps=1e-6):
        probs = torch.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, num_classes=2).permute(0, 3, 1, 2)

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_oh, dims)
        union = torch.sum(probs + targets_oh, dims)

        dice = (2. * intersection + eps) / (union + eps)
        return 1 - dice.mean()

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice_loss(logits, targets)
        return ce_loss + dice_loss
