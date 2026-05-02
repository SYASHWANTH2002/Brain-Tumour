import torch
from losses import DiceCELoss

criterion = DiceCELoss()

logits = torch.randn(2, 2, 240, 240)
targets = torch.randint(0, 2, (2, 240, 240))

loss = criterion(logits, targets)
print("Loss:", loss.item())
