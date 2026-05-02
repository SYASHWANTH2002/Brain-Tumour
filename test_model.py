import torch
from model import AttentionCNNTransformerUNet

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

model = AttentionCNNTransformerUNet().to(device)

x = torch.randn(1, 4, 240, 240).to(device)
y = model(x)

print("Output shape:", y.shape)
