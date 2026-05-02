from dataset import BraTSDataset

ds = BraTSDataset(
    "data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
)

x, y, pid = ds[0]
print("Input:", x.shape)
print("Mask:", y.shape)
print("Labels:", y.unique())
print("Patient:", pid)
