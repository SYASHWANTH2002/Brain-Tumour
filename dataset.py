import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset


class BraTSDataset(Dataset):
    """
    BraTS 2020 Slice Dataset
    - Loads ONLY tumor-containing slices
    - Returns patient_id (REQUIRED for patient-level Dice)
    - Whole Tumor (WT): label > 0
    """

    def __init__(self, root_dir, max_slices=80):
        """
        Args:
            root_dir (str):
                data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData
            max_slices (int):
                Max slices per patient (to control size)
        """
        self.root_dir = root_dir
        self.samples = []

        patients = sorted(os.listdir(root_dir))

        for patient in patients:
            pdir = os.path.join(root_dir, patient)
            if not os.path.isdir(pdir):
                continue

            files = {
                "flair": None,
                "t1": None,
                "t1ce": None,
                "t2": None,
                "seg": None
            }

            for f in os.listdir(pdir):
                if f.endswith("_flair.nii"):
                    files["flair"] = os.path.join(pdir, f)
                elif f.endswith("_t1.nii"):
                    files["t1"] = os.path.join(pdir, f)
                elif f.endswith("_t1ce.nii"):
                    files["t1ce"] = os.path.join(pdir, f)
                elif f.endswith("_t2.nii"):
                    files["t2"] = os.path.join(pdir, f)
                elif f.endswith("_seg.nii"):
                    files["seg"] = os.path.join(pdir, f)

            # skip incomplete patients
            if None in files.values():
                continue

            seg_vol = nib.load(files["seg"]).get_fdata()
            depth = seg_vol.shape[2]

            for z in range(min(depth, max_slices)):
                if np.sum(seg_vol[:, :, z]) > 0:
                    # store patient_id for patient-level evaluation
                    self.samples.append((files, z, patient))

        print(f"[INFO] Loaded {len(self.samples)} tumor slices")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        files, z, patient_id = self.samples[idx]

        def load_slice(path):
            vol = nib.load(path).get_fdata()
            slice_ = vol[:, :, z]
            slice_ = np.nan_to_num(slice_)
            slice_ = (slice_ - slice_.mean()) / (slice_.std() + 1e-5)
            return slice_

        flair = load_slice(files["flair"])
        t1    = load_slice(files["t1"])
        t1ce  = load_slice(files["t1ce"])
        t2    = load_slice(files["t2"])

        x = np.stack([flair, t1, t1ce, t2], axis=0)
        x = torch.tensor(x, dtype=torch.float32)

        seg = nib.load(files["seg"]).get_fdata()[:, :, z]
        seg = torch.tensor(seg, dtype=torch.long)

        # Whole Tumor (WT): labels > 0
        seg = (seg > 0).long()

        return x, seg, patient_id
