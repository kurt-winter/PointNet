import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class ParisLilleDataset(Dataset):
    """
    Dataset for Paris-Lille 3D segmentation.

    Parameters
    ----------
    root_dir : str
        Directory containing preprocessed blocks named *_xyz.npy and *_labels.npy.
    file_list : list of str, optional
        List of block basenames (without suffix) to include. If None, include all blocks.
    num_point : int, default=4096
        Number of points to sample per block (with replacement if fewer points exist).
    normalize : bool, default=True
        Whether to apply global min-max normalization to XYZ coordinates.
    """

    def __init__(self, root_dir, file_list=None, num_point=4096, normalize=True, xyz_min=None, xyz_max=None, return_name=False):
        self.root_dir = root_dir
        self.num_point = num_point
        self.normalize = normalize

        # Gather all block basenames if not explicitly provided
        all_xyz = sorted(glob.glob(os.path.join(root_dir, "*_xyz.npy")))
        basenames = [os.path.basename(p).replace("_xyz.npy", "") for p in all_xyz]
        self.basenames = file_list or basenames

        # Build paths to XYZ and label files
        self.data_paths = [os.path.join(root_dir, b + "_xyz.npy") for b in self.basenames]
        self.label_paths = [os.path.join(root_dir, b + "_labels.npy") for b in self.basenames]

        # Global min/max: either use passed-in or compute now
        if self.normalize:
            if xyz_min is not None and xyz_max is not None:
                # reuse externally computed bounds
                self.xyz_min = xyz_min
                self.xyz_max = xyz_max
            else:
                # compute once over this file_list
                mins = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
                maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)
                for p in tqdm(self.data_paths, desc="Computing global XYZ ranges"):
                    pts = np.load(p).astype(np.float32)[:, :3]
                    mins = np.minimum(mins, pts.min(axis=0))
                    maxs = np.maximum(maxs, pts.max(axis=0))
                self.xyz_min = mins
                self.xyz_max = maxs

        # Compute class weights from the labels of all selected files
        label_hist = np.zeros(10, dtype=np.float64)
        for lp in tqdm(self.label_paths, desc="Computing label histogram"):
            lbls = np.load(lp)
            label_hist += np.bincount(lbls, minlength=10)
        label_prob = label_hist / label_hist.sum()
        lw = np.power(label_prob.max() / (label_prob + 1e-6), 1/3.0).astype(np.float32)
        self.labelweights = torch.from_numpy(lw)

        self.return_name = return_name

    def __len__(self):
        return len(self.basenames)

    def __getitem__(self, idx):
        # Load XYZ and labels
        pts = np.load(self.data_paths[idx]).astype(np.float32)[:, :3]
        lbl = np.load(self.label_paths[idx]).astype(np.int64)

        # Apply global min-max normalization
        if self.normalize:
            pts = (pts - self.xyz_min) / (self.xyz_max - self.xyz_min + 1e-6)

        # Randomly sample or pad points to fixed size
        N = pts.shape[0]
        if N >= self.num_point:
            choice = np.random.choice(N, self.num_point, replace=False)
        else:
            choice = np.random.choice(N, self.num_point, replace=True)
        pts = pts[choice, :]
        lbl = lbl[choice]

        base = self.basenames[idx]  # e.g., 'Lille1_1_block_00000'
        if self.return_name:
            return torch.from_numpy(pts), torch.from_numpy(lbl), base
        else:
            return torch.from_numpy(pts), torch.from_numpy(lbl)

        # added return_name
        # Convert to torch tensors
        #return torch.from_numpy(pts), torch.from_numpy(lbl)
