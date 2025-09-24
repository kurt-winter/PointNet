import numpy as np
import os
import torch
from torch.utils.data import Dataset
import glob

class ParisLilleDataset(Dataset):
    def __init__(self, root_dir, split='train', split_ratio=0.8, num_point=4096, block_size=1.0):
        self.root_dir = root_dir
        self.num_point = num_point
        self.block_size = block_size
        self.split = split
        self.split_ratio = split_ratio

        # === Gather all blocks before splitting ===
        all_data_paths = sorted(glob.glob(os.path.join(root_dir, "*_xyz.npy")))  # CHANGED
        all_label_paths = [p.replace('_xyz.npy', '_labels.npy') for p in all_data_paths]  # CHANGED

        # === Split into train/test ===
        n_total = len(all_data_paths)  # CHANGED
        n_train = int(n_total * split_ratio)  # CHANGED
        if split == 'train':  # CHANGED
            self.data_paths  = all_data_paths[:n_train]  # CHANGED
            self.label_paths = all_label_paths[:n_train]  # CHANGED
        else:  # CHANGED
            self.data_paths  = all_data_paths[n_train:]  # CHANGED
            self.label_paths = all_label_paths[n_train:]  # CHANGED

        # === Compute per-block sampling weights (oversampling rare classes) ===
        #rare_labels = {2,3,4,5,6,7,8,9}  # CHANGED: all except building(0), pole(1)
        #block_weights = []  # CHANGED
        #for lbl_path in self.label_paths:  # CHANGED
        #    lbls = np.load(lbl_path)  # CHANGED
        #    # if block contains ANY rare label, boost sampling
        #    block_weights.append(5.0 if np.isin(lbls, list(rare_labels)).any() else 1.0)  # CHANGED
        #self.block_weights = torch.DoubleTensor(block_weights)  # CHANGED

        # === Original label-weight computation ===
        label_hist = np.zeros(10)
        for label_file in self.label_paths:
            labels = np.load(label_file)
            bincount = np.bincount(labels, minlength=10)
            label_hist += bincount

        label_prob = label_hist / np.sum(label_hist)
        # cube-root reweighting (baseline)
        self.labelweights = np.power(np.amax(label_prob) / (label_prob + 1e-6), 1/3.0)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        xyz_file = self.data_paths[idx]
        label_file = self.label_paths[idx]

        points = np.load(xyz_file).astype(np.float32)
        labels = np.load(label_file).astype(np.int64)

        return torch.from_numpy(points), torch.from_numpy(labels)
