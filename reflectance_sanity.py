# reflectance_sanity.py

import torch
import numpy as np
from data_utils.parislille_dataset import ParisLilleDataset
from models.pointnet_sem_seg import get_model

# --- 1) load one block ---
ds = ParisLilleDataset(
    root_dir='/data/AI_Dev_Data/Paris-Lille-3D/training_10_classes/processed_blocks',
    split='train',
    split_ratio=1.0,
    num_point=4096,
    block_size=1.0
)
pts, lbl = ds[0]   # pts is either np.ndarray or torch.Tensor, shape [4096,4]
print("pts:", type(pts), "shape:", pts.shape)

# --- 2) ensure it's a torch.FloatTensor [4096,4] ---
if isinstance(pts, np.ndarray):
    pts = torch.from_numpy(pts)
pts = pts.float()  # now torch.Tensor

# --- 3) reshape to [1,4,4096] for PointNet ---
#  unsqueeze batch dim, then swap to (C,N)
x = pts.unsqueeze(0).transpose(2,1).cuda()  
print("input to model:", x.shape)  # should be [1,4,4096]

# --- 4) load your 4-channel PointNet model and forward ---
model = get_model(num_class=10).cuda()
model.eval()

with torch.no_grad():
    logits, _ = model(x)  
print("logits:", logits.shape)  # should be [1,10,4096]
