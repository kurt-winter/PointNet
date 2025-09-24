import os
import numpy as np
from plyfile import PlyData
from tqdm import tqdm
from scipy.spatial import cKDTree

# uses cKDTree query instead of NumPy masking

# ---------------- CONFIG ----------------
#DATA_DIR = "/data/AI_Dev_Data/Paris-Lille-3D/training_10_classes"
DATA_DIR = "/data/AI_Dev_Data/Paris-Lille-3D/test_10_classes"
OUTPUT_SUBDIR = "processed_blocks"
BLOCK_SIZE = 1.0
STRIDE = 0.5
NUM_POINTS = 4096
MIN_POINTS_PER_BLOCK = 100
#FILES_TO_PROCESS = ["Lille1_1.ply", "Lille1_2.ply"]
#FILES_TO_PROCESS = ["ajaccio_2.ply", "ajaccio_57.ply", "dijon_9"]
FILES_TO_PROCESS = ["ajaccio_57.ply"]
# ----------------------------------------

OUTPUT_DIR = os.path.join(DATA_DIR, OUTPUT_SUBDIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_ply(filepath):
    plydata = PlyData.read(filepath)
    data = plydata['vertex'].data
    xyz = np.stack([data['x'], data['y'], data['z']], axis=1)
    labels = data['class'] if 'class' in data.dtype.names else None
    return xyz, labels

def split_into_blocks_fast(xyz, labels=None):
    blocks = []
    coords = xyz[:, :2]
    x_min, y_min = np.min(coords, axis=0)
    x_max, y_max = np.max(coords, axis=0)

    grid_x = np.arange(x_min, x_max, STRIDE)
    grid_y = np.arange(y_min, y_max, STRIDE)

    tree = cKDTree(coords)

    for gx in tqdm(grid_x, desc="Grid X", leave=False):
        for gy in grid_y:
            center = np.array([[gx + BLOCK_SIZE / 2, gy + BLOCK_SIZE / 2]])
            idx = tree.query_ball_point(center, BLOCK_SIZE * np.sqrt(2) / 2)
            if not idx or len(idx[0]) < MIN_POINTS_PER_BLOCK:
                continue
            idx = idx[0]
            block_pts = xyz[idx]
            block_lbls = labels[idx] if labels is not None else None
            blocks.append((block_pts, block_lbls))
    return blocks

def sample_block(points, labels, num_points):
    N = points.shape[0]
    idx = np.random.choice(N, num_points, replace=(N < num_points))
    return points[idx], labels[idx] if labels is not None else None

def preprocess():
    block_idx = 0
    for fname in FILES_TO_PROCESS:
        fpath = os.path.join(DATA_DIR, fname)
        print(f"\n?? Loading: {fname}")
        xyz, labels = load_ply(fpath)
        print(f"   ? Loaded {xyz.shape[0]:,} points")

        blocks = split_into_blocks_fast(xyz, labels)
        print(f"   ? {len(blocks)} valid blocks")

        for pts, lbls in tqdm(blocks, desc="Saving blocks", leave=False):
            pts_centered = pts.copy()
            pts_centered[:, 0] -= np.mean(pts[:, 0])
            pts_centered[:, 1] -= np.mean(pts[:, 1])
            sampled_pts, sampled_lbls = sample_block(pts_centered, lbls, NUM_POINTS)

            np.save(os.path.join(OUTPUT_DIR, f"block_{block_idx}_xyz.npy"), sampled_pts)
            if sampled_lbls is not None:
                np.save(os.path.join(OUTPUT_DIR, f"block_{block_idx}_labels.npy"), sampled_lbls)
            block_idx += 1

        print(f"? Done with {fname}. Total saved so far: {block_idx} blocks")

if __name__ == "__main__":
    preprocess()
