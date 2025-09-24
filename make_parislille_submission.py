#!/usr/bin/env python3
"""
make_parislille_submission.py

Given:
  - A trained segmentation model checkpoint (PointNet or PointNet++).
  - A �processed_blocks/� directory under each test-cloud folder, containing:
       * block_{i}_xyz.npy      (4096�3 float32) � the block�s point coordinates
       * block_{i}_inds.npy     (4096 int64)       � the original global indices [0..N-1]
  - CUDA-capable GPU (or CPU fallback).

Produces:
  - dijon_9.txt        (10 000 000 lines, each line: integer in [0..9] for that point)
  - ajaccio_2.txt
  - ajaccio_57.txt

Usage:
  cd Pointnet_Pointnet2_pytorch
  python make_parislille_submission.py \
        --model pointnet2_sem_seg \
        --checkpoint log/sem_seg/baseline_pointnetpp/checkpoints/best_model.pth \
        --batch_size 16 \
        --num_point 4096 \
        --test_root /data/AI_Dev_Data/Paris-Lille-3D/test_10_classes \
        --out_dir submission_txt
"""

import os
import argparse
import glob
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Make sure Python can import your model and utils
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'data_utils'))

# Replace with whichever dataset class you used (if needed)
from data_utils.parislille_dataset import ParisLilleDataset

def parse_args():
    parser = argparse.ArgumentParser("Generate Paris-Lille-3D test-set submission files")
    parser.add_argument('--model', type=str, required=True,
                        help="Name of the model file (without .py), e.g. 'pointnet_sem_seg' or 'pointnet2_sem_seg'")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Path to your trained checkpoint (best_model.pth or model.pth)")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Batch size for inference on each block (default: 16)")
    parser.add_argument('--num_point', type=int, default=4096,
                        help="Number of points per block (must match preprocessing) (default: 4096)")
    parser.add_argument('--test_root', type=str, required=True,
                        help="Root directory of test_10_classes, e.g. /data/AI_Dev_Data/Paris-Lille-3D/test_10_classes")
    parser.add_argument('--out_dir', type=str, default='submission_txt',
                        help="Where to write dijon_9.txt, ajaccio_2.txt, ajaccio_57.txt (default: submission_txt)")
    parser.add_argument('--device', type=str, default='cuda:0',
                        help="PyTorch device to run inference on. If using CUDA_VISIBLE_DEVICES, keep this 'cuda:0'")
    return parser.parse_args()

def load_model(model_name, checkpoint_path, num_classes, device):
    """
    Dynamically import and load the model, then load its weights.
    Returns a model in eval() mode, on the specified device.
    """
    module = __import__(model_name)  # this expects e.g. models/pointnet2_sem_seg.py to define get_model, get_loss
    ModelClass = getattr(module, 'get_model')
    model = ModelClass(num_classes).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

def infer_on_blocks_for_cloud(test_cloud_name, args, model):
    """
    For a single test cloud (e.g. 'dijon_9'), read all block_*.npy under
      {args.test_root}/{test_cloud_name}/processed_blocks/
    and run inference. Accumulate votes for each original point:
      - block_i_xyz.npy   ? [4096, 3]  (but we don�t actually need XYZ here)
      - block_i_inds.npy  ? [4096]     (global indices ? [0..N-1])
    We'll produce a final `pred_labels.npy` of shape [N], with one integer ? [0..9].
    """
    dev = torch.device(args.device)
    pb_root = os.path.join(args.test_root, test_cloud_name, 'processed_blocks')
    if not os.path.isdir(pb_root):
        raise RuntimeError(f"Processed-blocks folder not found:\n  {pb_root}")

    # Find all blocks (sorted by i). Each block has:
    #    block_{i}_xyz.npy     and block_{i}_inds.npy
    xyz_files = sorted(glob.glob(os.path.join(pb_root, 'block_*_xyz.npy')))
    if len(xyz_files) == 0:
        raise RuntimeError(f"No block_*_xyz.npy found under {pb_root}")

    # We assume each xyz file has a matching inds file:
    #   block_123_xyz.npy  ? block_123_inds.npy
    def get_inds_path(xyz_path):
        return xyz_path.replace('_xyz.npy', '_inds.npy')

    # First pass: figure out how many total points in this cloud
    # We can load the largest index from all inds to know N = max_index+1
    max_idx = -1
    for xyz_path in xyz_files:
        inds_path = get_inds_path(xyz_path)
        if not os.path.isfile(inds_path):
            raise RuntimeError(f"Missing index file for block:\n  {inds_path}")
        inds = np.load(inds_path)
        local_max = inds.max()
        if local_max > max_idx:
            max_idx = local_max
    N = int(max_idx) + 1
    print(f"\n[{test_cloud_name}] Total points (deduced from indices): {N:,}")

    # Prepare a vote-matrix: [N, 10], initially zeros (float32)
    vote_matrix = np.zeros((N, 10), dtype=np.int32)

    # Now run inference block-by-block
    with torch.no_grad():
        for xyz_path in tqdm(xyz_files, desc=f"Inference blocks for {test_cloud_name}"):
            inds_path = get_inds_path(xyz_path)
            block_inds = np.load(inds_path).astype(np.int64)   # shape [4096]
            block_xyz  = np.load(xyz_path).astype(np.float32)  # [4096, 3]

            # Build a tensor of shape [1, num_point, 3], move to GPU
            pts = torch.from_numpy(block_xyz).unsqueeze(0).to(dev)  # [1, 4096, 3]
            # We need to reshape to [1, 3, 4096] and (optionally) rotate
            pts = pts.transpose(2, 1)  # ? [1, 3, 4096]

            # Forward pass: we only need logits ? [1, 10, 4096]
            logits, _ = model(pts)  # logits: [1, 4096, 10]
            # (some PointNet variants return [B, num_classes, N], but ours returns [B, N, num_classes])
            if logits.shape[-1] == 10 and logits.ndim == 3:
                # They used [B, N, C] convention
                logits = logits  # nothing to reorder
            else:
                # If the model returned [B, C, N], then transpose:
                logits = logits.transpose(2, 1)  # [1, N, C]

            # Now get predicted labels per point: argmax over last dim
            preds = logits.argmax(dim=2).squeeze(0).cpu().numpy().astype(np.int32)  # [4096]

            # �Vote� into vote_matrix:
            #   For each j in [0..4095], global_idx = block_inds[j], predicted_label = preds[j]:
            #   increment vote_matrix[global_idx, predicted_label] += 1
            vote_matrix[block_inds, preds] += 1

    # After processing all blocks, take argmax over axis=1 to get a final label per point
    final_labels = vote_matrix.argmax(axis=1).astype(np.int32)  # shape [N]

    return final_labels


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    # Load your model (10 coarse classes)
    model = load_model(args.model, args.checkpoint, num_classes=10, device=device)
    print(f"Loaded {args.model} checkpoint ? will run on {device}\n")

    # The three test-cloud names correspond to folder names under test_root:
    TEST_NAMES = ['dijon_9', 'ajaccio_2', 'ajaccio_57']

    for cloud in TEST_NAMES:
        print(f"=== Processing {cloud} ===")
        labels = infer_on_blocks_for_cloud(cloud, args, model)
        txt_path = os.path.join(args.out_dir, f"{cloud}.txt")
        print(f"Writing {labels.shape[0]} labels to {txt_path} �")
        # Write one label per line (as an integer 0..9). Submission expects labels ? [1..9],
        # but according to the README, �0=unclassified� is ignored�so we still write �0� for unclassified.
        # If you prefer to map to 1..9, do: labels += 1  (but instructions said 1..9 ignoring unclassified).
        with open(txt_path, 'w') as f:
            for lbl in labels:
                f.write(f"{int(lbl)}\n")

    print("\nAll done! Your submission files are in:", args.out_dir)
    print("? Zip that folder (e.g. `zip -r submission.zip submission_txt/`) and upload.")


if __name__ == '__main__':
    main()
