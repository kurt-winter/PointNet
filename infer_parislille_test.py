# infer_parislille_test.py
"""
Run inference on Paris-Lille-3D test blocks (no ground-truth).
For each block_{i}_xyz.npy under
  /data/AI_Dev_Data/Paris-Lille-3D/test_10_classes/processed_blocks/
this script will produce
  block_{i}_pred.npy
containing one integer [0..9] per point in the order they appear.

Usage:
    CUDA_VISIBLE_DEVICES=0 python infer_parislille_test.py \
        --model pointnet_sem_seg \
        --log_dir baseline_pointnet \
        --batch_size 16 \
        --npoint 4096
"""

import os
import argparse
import torch
import importlib
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- Dataset that only loads XYZ (no labels) -----------------------------------
class ParisLilleInferDataset(Dataset):
    def __init__(self, root_dir, num_point=4096):
        self.root_dir = root_dir
        self.num_point = num_point
        # find all *_xyz.npy files
        self.data_paths = sorted([p for p in os.listdir(root_dir) if p.endswith("_xyz.npy")])

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        fname = self.data_paths[idx]
        xyz = np.load(os.path.join(self.root_dir, fname)).astype(np.float32)  # [N,3]
        # if the block size isn't exactly num_point, replicate or subsample:
        N = xyz.shape[0]
        if N >= self.num_point:
            choice = np.random.choice(N, self.num_point, replace=False)
        else:
            choice = np.random.choice(N, self.num_point, replace=True)
        sampled = xyz[choice, :]  # [num_point, 3]
        return sampled, fname, choice  # we return `choice` so we can map back to original order

# --- Argument parsing ----------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser("Infer Paris-Lille-3D Test Blocks (no GT)")
    parser.add_argument('--model',      type=str, required=True,
                        help='Model module name: pointnet_sem_seg or pointnet2_sem_seg')
    parser.add_argument('--log_dir',    type=str, required=True,
                        help='Experiment folder (under log/sem_seg/) containing best_model.pth')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size [default:16]')
    parser.add_argument('--npoint',     type=int, default=4096, help='Points per block [default:4096]')
    parser.add_argument('--gpu',        type=str, default='0', help='Which GPU to use [default:0]')
    return parser.parse_args()

# --- Main Inference Routine ----------------------------------------------------
def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # locate checkpoint:
    checkpoint_path = os.path.join('log/sem_seg', args.log_dir, 'checkpoints', 'best_model.pth')
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # build model:
    NUM_CLASSES = 10
    MODEL = importlib.import_module(args.model)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    ckpt = torch.load(checkpoint_path)
    classifier.load_state_dict(ckpt['model_state_dict'])
    classifier.eval()

    # prepare dataset & dataloader:
    test_root = '/data/AI_Dev_Data/Paris-Lille-3D/test_10_classes/processed_blocks'
    dataset = ParisLilleInferDataset(root_dir=test_root, num_point=args.npoint)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=4,
                        pin_memory=True,
                        drop_last=False)

    # create folder to hold predictions:
    pred_dir = os.path.join(test_root, 'predictions')
    os.makedirs(pred_dir, exist_ok=True)

    # Inference loop:
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader)):
            points_batch, fnames, choices_batch = batch
            # points_batch: [B, num_point, 3]
            B = points_batch.shape[0]
            points = points_batch.permute(0,2,1).contiguous().cuda()  # [B,3,N]
            # forward:
            seg_pred, _ = classifier(points)  # [B, num_classes, N]
            seg_pred = seg_pred.permute(0,2,1).contiguous()  # [B, N, num_classes]
            pred_labels = seg_pred.argmax(dim=2).cpu().numpy()  # [B, N]

            # For each block in the batch, un-permute back to original ordering:
            for b in range(B):
                fname = fnames[b]                     # e.g. "block_123_xyz.npy"
                choice = choices_batch[b].numpy()     # [num_point] indices into original
                original_N = int(fname.split('_xyz.npy')[0].split('_')[-1])  # not always needed
                # We need to reconstruct an array of length = original number of points in block.
                # But since each block_{i}_xyz.npy always has exactly num_point points,
                # we can just save pred_labels[b] as-is, paired with the original ordering.
                # In other words: block_i was already downsampled to exactly num_point,
                # so just save one `.npy` of length num_point with the labels.

                # Compute output filename:
                out_fname = fname.replace('_xyz.npy', '_pred.npy')
                out_fpath = os.path.join(pred_dir, out_fname)
                np.save(out_fpath, pred_labels[b])

    print("? Inference complete. Predictions saved in:", pred_dir)

if __name__ == '__main__':
    args = parse_args()
    main(args)
