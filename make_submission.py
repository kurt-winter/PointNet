#!/usr/bin/env python3
import os, sys, glob, zipfile, argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# make sure this points at your models/ folder
sys.path.append(os.path.join(os.path.dirname(__file__), "models"))
from pointnet_sem_seg import get_model

def load_model(checkpoint_path, device="cuda:0"):
    model = get_model(10).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

def predict_cloud(model, device, block_dir, num_classes):
    """
    block_dir e.g. /�/processed_blocks/ajaccio_2/processed_blocks
    """
    # grab your existing npy blocks
    xyz_files = sorted(glob.glob(os.path.join(block_dir, "*_block_*_xyz.npy")))
    inds_files = [f.replace("_xyz.npy","_inds.npy") for f in xyz_files]
    if not xyz_files:
        raise RuntimeError(f"No blocks found in {block_dir}")

    TOTAL = 10_000_000
    all_probs = np.zeros((TOTAL, num_classes), dtype=np.float32)

    for xyz_f, ind_f in tqdm(zip(xyz_files, inds_files),
                             total=len(xyz_files),
                             desc="    blocks", leave=False):
        pts  = np.load(xyz_f).astype(np.float32)    # (B,3+feat)
        idxs = np.load(ind_f).astype(np.int64)      # (B,)

        # forward
        x = torch.from_numpy(pts).float().to(device).unsqueeze(0).transpose(2,1)
        with torch.no_grad():
            logits, _ = model(x)                    # [1,N,C]
            probs = F.softmax(logits, dim=2)[0].cpu().numpy()  # (N,C)

        # accumulate
        all_probs[idxs] += probs

    # final argmax
    preds = all_probs.argmax(axis=1)

    # find any leftover zeros and re-assign to 2nd best
    zeros = np.where(preds == 0)[0]
    if len(zeros):
        # for each zero-row, argsort gives ascending scores ? [-2] is 2nd best
        ordering = np.argsort(all_probs[zeros], axis=1)
        second = ordering[:, -2]
        preds[zeros] = second

    return preds

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir",    required=True,
                   help="where best_model.pth lives")
    p.add_argument("--processed_root", required=True,
                   help="root of test clouds, each subfolder has processed_blocks/")
    p.add_argument("--output_zip",   default="submission.zip")
    args = p.parse_args()

    device = "cuda:0"
    ckpt = os.path.join(args.model_dir, "best_model.pth")
    model = load_model(ckpt, device)

    out_txts = []
    for cloud in sorted(os.listdir(args.processed_root)):
        block_dir = os.path.join(args.processed_root, cloud, "processed_blocks")
        if not os.path.isdir(block_dir):
            continue
        print(f"=== Processing {cloud} ===")
        preds = predict_cloud(model, device, block_dir, num_classes=10)

        # write exactly 10M lines, one label per line
        fn = f"{cloud}.txt"
        np.savetxt(fn, preds.astype(np.int32), fmt="%d")
        out_txts.append(fn)

    # bundle them up
    with zipfile.ZipFile(args.output_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        for fn in out_txts:
            zf.write(fn)
    print("Wrote", args.output_zip)

if __name__ == "__main__":
    main()
