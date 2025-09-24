# check_processed_blocks.py
import os, glob
import numpy as np
from collections import Counter

def scan_blocks(root_dir):
    label_paths = sorted(glob.glob(os.path.join(root_dir, "*_labels.npy")))
    global_counts = Counter()
    for p in label_paths:
        labs = np.load(p)
        unique = np.unique(labs)
        counts = Counter(labs.tolist())
        print(f"{os.path.basename(p)}: unique labels = {unique.tolist()}")
        global_counts.update(counts)
    print("\n=== Global label counts ===")
    for cls in range(10):
        print(f"  class {cls:2d}: {global_counts[cls]} points")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("blocks_dir", help="path to your processed_blocks directory")
    args = parser.parse_args()
    scan_blocks(args.blocks_dir)
