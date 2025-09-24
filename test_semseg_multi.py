import os
import sys
import glob
import argparse
import importlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

# Dataset
from data_utils.parislille_dataset import ParisLilleDataset

# BASE & PYTHON PATH SETUP
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))

NUM_CLASSES = 10
seg_label_to_cat = {
    0: "unclassified",
    1: "ground",
    2: "building",
    3: "pole - road sign - traffic light",
    4: "bollard - small pole",
    5: "trash can",
    6: "barrier",
    7: "pedestrian",
    8: "car",
    9: "natural - vegetation"
}

def parse_args():
    parser = argparse.ArgumentParser('PointNet Semantic Segmentation Testing (multi-fold ensemble)')
    parser.add_argument('--model', type=str, default='pointnet_sem_seg')
    # EITHER pass multiple checkpoints explicitly...
    parser.add_argument('--checkpoints', type=str, nargs='+', default=None,
                        help='List of checkpoint paths (e.g. fold1..fold5).')
    # ...OR point to a directory and pattern to glob
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Directory containing fold checkpoints.')
    parser.add_argument('--pattern', type=str, default='best_model_fold_*.pth',
                        help='Glob pattern inside --ckpt_dir.')
    parser.add_argument('--ensemble', type=str, default='avg', choices=['avg','vote'],
                        help='Ensemble method: average probabilities or majority vote.')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--npoint', type=int, default=4096)
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(BASE_DIR, 'data', 'processed_blocks'),
                        help='Directory with processed blocks')
    parser.add_argument('--xyz_range', type=str, default=None,
                        help='Optional path to xyz_range.npz; if omitted we infer from first ckpt.')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Where to save plots; defaults to first checkpoint dir.')
    return parser.parse_args()

def plot_test_metrics(conf_mat, per_class_iou, seg_label_to_cat, save_dir=None, title_suffix=''):
    C = len(per_class_iou)
    classes = [seg_label_to_cat[i] for i in range(C)]
    total_per_class = conf_mat.sum(axis=1)

    ylabels = [f"{classes[i]} ({total_per_class[i]:,})" for i in range(C)]
    xs = np.arange(C)

    # 1) Per-class IoU bar chart
    plt.figure(figsize=(10,5))
    plt.bar(xs, per_class_iou)
    plt.xticks(xs, classes, rotation=45, ha='right', fontsize=9)
    plt.ylabel('IoU', fontsize=12)
    plt.title(f'Per-Class IoU on Lille2{title_suffix}', fontsize=14)
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'iou_per_class.png'), dpi=200)
    else:
        plt.show()

    # 2) Confusion matrix (counts) with annotations + highlighted diagonal
    plt.figure(figsize=(12,10))
    cmap = plt.get_cmap('viridis')
    norm = Normalize(vmin=0, vmax=conf_mat.max() if conf_mat.max()>0 else 1)
    im = plt.imshow(conf_mat, norm=norm, cmap=cmap)
    cb = plt.colorbar(im, label='Count')
    cb.ax.tick_params(labelsize=10)

    plt.xticks(xs, classes, rotation=45, ha='right', fontsize=9)
    plt.yticks(xs, ylabels, fontsize=9)
    plt.xlabel('Predicted label', fontsize=12)
    plt.ylabel('True label (count)', fontsize=12)
    plt.title(f'Confusion Matrix on Lille2{title_suffix}', fontsize=14)

    ax = plt.gca()
    for i in range(C):
        for j in range(C):
            cnt = conf_mat[i, j]
            if cnt == 0:
                continue
            rgba = cmap(norm(cnt))
            lum = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
            txt_color = 'black' if lum > 0.5 else 'white'
            plt.text(j, i, f"{cnt:,}", ha='center', va='center', color=txt_color, fontsize=8)
        rect = Rectangle((i-0.5, i-0.5), 1, 1, linewidth=2, edgecolor='white', facecolor='none')
        ax.add_patch(rect)

    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'confusion_matrix_annotated.png'), dpi=200)
    else:
        plt.show()

def discover_checkpoints(args):
    if args.checkpoints:
        ckpts = [os.path.abspath(p) for p in args.checkpoints]
    elif args.ckpt_dir:
        ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, args.pattern)))
    else:
        raise ValueError("Provide either --checkpoints (list of paths) or --ckpt_dir with --pattern.")
    if len(ckpts) == 0:
        raise ValueError("No checkpoints found.")
    return ckpts

def load_models(ckpt_paths, model_name, device):
    MODEL = importlib.import_module(model_name)
    models = []
    for p in ckpt_paths:
        m = MODEL.get_model(NUM_CLASSES).to(device)
        ckpt = torch.load(p, map_location=device)
        m.load_state_dict(ckpt['model_state_dict'])
        m.eval()
        models.append(m)
    return models

def seg_to_BNC(seg_pred, npoints, num_classes):
    """
    Convert model output to shape [B, N, C] robustly.
    Accepts [B, N, C], [B, C, N], or [B*N, C].
    """
    if seg_pred.dim() == 3:
        B, A, B_or_C = seg_pred.shape
        # case [B, N, C]
        if B_or_C == num_classes:
            return seg_pred
        # case [B, C, N]
        if A == num_classes:
            return seg_pred.permute(0, 2, 1).contiguous()
        raise RuntimeError(f"Unexpected seg_pred shape {tuple(seg_pred.shape)}")
    elif seg_pred.dim() == 2:
        # assume [B*N, C]
        BN, C = seg_pred.shape
        assert C == num_classes, f"seg_pred second dim {C} != num_classes {num_classes}"
        B = BN // npoints
        return seg_pred.view(B, npoints, num_classes).contiguous()
    else:
        raise RuntimeError(f"Unexpected seg_pred dim {seg_pred.dim()}")

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Discover checkpoints
    ckpt_paths = discover_checkpoints(args)
    print("Using checkpoints:")
    for p in ckpt_paths:
        print("  -", p)

    # Output dir
    out_dir = args.out_dir or os.path.dirname(ckpt_paths[0])
    os.makedirs(out_dir, exist_ok=True)

    # Load xyz ranges (global normalization)
    if args.xyz_range:
        xyz_path = args.xyz_range
    else:
        # infer from first checkpoint sibling ../xyz_range.npz
        xyz_path = os.path.join(os.path.dirname(ckpt_paths[0]), '..', 'xyz_range.npz')
    data = np.load(xyz_path)
    xyz_min, xyz_max = data['xyz_min'], data['xyz_max']
    print(f"Loaded global XYZ min: {xyz_min.tolist()}")
    print(f"Loaded global XYZ max: {xyz_max.tolist()}")

    # Load models
    models = load_models(ckpt_paths, args.model, device)

    # Prepare Lille2 dataset
    all_xyz = sorted(glob.glob(os.path.join(args.data_dir, '*_xyz.npy')))
    basenames = [os.path.basename(p).replace('_xyz.npy', '') for p in all_xyz]
    test_basenames = [b for b in basenames if 'Lille2' in b]
    print(f"Found {len(test_basenames)} Lille2 blocks for testing")

    test_ds = ParisLilleDataset(
        args.data_dir,
        file_list=test_basenames,
        num_point=args.npoint,
        normalize=True,
        xyz_min=xyz_min,
        xyz_max=xyz_max
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=8)

    # Metrics
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    # Inference (ensemble)
    with torch.no_grad():
        loop = tqdm(test_loader, desc='Testing on Lille2 (ensemble)')
        for pts, lbl in loop:
            pts, lbl = pts.to(device), lbl.to(device)     # pts: [B,N,3], lbl: [B,N]
            B, N, _ = pts.shape
            pts_in = pts.transpose(2, 1)                  # [B,3,N]

            if args.ensemble == 'avg':
                # accumulate probabilities
                sum_probs = None  # [B,N,C]
                for m in models:
                    seg_pred, _ = m(pts_in)              # variety of shapes
                    seg_bnc = seg_to_BNC(seg_pred, N, NUM_CLASSES)  # [B,N,C]
                    probs = torch.softmax(seg_bnc, dim=2)           # over classes
                    sum_probs = probs if sum_probs is None else (sum_probs + probs)
                avg_probs = sum_probs / len(models)
                preds = avg_probs.argmax(dim=2).reshape(-1).cpu().numpy()  # [B*N]
            else:  # majority vote
                votes = []
                for m in models:
                    seg_pred, _ = m(pts_in)
                    seg_bnc = seg_to_BNC(seg_pred, N, NUM_CLASSES)
                    pred_i = seg_bnc.argmax(dim=2)       # [B,N]
                    votes.append(pred_i)
                # stack votes [M,B,N] -> majority along axis 0
                V = torch.stack(votes, dim=0)            # [M,B,N]
                # simple mode along M
                preds = torch.mode(V, dim=0).values.reshape(-1).cpu().numpy()

            gt = lbl.view(-1).cpu().numpy()
            # update confusion
            for t, p in zip(gt, preds):
                conf_mat[t, p] += 1

    # IoUs
    per_class_iou = {}
    print("\n==== IoU ====")
    for c in range(NUM_CLASSES):
        tp = conf_mat[c, c]
        fp = conf_mat[:, c].sum() - tp
        fn = conf_mat[c, :].sum() - tp
        denom = tp + fp + fn
        iou_c = tp / denom if denom > 0 else 0.0
        per_class_iou[c] = iou_c
        print(f"Class {c} ({seg_label_to_cat[c]}): IoU = {iou_c:.4f}")

    mean_iou = sum(per_class_iou.values()) / NUM_CLASSES
    print(f"\nOverall mIoU on Lille2: {mean_iou:.4f}")

    print("\n==== Precision, Recall, F1 ====")
    precision = {}
    recall    = {}
    f1_score  = {}

    for c in range(NUM_CLASSES):
        tp = conf_mat[c, c]
        fp = conf_mat[:, c].sum() - tp
        fn = conf_mat[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precision[c] = prec; recall[c] = rec; f1_score[c] = f1
        print(f"Class {c} ({seg_label_to_cat[c]}): P={prec:.4f}, R={rec:.4f}, F1={f1:.4f}")

    macro_p = sum(precision.values()) / NUM_CLASSES
    macro_r = sum(recall.values())    / NUM_CLASSES
    macro_f1 = sum(f1_score.values()) / NUM_CLASSES

    tp_sum = sum(conf_mat[i, i] for i in range(NUM_CLASSES))
    fp_sum = sum(conf_mat[:, i].sum() - conf_mat[i, i] for i in range(NUM_CLASSES))
    fn_sum = sum(conf_mat[i, :].sum() - conf_mat[i, i] for i in range(NUM_CLASSES))
    micro_p = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
    micro_r = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    print(f"\nMacro Precision: {macro_p:.4f}, Macro Recall: {macro_r:.4f}, Macro F1: {macro_f1:.4f}")
    print(f"Micro Precision: {micro_p:.4f}, Micro Recall: {micro_r:.4f}, Micro F1: {micro_f1:.4f}\n")

    # Plots
    iou_list = [per_class_iou[c] for c in range(NUM_CLASSES)]
    title_suffix = f" (Ensemble {len(models)} folds, {args.ensemble})"
    plot_test_metrics(conf_mat, iou_list, seg_label_to_cat, save_dir=out_dir, title_suffix=title_suffix)

if __name__ == '__main__':
    main()
