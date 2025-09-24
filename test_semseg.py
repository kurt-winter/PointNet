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
    parser = argparse.ArgumentParser('PointNet Semantic Segmentation Testing')
    parser.add_argument('--model', type=str, default='pointnet_sem_seg')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to best_model.pth')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--npoint', type=int, default=4096)
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(BASE_DIR, 'data', 'processed_blocks'),
                        help='Directory with processed blocks')
    return parser.parse_args()


def plot_test_metrics(conf_mat, per_class_iou, seg_label_to_cat, save_dir=None):
    C = len(per_class_iou)
    classes = [seg_label_to_cat[i] for i in range(C)]
    total_per_class = conf_mat.sum(axis=1)  # sum over predictions for each true label
    
    # build y-tick labels: "class_name (N)"
    ylabels = [f"{classes[i]} ({total_per_class[i]:,})" for i in range(C)]
    xs = np.arange(C)

    # 1) Per-class IoU bar chart
    plt.figure(figsize=(10,5))
    plt.bar(xs, per_class_iou)
    plt.xticks(xs, classes, rotation=45, ha='right', fontsize=9)
    plt.ylabel('IoU', fontsize=12)
    plt.title('Per-Class IoU on Lille2', fontsize=14)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'iou_per_class.png'), dpi=200)
    else:
        plt.show()

    # 2) Confusion matrix heatmap (linear scale)
    plt.figure(figsize=(12,10))
    cmap = plt.get_cmap('viridis')
    norm = Normalize(vmin=0, vmax=conf_mat.max())
    im = plt.imshow(conf_mat, norm=norm, cmap=cmap)
    cb = plt.colorbar(im, label='Count')
    cb.ax.tick_params(labelsize=10)

    # set ticks and labels
    plt.xticks(xs, classes, rotation=45, ha='right', fontsize=9)
    plt.yticks(xs, ylabels, fontsize=9)
    plt.xlabel('Predicted label', fontsize=12)
    plt.ylabel('True label (count)', fontsize=12)
    plt.title('Confusion Matrix on Lille2', fontsize=14)

    ax = plt.gca()

    # annotate each cell and highlight diagonal
    for i in range(C):
        for j in range(C):
            cnt = conf_mat[i, j]
            if cnt == 0:
                continue
            # pick text color for contrast
            rgba = cmap(norm(cnt))
            lum = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
            txt_color = 'black' if lum > 0.5 else 'white'
            plt.text(
                j, i,
                f"{cnt:,}",
                ha='center', va='center',
                color=txt_color,
                fontsize=8
            )
        # highlight the correct cell for class i
        rect = Rectangle(
            (i-0.5, i-0.5), 1, 1,
            linewidth=2, edgecolor='white', facecolor='none'
        )
        ax.add_patch(rect)

    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'confusion_matrix_annotated.png'), dpi=200)
    else:
        plt.show()



def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # MODEL LOADING
    MODEL = importlib.import_module(args.model)
    classifier = MODEL.get_model(NUM_CLASSES).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()

    # Load the global ranges computed during training
    xyz_path = os.path.join(os.path.dirname(args.checkpoint), '..', 'xyz_range.npz')
    data = np.load(xyz_path)
    xyz_min, xyz_max = data['xyz_min'], data['xyz_max']
    print(f"Loaded global XYZ min: {xyz_min.tolist()}")
    print(f"Loaded global XYZ max: {xyz_max.tolist()}")

    # Prepare Lille2 dataset
    all_xyz = sorted(glob.glob(os.path.join(args.data_dir, '*_xyz.npy')))
    basenames = [os.path.basename(p).replace('_xyz.npy','') for p in all_xyz]
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

    # Inference
    with torch.no_grad():
        loop = tqdm(test_loader, desc='Testing on Lille2')
        for pts, lbl in loop:
            pts, lbl = pts.to(device), lbl.to(device)
            pts = pts.transpose(2,1)
            seg_pred, _ = classifier(pts)
            seg_pred = seg_pred.view(-1, NUM_CLASSES)

            preds = seg_pred.argmax(dim=1).cpu().numpy()
            gt    = lbl.view(-1).cpu().numpy()
            for t, p in zip(gt, preds):
                conf_mat[t, p] += 1

    # Compute IoUs
    per_class_iou = {}
    print("\n==== IuO ====")
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

    print("\n==== Prec, Rec, F1 ====")

     # Compute per-class precision, recall, F1
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

        precision[c] = prec
        recall[c]    = rec
        f1_score[c]  = f1

        print(f"Class {c} ({seg_label_to_cat[c]}): "
              f"P={prec:.4f}, R={rec:.4f}, F1={f1:.4f}")

    # Macro-averages (simple mean over classes)
    macro_p = sum(precision.values()) / NUM_CLASSES
    macro_r = sum(recall.values())    / NUM_CLASSES
    macro_f1 = sum(f1_score.values()) / NUM_CLASSES

    # Micro-averages (global sums)
    tp_sum = sum(conf_mat[i, i] for i in range(NUM_CLASSES))
    fp_sum = sum(conf_mat[:, i].sum() - conf_mat[i, i] for i in range(NUM_CLASSES))
    fn_sum = sum(conf_mat[i, :].sum() - conf_mat[i, i] for i in range(NUM_CLASSES))

    micro_p = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
    micro_r = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    print(f"Macro Precision: {macro_p:.4f}, Macro Recall: {macro_r:.4f}, Macro F1: {macro_f1:.4f}")
    print(f"Micro Precision: {micro_p:.4f}, Micro Recall: {micro_r:.4f}, Micro F1: {micro_f1:.4f}\n")

    # build an *ordered* list of IoUs for plotting
    iou_list = [per_class_iou[c] for c in range(NUM_CLASSES)]
    plot_test_metrics(
        conf_mat,
        iou_list,
        seg_label_to_cat,
        save_dir=os.path.dirname(args.checkpoint)
    )

if __name__ == '__main__':
    main()
