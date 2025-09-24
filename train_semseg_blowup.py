import matplotlib
matplotlib.use('Agg')   # <-- no GUI, just image file backend

import argparse
import os
import torch
import torch.nn.functional as F
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from data_utils.parislille_dataset import ParisLilleDataset
from torch.utils.data import WeightedRandomSampler
from torch_cluster import knn_graph

from torch.optim.lr_scheduler import OneCycleLR

#from data_utils.lovasz_losses import lovasz_softmax
from data_utils.lovasz_losses import lovasz_softmax_flat


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

#torch.backends.cudnn.benchmark = True

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
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet_sem_seg')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--decay_rate', type=float, default=1e-4)
    parser.add_argument('--npoint', type=int, default=4096)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--lr_decay', type=float, default=0.7)
    parser.add_argument('--physics_loss', action='store_true')
    return parser.parse_args()

def focal_loss(inputs, targets, alpha, gamma=2.0, reduction='mean'):
    """
    inputs:   [B*N, C] raw logits
    targets:  [B*N]   ground-truth labels in 0�C-1
    alpha:    tensor[C] per-class weight
    """
    # standard CE - added ignore index
    ce = F.cross_entropy(inputs, targets, weight=alpha, reduction='none', ignore_index=255)
    # pt = exp(-CE)
    p_t = torch.exp(-ce)
    # focal term (1 - pt)^?
    loss = ( (1 - p_t)**gamma * ce )
    return loss.mean() if reduction=='mean' else loss.sum()

def physics_loss_smoothness(pts_u, logits_u, k=16):
    # pts_u: [B, N, D]  (D may be 4 if you added reflectance)
    # logits_u: [B, N, C]

    # 1) slice off only the XYZ channels
    pts_xyz = pts_u[:, :, :3]                  # now [B, N, 3]
    B, N, _   = pts_xyz.shape

    # flatten to [(B*N), 3]
    xyz = pts_xyz.contiguous().view(B * N, 3)
    batch_index = torch.arange(B, device=xyz.device).view(B,1).repeat(1,N).view(-1)
    edge_index = knn_graph(xyz, k, batch=batch_index, loop=False)  # [2, E]

    B, N, C = logits_u.shape

    # get softmaxed predictions
    P = F.softmax(logits_u.view(B, N, -1), dim=2).view(B*N, -1)      # [B*N, C]
    i,j = edge_index
    return F.mse_loss(P[i], P[j])

def physics_loss_height(pts_u, logits_u, ground_class=1, z_thresh=0.2):
    """
    Penalize high 'ground' probability above some Z threshold.

    pts_u    : Tensor of shape [B, 3, N] (x,y,z in channels 0,1,2)
    logits_u : Tensor of shape [B, N, C]
    """
    # 1) pull out the Z coordinate, shape [B, N]
    #    pts_u is [B,3,N], so pts_u[:,2,:] is [B,N]
    z = pts_u[:, 2, :].reshape(-1)    # -> [B*N]

    # 2) flatten logits to [B*N, C] and softmax
    B, N, C = logits_u.shape
    logits_flat = logits_u.reshape(-1, C)        # -> [B*N, C]
    probs      = F.softmax(logits_flat, dim=1)   # -> [B*N, C]

    # 3) grab the ground probability
    ground_p = probs[:, ground_class]            # -> [B*N]

    # 4) mask out all points below the height threshold
    mask = (z > z_thresh)
    if mask.sum() == 0:
        # no high points at all, so zero penalty
        return torch.tensor(0., device=pts_u.device)

    # 5) average ground-prob over only the �too-high� points
    return ground_p[mask].mean()

def physics_loss_entropy(logits):
    """
    Encourage moderate confidence: high-entropy = less spikey.
    logits:  [B*N, C]
    """
    P = F.softmax(logits, dim=1)
    ent = - (P * torch.log(P + 1e-8)).sum(dim=1)  # [B*N]
    return ent.mean()

def physics_loss_refl_smoothness(pts_u, logits, refl, k=16, sigma=0.1):
    """
    Reflectance-weighted smoothness:
      encourages P[i] � P[j] when pts[i]�pts[j] *and* refl[i]�refl[j].
    points:    [B, 3, N]
    logits:    [B, N, C]
    refl:      [B, 1, N]   (normalized to [0,1])
    """
    # 1) slice off only the XYZ channels
    pts_xyz = pts_u[:, :, :3]                  # now [B, N, 3]
    B, N, _   = pts_xyz.shape

    # (1) build a k-NN over the *XYZ* coords
    xyz = pts_xyz.contiguous().view(B * N, 3)
    batch = torch.arange(B, device=xyz.device).view(B,1).repeat(1,N).view(-1)
    edge_index = knn_graph(xyz, k, batch=batch, loop=False)  # [2, E]

    # (2) get softmaxed predictions per-point
    B, N, C = logits.shape
    P = F.softmax(logits.reshape(-1, C), dim=1)               # [B*N, C]

    # (3) compute a weight per-edge based on reflectance similarity
    B, N  = refl.shape
    r = refl.contiguous().view(B*N)                          # [B*N]
    i,j = edge_index
    w = torch.exp( - ((r[i] - r[j])**2) / (2*sigma*sigma) )   # [E]

    # (4) weighted MSE on the probabilities
    return ( w.unsqueeze(1) * (P[i] - P[j]).pow(2) ).mean()

def inplace_relu(m):
    if isinstance(m, torch.nn.ReLU):
        m.inplace = True

def plot_confusion(cm: np.ndarray, class_names: list[str]) -> plt.Figure:
    """
    Returns a matplotlib Figure containing the plotted confusion matrix.
    `cm` is an N�N array, `class_names` a list of length N.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    n = len(class_names)
    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # annotate each cell with its count
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )
    fig.tight_layout()
    return fig

def main(args):
    # 1) GPU setup
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) Logging + dirs
    timestr = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    base_dir = Path('log/sem_seg')
    run_dir  = base_dir / (args.log_dir or timestr)
    ckpt_dir = run_dir / 'checkpoints'; ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir  = run_dir / 'logs';        log_dir.mkdir(exist_ok=True)

    # file logger
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(str(log_dir/f"{args.model}.txt"))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    # TensorBoard writer
    tb_dir = run_dir / 'tb_logs'; tb_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))

    logger.info(f"PARAMS: {args}")
    print(f"PARAMS: {args}")

    # 3) Data
    TRAIN_ROOT = '/data/AI_Dev_Data/Paris-Lille-3D/training_10_classes/processed_blocks'
    TRAIN_DATASET = ParisLilleDataset(root_dir=TRAIN_ROOT,
                                     split='train',
                                     augment=True,
                                     split_ratio=0.85,
                                     num_point=args.npoint,
                                     block_size=1.0)
    TEST_DATASET  = ParisLilleDataset(root_dir=TRAIN_ROOT,
                                     split='test',
                                     augment=False,
                                     split_ratio=0.85,
                                     num_point=args.npoint,
                                     block_size=1.0)

    # block weight check
    w = np.array(TRAIN_DATASET.block_weights)
    print(f"block_weights -> min {w.min():.3f}, max {w.max():.3f}, mean {w.mean():.3f}, std {w.std():.3f}")
    # ---
    
    ''''''
    sampler = WeightedRandomSampler(
        weights=TRAIN_DATASET.block_weights,
        num_samples=len(TRAIN_DATASET),
        replacement=True
    )

    #use sampler instead of shuffle=true
    train_loader = torch.utils.data.DataLoader(
        TRAIN_DATASET, batch_size=args.batch_size, sampler=sampler,
        num_workers=32, pin_memory=True, drop_last=True,
        worker_init_fn=lambda x: np.random.seed(x + int(time.time()))
    )
    
    '''
    train_loader = torch.utils.data.DataLoader(
        TRAIN_DATASET, batch_size=args.batch_size, shuffle=True,
        num_workers=32, pin_memory=True, drop_last=True,
        worker_init_fn=lambda x: np.random.seed(x + int(time.time()))
    )'''

    pts_batch, labs_batch = next(iter(train_loader))
    unique, counts = torch.unique(labs_batch, return_counts=True)
    print(">>> Sanity check � labels in this first batch:", 
          dict(zip(unique.tolist(), counts.tolist())))


    test_loader = torch.utils.data.DataLoader(
        TEST_DATASET, batch_size=args.batch_size, shuffle=False,
        num_workers=32, pin_memory=True, drop_last=True
    )

    # sampling weights
    #weights = torch.Tensor(TRAIN_DATASET.labelweights).to(device)
    logger.info(f"Class weights: {TRAIN_DATASET.labelweights}")
    print("Class weights:", TRAIN_DATASET.labelweights)

    # 4) Model & optimizer
    MODEL = importlib.import_module(args.model)
    net       = MODEL.get_model(NUM_CLASSES).to(device)
    #criterion = MODEL.get_loss().to(device)
    net.apply(inplace_relu)

    # ����� TEMP ISOLATION: use plain CE so we know exactly what�s happening �����
    from torch.nn import CrossEntropyLoss

    # TRAIN_DATASET.labelweights is your numpy array of length NUM_CLASSES
    weights = torch.tensor(TRAIN_DATASET.labelweights, dtype=torch.float32).to(device)
    criterion = CrossEntropyLoss(weight=weights, ignore_index=255).to(device)
    # ----------------------------------

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=args.learning_rate,
            betas=(0.9,0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=args.learning_rate,
            momentum=0.9
        )

    # 2) set up OneCycleLR
    total_steps = args.epoch * len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=total_steps,
        pct_start=0.10,         # 10% of steps for warm-up
        anneal_strategy='cos', # cosine down-ramp after warm-up
        div_factor=10.0,
        final_div_factor=100.0   # end LR = max_lr/100
    )

    best_mIoU = 0.0
    best_epoch = 0

    # 5) Training loop
    for epoch in range(args.epoch):

        print(f"\n==== Starting Epoch {epoch+1}/{args.epoch} ====")

        # adjust LR ---- Now OneCycleLR!!! No need for the next 4 lines
        #lr = max(args.learning_rate * (args.lr_decay ** (epoch//args.step_size)), 1e-5)
        #for pg in optimizer.param_groups:
        #    pg['lr'] = lr
        #logger.info(f"Epoch {epoch+1}/{args.epoch}  LR={lr:.6f}")
        #writer.add_scalar('train/learning_rate', lr, epoch)

        # --- TRAIN ---
        net.train()
        train_loss = 0.0
        train_correct = 0
        train_total   = 0

        smooth_sum   = 0.0
        refl_sum     = 0.0
        height_sum   = 0.0
        ent_sum      = 0.0
        batch_count = 0

        train_loss_ce = 0.0
        train_loss_lov = 0.0

        for i, (points, target) in tqdm(enumerate(train_loader), total=len(train_loader), desc='Train'):
            optimizer.zero_grad()

            # pts already augmented & normalized in __getitem__
            pts = points.to(device)
            lbl = target.to(device)

            # rotate + to cuda
            #pts_np = points.data.numpy()
            #pts_np[:,:,:3] = provider.rotate_point_cloud_z(pts_np[:,:,:3])
            #pts = torch.Tensor(pts_np).to(device)
            #lbl = target.to(device)

            '''
            pts = pts.transpose(2,1)
            logits, trans_feat = net(pts)
            logits = logits.contiguous().view(-1, NUM_CLASSES)
            lbl_flat = lbl.view(-1)

            # --- supervised cross-entropy ---
            loss_sup = criterion(logits, lbl_flat, trans_feat, weights)
            #loss = focal_loss(inputs=logits, targets=lbl_flat, alpha=weights, gamma=2.0)

            w_ce     = 1.0
            w_smooth = 0.1    # geometry-only smoothness
            w_refl   = 0.1    # reflectance-aware smoothness
            w_height = 0.2    # height consistency
            w_ent    = 0.01   # entropy regularizer

            if args.physics_loss:
                # un-flatten shape ? [B, N, C] and [B,3,N]
                B, _, _ = pts.shape         # pts is [B, C_in, N]
                logits_u = logits.view(B, NUM_CLASSES, -1).permute(0,2,1)
                pts_u    = pts        # originally [B, 3, N]
                refl_u   = pts[:,3:4,:]  # [B,1,N] if you kept refl in the last channel

                # the smoothness from before
                Ls = physics_loss_smoothness(pts_u, logits_u, k=16)
                # this brand-new reflectance-weighted smoothness
                Lr = physics_loss_refl_smoothness(pts_u, logits_u, refl_u, k=16, sigma=0.1)
                # and maybe your height / entropy penalties too
                Lh = physics_loss_height(pts_u, logits_u, ground_class=1, z_thresh=0.2)
                Le = physics_loss_entropy(logits)

                # combine with small weights
                loss = w_ce*loss_sup + w_smooth*Ls + w_refl*Lr + w_height*Lh + w_ent*Le

            else:
                loss = loss_sup
            '''

            pts = pts.transpose(2,1)       # [B, 3(+feat), N]
            B, _, N = pts.shape            # grab batch-size and num-points
            logits, trans_feat = net(pts)  # logits: [B, C, N]

            w_ce     = 1.0
            w_smooth = 0.001    # geometry-only smoothness
            w_refl   = 0.001    # reflectance-aware smoothness
            w_height = 0.01    # height consistency
            w_ent    = 0.0001   # entropy regularizer

            # --- physics losses want the unflattened [B,N,C] logits and [B,3,N] pts ---
            if args.physics_loss:
                #logits_u = logits.permute(0, 2, 1)  # [B,C,N] ? [B,N,C]
                #pts_u    = pts                     # [B, 3, N]
                #refl_u   = pts[:, 3:4, :]          # [B, 1, N]  (reflectance channel)
                logits_u = logits.view(-1, NUM_CLASSES, logits.size(0)//NUM_CLASSES) \
                          .permute(0,2,1)
                B, N, C = logits_u.shape
                pts_u  = pts[:, :3, :].contiguous()   # [B,3,N]
                refl_u = pts[:, 3, :].contiguous()    # [B,N]
                Ls = physics_loss_smoothness   (pts_u,   logits_u, k=16)
                Lr = physics_loss_refl_smoothness(pts_u, logits_u, refl_u, k=16, sigma=0.1)
                Lh = physics_loss_height       (pts_u,   logits_u, ground_class=1, z_thresh=0.2)
                #Le = physics_loss_entropy      (logits_u.view(-1, NUM_CLASSES))  # or logits.view(-1,C)

            # --- now flatten for the supervised CE loss ---
            logits_flat = logits.permute(0,2,1).contiguous().view(-1, NUM_CLASSES)  # [B*N, C]
            lbl_flat    = lbl.view(-1)

            #loss_sup = criterion(logits_flat, lbl_flat, trans_feat, weights)
            #loss_sup = criterion(logits_flat, lbl_flat)
            loss_ce = criterion(logits_flat, lbl_flat)

            # Lovász-Softmax loss
            probs    = F.softmax(logits_flat, dim=1)
            # lovasz_softmax_flat takes (probas_flat: [P,C], labels_flat: [P], classes=..e.)
            #loss_sup = lovasz_softmax_flat(probs, lbl_flat, classes='present')
            loss_lov = lovasz_softmax_flat(probs, lbl_flat, classes='present')

            loss_sup = 0.5 * loss_ce + 0.5 * loss_lov


            if args.physics_loss:
                loss = (w_ce   * loss_sup +
                        w_smooth * Ls +
                        w_refl   * Lr +
                        w_height * Lh )#+
                        #w_ent    * Le)
            else:
                loss = loss_sup
        


            loss.backward()

            #gradient clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

            optimizer.step()

            # Scheduler!!!#############
            scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            # if you want per-batch logging:
            writer.add_scalar('train/learning_rate', current_lr, epoch * len(train_loader) + i)
            # or if you prefer one point per epoch, log here:
            # writer.add_scalar('train/learning_rate', current_lr, epoch)
            ##########################

            #preds = logits.max(1)[1]
            preds = logits.argmax(dim=2)
            preds_flat  = preds.reshape(-1)  # -> [B*N]
            #train_correct += (preds==lbl_flat).sum().item()
            train_correct += (preds_flat==lbl_flat).sum().item()
            train_total   += lbl_flat.size(0)
            train_loss    += loss.item()

            train_loss_ce = loss_ce.item()
            train_loss_lov = loss_lov.item()

            if args.physics_loss:
                smooth_sum += Ls.item()
                refl_sum   += Lr.item()
                height_sum += Lh.item()
                #ent_sum    += Le.item()
            batch_count += 1

        avg_train_loss = train_loss / len(train_loader)
        train_acc      = train_correct / train_total
        logger.info(f"Train loss: {avg_train_loss:.6f}, acc: {train_acc:.4f}")
        print(f"Training mean loss: {avg_train_loss:.6f}")
        print(f"Training accuracy: {train_acc:.6f}")

        # TB logs
        writer.add_scalar('train/loss',     avg_train_loss, epoch)
        writer.add_scalar('train/accuracy', train_acc,      epoch)

        writer.add_scalar('train/loss_ce', train_loss_ce   / batch_count, epoch)
        writer.add_scalar('train/loss_lov', train_loss_lov   / batch_count, epoch)

        if args.physics_loss:
            writer.add_scalar('train/L_smooth_epoch', smooth_sum   / batch_count, epoch)
            writer.add_scalar('train/L_refl_epoch',   refl_sum     / batch_count, epoch)
            writer.add_scalar('train/L_height_epoch', height_sum   / batch_count, epoch)
            #writer.add_scalar('train/L_ent_epoch',    ent_sum      / batch_count, epoch)

        # --- EVAL ---
        net.eval()
        eval_loss = 0.0
        eval_correct = 0
        eval_total   = 0
        eval_ce_loss  = 0.0
        eval_lov_loss = 0.0

        all_preds = []
        all_gts   = []
        all_probs = []

        with torch.no_grad():
            for points, target in tqdm(test_loader, total=len(test_loader), desc='Eval'):
                pts = points.to(device)
                lbl = target.to(device)

                pts = pts.transpose(2,1)
                logits, trans_feat = net(pts)
                #logits = logits.contiguous().view(-1, NUM_CLASSES)
                logits_flat = logits.permute(0,2,1).contiguous().view(-1, NUM_CLASSES)
                lbl_flat = lbl.view(-1)

                #loss = criterion(logits, lbl_flat, trans_feat, weights)
                # CE eval loss
                ce_loss = criterion(logits_flat, lbl_flat)
                eval_ce_loss += ce_loss.item()

                # Lov�asz eval loss (if you want)
                probs = F.softmax(logits_flat, dim=1)
                lov_loss = lovasz_softmax_flat(probs, lbl_flat, classes='present')
                eval_lov_loss += lov_loss.item()

                loss = 0.5*ce_loss + 0.5*lov_loss
                eval_loss = loss.item()

                preds = logits_flat.max(1)[1].cpu().numpy()
                all_preds.append(preds)
                all_gts.append(lbl_flat.cpu().numpy())

                probs = F.softmax(logits_flat, dim=1).cpu().numpy()
                all_probs.append(probs)

                eval_correct += (preds==(lbl_flat.cpu().numpy())).sum()
                eval_total   += lbl_flat.size(0)

        avg_eval_lov_loss = eval_lov_loss / len(test_loader)
        avg_eval_ce_loss = eval_ce_loss / len(test_loader)
        avg_eval_loss = eval_loss / len(test_loader)
        eval_acc      = eval_correct / eval_total

        # flatten arrays
        all_preds = np.concatenate(all_preds)
        all_gts   = np.concatenate(all_gts)
        all_probs = np.concatenate(all_probs, axis=0)

        # confusion matrix ? IoU / precision / recall
        cm = confusion_matrix(all_gts, all_preds, labels=list(range(NUM_CLASSES)))
        TP = np.diag(cm)
        FP = cm.sum(axis=0) - TP
        FN = cm.sum(axis=1) - TP

        iou_per_class = TP / (TP + FP + FN + 1e-6)
        acc_per_class = TP / (cm.sum(axis=1) + 1e-6)
        precision_per = TP / (TP + FP + 1e-6)
        recall_per    = TP / (TP + FN + 1e-6)
        mIoU   = iou_per_class.mean()
        p_macro = precision_per.mean()
        r_macro = recall_per.mean()

        # console print
        print(f"eval mean lov loss: {avg_eval_lov_loss:.6f}")
        print(f"eval mean ce loss: {avg_eval_ce_loss:.6f}")
        print(f"eval mean loss: {avg_eval_loss:.6f}")
        print(f"eval point avg class IoU (mIoU): {mIoU:.6f}")
        print(f"eval point accuracy: {eval_acc:.6f}")
        for c in range(NUM_CLASSES):
            print(f"class {seg_label_to_cat[c]:<12} IoU: {iou_per_class[c]:.3f}")

        print(f"eval precision (macro): {p_macro:.6f}")
        print(f"eval recall    (macro): {r_macro:.6f}")
        for c in range(NUM_CLASSES):
            print(f" class {seg_label_to_cat[c]:<12} P: {precision_per[c]:.3f}  R: {recall_per[c]:.3f} Acc: {acc_per_class[c]:.3f} ")

        # TB logs
        writer.add_scalar('eval/lov_loss',     avg_eval_lov_loss, epoch)
        writer.add_scalar('eval/ce_loss',     avg_eval_ce_loss, epoch)
        writer.add_scalar('eval/accuracy', eval_acc,      epoch)
        writer.add_scalar('eval/mIoU',     mIoU,          epoch)
        writer.add_scalar('eval/precision', p_macro,      epoch)
        writer.add_scalar('eval/recall',    r_macro,      epoch)
        for c in range(NUM_CLASSES):
            writer.add_scalar(f'eval/IoU_class/{c}',      iou_per_class[c],   epoch)
            writer.add_scalar(f'eval/Precision_class/{c}',precision_per[c], epoch)
            writer.add_scalar(f'eval/Recall_class/{c}',   recall_per[c],    epoch)
            writer.add_scalar(f'eval/Accuracy_class/{c}',   acc_per_class[c],    epoch)

        # plot + write confusion matrix figure
        fig = plot_confusion(cm, [seg_label_to_cat[i] for i in range(NUM_CLASSES)])
        writer.add_figure('eval/ConfusionMatrix', fig, epoch)
        plt.close(fig)

        # Save best
        if mIoU > best_mIoU:
            best_mIoU = mIoU
            best_epoch = epoch+1
            savepath = ckpt_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mIoU': best_mIoU
            }, savepath)
            print(f"Saved best model with mIoU={best_mIoU:.6f}")

        print(f"eval best mIoU: {best_mIoU:.6f}")
        print(f"eval best epoch: {best_epoch}")

    writer.close()

if __name__ == '__main__':
    args = parse_args()
    main(args)


    '''
    def physics_loss(pred_labels, points):
    z = torch.nan_to_num(points[:, 2, :], nan=0.0)
    penalty = torch.zeros_like(z, dtype=torch.float32)


    road_mask = (pred_labels == 1)
    penalty += road_mask * (z - 2.0).clamp(min=0.0)


    building_mask = (pred_labels == 3)
    penalty += building_mask * (1.0 - z).clamp(min=0.0)


    return penalty.mean()
    
    
    def physics_loss(pred_labels, points):
    z = torch.nan_to_num(points[:, 2, :], nan=0.0)
    penalty = torch.zeros_like(z, dtype=torch.float32)


    road_mask = (pred_labels == 1)
    penalty += road_mask * (z - 2.0).clamp(min=0.0)


    building_mask = (pred_labels == 3)
    penalty += building_mask * (1.0 - z).clamp(min=0.0)
    
    https://scikit-learn.org/stable/modules/cross_validation.html'''
