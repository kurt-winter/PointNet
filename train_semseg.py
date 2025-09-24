import os
import sys
import glob
import argparse
import importlib
import shutil
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm
import torch.nn.functional as F
import csv
import json

from data_utils.parislille_dataset import ParisLilleDataset

# BASE & PYTHON PATH SETUP
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
    parser = argparse.ArgumentParser('PointNet Semantic Segmentation Training')
    parser.add_argument('--model', type=str, default='pointnet_sem_seg')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--decay_rate', type=float, default=1e-4)
    parser.add_argument('--npoint', type=int, default=4096)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--lr_decay', type=float, default=0.7)
    parser.add_argument('--physics_loss', action='store_true', default=False)
    parser.add_argument('--physics_weight', type=float, default=0.1, help='Scale for physics penalty term')
    parser.add_argument('--smooth_weight',  type=float, default=0.01, help='Scale for smoothness term')
    parser.add_argument('--floor_file', type=str, default='floor_planes.json', help='JSON with per-area floor plane coefficients')
    parser.add_argument('--bld_plane_margin', type=float, default=0.10,
                    help='Meters buildings should sit above the plane (tolerance).')
    parser.add_argument('--bld_below_weight', type=float, default=0.20,
                    help='Weight for the building-below-plane penalty in physics loss.')
    return parser.parse_args()


def log_string(out_str, log_file=None):
    print(out_str)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(out_str + '\n')


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)

def load_floor_planes(json_path):
    with open(json_path, 'r') as f:
        J = json.load(f)
    planes = {}
    for area, info in J.items():
        a, b, c = info['coeff']
        planes[area] = (float(a), float(b), float(c))
    return planes

def area_from_blockname(name: str) -> str:
    # map your basenames to areas; adapt if your names differ
    if name.startswith('Lille1_1'): return 'Lille1_1'
    if name.startswith('Lille1_2'): return 'Lille1_2'
    if name.startswith('Paris'):     return 'Paris'
    # Lille2 is testset only; we won't use planes for training on it
    return None


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Directories
    if args.log_dir:
        experiment_dir = args.log_dir
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, 'checkpoints'), exist_ok=True)
    else:
        experiment_dir = BASE_DIR

    # Init csv
    METRICS_CSV = os.path.join(experiment_dir, 'metrics.csv')

    # MODEL LOADING
    MODEL = importlib.import_module(args.model)
    shutil.copy(f'models/{args.model}.py', str(experiment_dir))
    shutil.copy('models/pointnet2_utils.py', str(experiment_dir))

    classifier = MODEL.get_model(NUM_CLASSES).to(device)
    criterion  = MODEL.get_loss().to(device)
    classifier = classifier.apply(weights_init)

    # Try resume
    start_epoch = 0
    '''
    try:
        ckpt = torch.load(os.path.join(experiment_dir, 'checkpoints', 'best_model.pth'))
        start_epoch = ckpt.get('epoch', 0)
        classifier.load_state_dict(ckpt['model_state_dict'])
        log_string('Use pre-trained model', os.path.join(experiment_dir, 'log.txt'))
    except Exception:
        log_string('No existing model, starting from scratch', os.path.join(experiment_dir, 'log.txt'))
    '''

    # floor plane data
    area_planes = load_floor_planes(args.floor_file)  # {'Lille1_1': (a,b,c), ...}

    print(area_planes)

    # Data setup: exclude Lille2
    ROOT_DATA = '/data/AI_Dev_Data/Paris-Lille-3D/training_10_classes/processed_blocks'
    all_xyz = sorted(glob.glob(os.path.join(ROOT_DATA, '*_xyz.npy')))
    basenames = [os.path.basename(p).replace('_xyz.npy','') for p in all_xyz]
    all_train = [b for b in basenames if 'Lille2' not in b]

    #new max
    #ds_full = ParisLilleDataset(ROOT_DATA, file_list=basenames, normalize=True) #also include Lille2 here
    #xyz_min, xyz_max = ds_full.xyz_min, ds_full.xyz_max

    #load from before
    xyz_path = os.path.join(experiment_dir, 'xyz_range.npz')
    data = np.load(xyz_path)
    xyz_min, xyz_max = data['xyz_min'], data['xyz_max']
    print(f"Loaded global XYZ min: {xyz_min.tolist()}")
    print(f"Loaded global XYZ max: {xyz_max.tolist()}")

    #xyz_min, xyz_max = ds_full.xyz_min, ds_full.xyz_max

    z_min, z_max = float(xyz_min[2]), float(xyz_max[2])

    # Print to console & log
    print(f"Global XYZ min: {xyz_min.tolist()}")
    print(f"Global XYZ max: {xyz_max.tolist()}")

    # Save to disk for later reuse
    np.savez(
        os.path.join(experiment_dir, 'xyz_range.npz'),
        xyz_min=xyz_min,
        xyz_max=xyz_max
    )
    print(f"Saved global XYZ ranges to {os.path.join(experiment_dir, 'xyz_range.npz')}")

    

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_train), start=1):
        log_string(f'\n=== Fold {fold}/5 ===', os.path.join(experiment_dir, 'log.txt'))
        train_list = [all_train[i] for i in train_idx]
        val_list   = [all_train[i] for i in val_idx]

        train_ds = ParisLilleDataset(ROOT_DATA, file_list=train_list,
                             num_point=args.npoint,
                             normalize=True,
                             xyz_min=xyz_min,
                             xyz_max=xyz_max,
                             return_name=True)
        val_ds   = ParisLilleDataset(ROOT_DATA, file_list=val_list,
                             num_point=args.npoint,
                             normalize=True,
                             xyz_min=xyz_min,
                             xyz_max=xyz_max,
                             return_name=True)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=8, drop_last=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                                  shuffle=False, num_workers=8)

        # Optimizer & scheduler
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate,
                                   weight_decay=args.decay_rate)
        else:
            optimizer = optim.SGD(classifier.parameters(), lr=args.learning_rate,
                                  momentum=0.9, weight_decay=args.decay_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=args.step_size,
                                              gamma=args.lr_decay)

        best_iou = 0.0
        
        for epoch in range(start_epoch+1, args.epoch+1):
            # Train
            classifier.train()
            loop = tqdm(train_loader, desc=f'Fold{fold} Epoch{epoch} [Train]')
            train_loss = total_correct = total_points = 0

            train_phys_loss = 0.0
            train_smooth_loss = 0.0
            train_ce_loss = 0.0


            for pts_raw, lbl, names in loop:
            #for pts_raw, lbl in loop: #changed for return names
                pts_raw, lbl = pts_raw.to(device), lbl.to(device) # B, N, 3
                pts = pts_raw.transpose(2,1) # B, 3, N
                optimizer.zero_grad()

                ''' removed for physics stuff
                seg_pred, trans_feat = classifier(pts)
                seg_pred = seg_pred.view(-1, NUM_CLASSES)
                loss = criterion(
                    seg_pred,
                    lbl.view(-1),
                    trans_feat,
                    train_ds.labelweights.to(device)
                )
                loss.backward() 
                optimizer.step()
                '''

                # physics stuff
                # forward
                seg_pred, trans_feat = classifier(pts)     # [B, N, C]
                seg_flat = seg_pred.view(-1, NUM_CLASSES)            # [B*N, C]

                # 1) base CE loss
                base_loss = criterion(
                    seg_flat,
                    lbl.view(-1),
                    trans_feat,
                    train_ds.labelweights.to(device)
                )

                train_ce_loss += base_loss.item()

                # 2) physics penalties
                # other ideas: plain ground ; vertical building; 
                ''' changed loss for interpolated ground plane
                if args.physics_loss:
                    # recover world Z
                    z_norm  = pts_raw[:,:,2]   # [B,N]
                    z_world = z_norm * (z_max - z_min) + z_min

                    seg_phy = seg_pred.permute(0, 2, 1)         # [B, C, N]
                    B, C, N = seg_phy.shape

                    probs = torch.softmax(seg_phy, dim=1) # [B,C,N]
                    pg = probs[:,1,:] * torch.relu  (z_world - 2.0)     
                    pb = probs[:,2,:] * torch.relu  (1.0 - z_world)     
                    pp = probs[:,7,:] * torch.relu  (z_world - 2.0)     
                    pc = probs[:,8,:] * torch.relu  (z_world - 3.0)     
                    pt = probs[:,5,:] * torch.relu  (z_world - 2.0)     
                    phys_loss = (pg + pb + pp + pc + pt).mean()

                    # 3) smoothness (simple KNN on the fly)
                    # compute pairwise dists and K neighbors
                    dists   = torch.cdist(pts_raw, pts_raw)        # [B,N,N]
                    knn_idx = dists.topk(9, largest=False).indices[:,:,1:]  # [B,N,8]
                    smooth = 0.0
                    for k in range(8):
                        nbr = seg_phy.gather(
                            2,
                            knn_idx[:,:,k]
                              .unsqueeze(1)
                              .expand(-1, C, -1)
                        )                                  # [B,C,N]
                        smooth += F.mse_loss(seg_phy, nbr)
                    smooth_loss = smooth / 8

                    train_phys_loss  += phys_loss.item()   if args.physics_loss else 0.0
                    train_smooth_loss+= smooth_loss.item() if args.physics_loss else 0.0
                    

                    loss = ( base_loss
                           + args.physics_weight * phys_loss
                           + args.smooth_weight  * smooth_loss )
                
                           '''
                # ----- Physics penalties via HAG from area plane -----
                if args.physics_loss:
                    # 1) de-normalize to world meters
                    scale = torch.tensor((xyz_max - xyz_min), device=device, dtype=pts_raw.dtype)  # [3]
                    off   = torch.tensor(xyz_min, device=device, dtype=pts_raw.dtype)              # [3]
                    P     = pts_raw * scale + off                          # [B,N,3] world coords
                    X, Y, Z = P[...,0], P[...,1], P[...,2]                 # [B,N] each

                    # 2) logits -> probs  (standardize shape to [B,N,C] no matter what model returns)
                    logits = seg_pred                                      # seg_pred from model
                    if logits.dim() == 3 and logits.shape[1] == NUM_CLASSES:   # [B,C,N]
                        logits_bnc = logits.permute(0,2,1).contiguous()        # -> [B,N,C]
                    else:                                                      # already [B,N,C]
                        logits_bnc = logits
                    probs = torch.softmax(logits_bnc, dim=2)               # [B,N,C]
                    B, N, C = probs.shape

                    # class probability fields
                    g     = probs[..., 1]
                    bld   = probs[..., 2]
                    ped   = probs[..., 7]
                    car   = probs[..., 8]
                    trash = probs[..., 5]
                    boll  = probs[..., 4]

                    # 3) per-sample plane coefficients (a,b,c) by area
                    B, N = X.shape
                    a = torch.zeros(B,1, device=device, dtype=P.dtype)
                    b = torch.zeros_like(a)
                    c = torch.zeros_like(a)
                    for bi, nm in enumerate(names):
                        area = area_from_blockname(nm)
                        if area is not None and area in area_planes:
                            aa, bb, cc = area_planes[area]       # floats from JSON (meters)
                            a[bi,0], b[bi,0], c[bi,0] = aa, bb, cc
                        else:
                            # rare fallback: local 5% Z as a flat ground
                            c[bi,0] = torch.quantile(Z[bi], 0.05)

                    z_ground = (a*X + b*Y + c)                   # [B,N]
                    hag      = (Z - z_ground).clamp_min(0.0)     # [B,N] meters above plane
                    dz = Z - z_ground                            # [B,N]  signed height (meters)

                    # 4) penalties (tune thresholds; all in meters and applied to HAG)
                    loss_ground = (g     * F.relu(hag - 0.30)).mean()
                    loss_ped    = (ped   * F.relu(hag - 2.20)).mean()
                    loss_car    = (car   * F.relu(hag - 3.00)).mean()
                    loss_trash  = (trash * F.relu(hag - 2.00)).mean()
                    loss_boll   = (boll  * F.relu(hag - 1.50)).mean()

                    # buildings should be above the plane by at least margin
                    loss_bld_above = (bld * F.relu(args.bld_plane_margin - dz)).mean()

                    phys_loss   = loss_ground + loss_ped + loss_car + loss_trash + loss_boll + args.bld_below_weight * loss_bld_above


                    # 5) smoothness (simple KNN on the fly)
                    # compute pairwise dists and K neighbors
                    # Compute k-NN in *coord* space; distances don't need grad.
                    # You can use XY only to save a bit of memory.
                    probs = torch.softmax(logits_bnc, dim=2)     # [B, N, C]
                    B, N, C = probs.shape
                    K = 8

                    # 1) k-NN in XY (no grad)
                    with torch.no_grad():
                        XY = pts_raw[..., :2].contiguous()       # [B, N, 2]
                        d  = torch.cdist(XY, XY)                 # [B, N, N]
                        knn_idx = d.topk(K+1, largest=False).indices[:, :, 1:]   # [B, N, K]

                    # 2) Gather neighbor class probs via a single flatten-then-gather
                    idx_flat = knn_idx.reshape(B, N*K)                           # [B, N*K]
                    idx_flat = idx_flat.unsqueeze(-1).expand(-1, -1, C)          # [B, N*K, C]
                    nbr_probs = torch.gather(probs, 1, idx_flat)                 # [B, N*K, C]
                    nbr_probs = nbr_probs.reshape(B, N, K, C)                    # [B, N, K, C]

                    # 3) Mean squared difference to neighbors
                    center = probs.unsqueeze(2)                                  # [B, N, 1, C]
                    smooth = (center - nbr_probs).pow(2).mean()
                    smooth_loss = smooth

                    train_phys_loss  += phys_loss.item()   if args.physics_loss else 0.0
                    train_smooth_loss+= smooth_loss.item() if args.physics_loss else 0.0
                    

                    loss = ( base_loss
                           + args.physics_weight * phys_loss
                           + args.smooth_weight  * smooth_loss )
                else:
                    loss = base_loss

                # backward + step
                loss.backward()
                optimizer.step()
                #End physics stuff


                train_loss += loss.item()

                # why....
                #preds = seg_pred.argmax(dim=1)
                #total_correct += (preds == lbl.view(-1)).sum().item()

                # flatten for accuracy
                seg_flat = seg_pred.view(-1, NUM_CLASSES)   # [B*N, C]
                lbl_flat = lbl.view(-1)                     # [B*N]
                preds_flat = seg_flat.argmax(dim=1)         # [B*N]
                total_correct += (preds_flat == lbl_flat).sum().item()

                
                total_points  += lbl.numel()
                loop.set_postfix(
                    loss=train_loss/(loop.n+1),
                    acc=total_correct/total_points if total_points > 0 else 0,
                    ce=train_ce_loss/(loop.n+1),
                    phys=train_phys_loss/(loop.n+1),
                    smooth=train_smooth_loss/(loop.n+1)
                )

            scheduler.step()

            # --- Validation ---
            classifier.eval()
            vloop = tqdm(val_loader, desc=f'Fold{fold} Epoch{epoch} [Val]')
            val_loss = 0.0
            # reset confusion matrix for this epoch
            conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

            phys_loss = 0.0
            smooth_loss_accum = 0.0

            with torch.no_grad():
                #for batch_idx, (pts_raw, lbl) in enumerate(vloop):
                for batch_idx, (pts_raw, lbl,names) in enumerate(vloop):
                    pts_raw, lbl = pts_raw.to(device), lbl.to(device)
                    pts = pts_raw.transpose(2,1)
                    seg_pred, trans_feat = classifier(pts)
                    ''' get HAG in
                    pred_flat = seg_pred.view(-1, NUM_CLASSES)

                    # loss
                    loss = criterion(
                        pred_flat,
                        lbl.view(-1),
                        trans_feat,
                        train_ds.labelweights.to(device)
                    )
                    val_loss += loss.item()

                    if args.physics_loss:
                        # recover world Z, compute probs
                        z_norm  = pts_raw[:,:,2]
                        z_world = z_norm*(z_max - z_min) + z_min

                        seg_phy = seg_pred.permute(0, 2, 1)         # [B, C, N]
                        probs   = torch.softmax(seg_phy, dim=1)

                        pg = probs[:,1,:] * torch.relu  (z_world - 2.0)     
                        pb = probs[:,2,:] * torch.relu  (1.0 - z_world)     
                        pp = probs[:,7,:] * torch.relu  (z_world - 2.0)     
                        pc = probs[:,8,:] * torch.relu  (z_world - 3.0)     
                        pt = probs[:,5,:] * torch.relu  (z_world - 2.0)     
                        phys_val = (pg + pb + pp + pc + pt).mean().item()

                        # 3) smoothness (simple KNN on the fly)
                        # compute pairwise dists and K neighbors
                        dists   = torch.cdist(pts_raw, pts_raw)        # [B,N,N]
                        knn_idx = dists.topk(9, largest=False).indices[:,:,1:]  # [B,N,8]
                        smooth = 0.0
                        for k in range(8):
                            nbr = seg_phy.gather(
                                2,
                                knn_idx[:,:,k]
                                .unsqueeze(1)
                                .expand(-1, C, -1)
                            )                                  # [B,C,N]
                            smooth += F.mse_loss(seg_phy, nbr)
                        smooth_loss = smooth / 8
                        smooth_val = smooth_loss.item() '''
                    
                    # standardize logits to [B,N,C]
                    if seg_pred.dim() == 3 and seg_pred.shape[1] == NUM_CLASSES:   # [B,C,N]
                        seg_bnc = seg_pred.permute(0, 2, 1).contiguous()            # ? [B,N,C]
                    else:
                        seg_bnc = seg_pred.contiguous()                             # [B,N,C]
                    B, N, C = seg_bnc.shape

                    # CE loss (like training)
                    pred_flat = seg_bnc.view(-1, C)
                    loss = criterion(
                        pred_flat,
                        lbl.view(-1),
                        trans_feat,
                        train_ds.labelweights.to(device)
                    )
                    val_loss += loss.item()

                    # ---- physics + smooth (for debug) ----
                    if args.physics_loss:
                        # de-normalize to world meters
                        scale = torch.tensor((xyz_max - xyz_min), device=device, dtype=pts_raw.dtype)
                        off   = torch.tensor(xyz_min, device=device, dtype=pts_raw.dtype)
                        P     = pts_raw * scale + off                                   # [B,N,3]
                        X, Y, Z = P[...,0], P[...,1], P[...,2]                          # [B,N]

                        # per-sample plane coeffs (a,b,c) by area
                        a = torch.zeros(B,1, device=device, dtype=P.dtype)
                        b = torch.zeros_like(a)
                        c = torch.zeros_like(a)
                        for bi, nm in enumerate(names):
                            area = area_from_blockname(nm)
                            if area and area in area_planes:
                                aa, bb, cc = area_planes[area]
                                a[bi,0], b[bi,0], c[bi,0] = aa, bb, cc
                            else:
                                c[bi,0] = torch.quantile(Z[bi], 0.05)  # rare fallback

                        z_ground = (a*X + b*Y + c)                      # [B,N]
                        hag      = (Z - z_ground).clamp_min(0.0)        # [B,N] meters

                        probs = torch.softmax(seg_bnc, dim=2)           # [B,N,C]
                        g     = probs[..., 1]
                        ped   = probs[..., 7]
                        car   = probs[..., 8]
                        trash = probs[..., 5]
                        boll  = probs[..., 4]

                        p_ground = (g    * F.relu(hag - 0.30)).mean() # ground shouldnt be >0.3m
                        p_ped    = (ped  * F.relu(hag - 2.20)).mean() # pedestrians shouldnt be >2.2m above plane
                        p_car    = (car  * F.relu(hag - 3.00)).mean() # cars shouldnt be >3.0m above plane
                        p_trash  = (trash* F.relu(hag - 2.00)).mean() # trash cans shouldnt be >2.0m
                        p_boll   = (boll * F.relu(hag - 1.50)).mean() # bollards shouldnt be >1.5m
                        dz = Z - z_ground
                        bld = probs[..., 2]
                        p_bld_above = (bld * F.relu(args.bld_plane_margin - dz)).mean()
                        phys_val = (p_ground + p_ped + p_car + p_trash + p_boll + (args.bld_below_weight * p_bld_above)).item()

                        # ---- smoothness (same as train, for logging) ----
                        probs = torch.softmax(seg_bnc, dim=2)                        # [B, N, C]
                        B, N, C = probs.shape
                        K = 8
                        with torch.no_grad():
                            XY = pts_raw[..., :2].contiguous()
                            d  = torch.cdist(XY, XY)                                 # [B, N, N]
                            knn_idx = d.topk(K+1, largest=False).indices[:, :, 1:]   # [B, N, K]

                        idx_flat = knn_idx.reshape(B, N*K).unsqueeze(-1).expand(-1, -1, C)  # [B, N*K, C]
                        nbr_probs = torch.gather(probs, 1, idx_flat).reshape(B, N, K, C)    # [B, N, K, C]
                        smooth = (probs.unsqueeze(2) - nbr_probs).pow(2).mean()
                        smooth_val = smooth.item()

                    else:
                        phys_val, smooth_val = 0.0, 0.0

                    phys_loss   += phys_val
                    smooth_loss_accum += smooth_val

                    # update confusion matrix
                    preds = pred_flat.argmax(dim=1).cpu().numpy()
                    gt    = lbl.view(-1).cpu().numpy()
                    for t, p in zip(gt, preds):
                        conf_mat[t, p] += 1

                    # progress bar: only show avg loss
                    # update tqdm to show them
                    avg_vloss   = val_loss   / (batch_idx+1)
                    avg_phys    = phys_loss  / (batch_idx+1)
                    avg_smooth  = smooth_loss_accum / (batch_idx+1)
                    vloop.set_postfix(
                        vloss=avg_vloss,
                        phys=avg_phys,
                        smooth=avg_smooth
                    )

            # compute per-class IoU and mean IoU
            per_class_iou = []
            for c in range(NUM_CLASSES):
                tp = conf_mat[c, c]
                fp = conf_mat[:, c].sum() - tp
                fn = conf_mat[c, :].sum() - tp
                denom = tp + fp + fn
                iou_c = tp / denom if denom > 0 else 0.0
                per_class_iou.append(iou_c)
                log_string(f"Class {c} ({seg_label_to_cat[c]}): IoU={iou_c:.4f}",os.path.join(experiment_dir, 'log.txt'))

            mean_iou = sum(per_class_iou) / NUM_CLASSES
            log_string(f"Epoch {epoch} Val mIoU: {mean_iou:.4f}", os.path.join(experiment_dir, 'log.txt'))

            # Compute averages
            avg_train_loss   = train_loss      / len(train_loader)
            avg_val_ce_loss     = val_loss        / len(val_loader)
            avg_train_ce_loss   = train_ce_loss      / len(train_loader)
            if args.physics_loss:
                avg_train_phys  = train_phys_loss      / len(train_loader)
                avg_train_smooth= train_smooth_loss  / len(train_loader)
                avg_val_phys    = phys_loss        / len(val_loader)
                avg_val_smooth  = smooth_loss_accum/ len(val_loader)
            else:
                avg_train_phys = avg_train_smooth = avg_val_phys = avg_val_smooth = 0.0

            # Write header once at start of fold 1
            if fold == 1 and epoch == 1:
                with open(METRICS_CSV, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'fold','epoch',
                        'train_loss','avg_train_ce_loss','train_phys','train_smooth',
                        'val_ce_loss','val_phys','val_smooth',
                        'val_mIoU'
                    ])

            # Append this epoch�s metrics
            with open(METRICS_CSV, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    fold, epoch,
                    f"{avg_train_loss:.6f}",
                    f"{avg_train_ce_loss:.6f}",
                    f"{avg_train_phys:.6f}",
                    f"{avg_train_smooth:.6f}",
                    f"{avg_val_ce_loss:.6f}",
                    f"{avg_val_phys:.6f}",
                    f"{avg_val_smooth:.6f}",
                    f"{mean_iou:.4f}"
                ])

            # Optionally echo a summary to your log.txt
            log_string(
                f"Metrics -> Fold{fold} Ep{epoch}: "
                f"Train L={avg_train_loss:.4f} CE={avg_train_ce_loss:.6f} P={avg_train_phys:.4f} S={avg_train_smooth:.4f} | "
                f"Val   L={avg_val_ce_loss:.4f} P={avg_val_phys:.4f} S={avg_val_smooth:.4f} | "
                f"mIoU={mean_iou:.4f}",
                os.path.join(experiment_dir, 'log.txt')
            )

            # Save best
            if mean_iou > best_iou:
                best_iou = mean_iou
                state = {'epoch': epoch,
                         'model_state_dict': classifier.state_dict()}
                torch.save(state, os.path.join(experiment_dir, 'checkpoints', f'best_model_fold_{fold}.pth'))
                log_string(f'Fold{fold} Epoch {epoch}: Saved best model (mIoU {best_iou:.4f})', os.path.join(experiment_dir, 'log.txt'))

            
            


if __name__ == '__main__':
    main()
