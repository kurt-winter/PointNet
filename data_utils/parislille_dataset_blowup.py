import numpy as np
import os
import torch
from torch.utils.data import Dataset
import glob

class ParisLilleDataset(Dataset):
    def __init__(self, root_dir, split='train', augment=False,
                 split_ratio=0.8, num_point=4096, block_size=1.0,
                 mixup_prob=0.5, mixup_alpha=0.2, num_classes=10):
        self.root_dir    = root_dir
        self.num_point   = num_point
        self.block_size  = block_size
        self.split       = split
        self.split_ratio = split_ratio
        self.augment     = augment

        self.mixup_prob  = mixup_prob
        self.mixup_alpha = mixup_alpha
        self.num_classes = num_classes

        # Gather all blocks
        all_data_paths  = sorted(glob.glob(os.path.join(root_dir, "*_xyz.npy")))
        all_label_paths = [p.replace('_xyz.npy', '_labels.npy') for p in all_data_paths]

        # Split train / val
        n_total = len(all_data_paths)
        n_train = int(n_total * split_ratio)
        if split == 'train':
            self.data_paths  = all_data_paths[:n_train]
            self.label_paths = all_label_paths[:n_train]
        else:
            self.data_paths  = all_data_paths[n_train:]
            self.label_paths = all_label_paths[n_train:]

        # === block sampling weights: oversample blocks that contain any "rare" class ===
        # rare = everything except ground(1) and building(2)
        rare_labels = {3,4,5,6,7,8,9}
        block_weights = []
        for lbl_path in self.label_paths:
            lbls = np.load(lbl_path)
            has_rare = bool(np.isin(lbls, list(rare_labels)).any())
            block_weights.append(5.0 if has_rare else 1.0)
        self.block_weights = torch.DoubleTensor(block_weights)



        # === per-class label weights for loss ===
        # count labels over entire split
        label_hist = np.zeros(self.num_classes, dtype=np.float64)
        for lbl_path in self.label_paths:
            lbls = np.load(lbl_path)
            label_hist += np.bincount(lbls, minlength=self.num_classes)

        label_prob = label_hist / label_hist.sum()
        #----- baseline: inverse-sqrt reweighting
        #weights = np.power(label_prob.max() / (label_prob + 1e-6), 0.5)
        # cube root
        #----- self.labelweights = np.power(np.amax(label_prob) / (label_prob + 1e-6), 1/3.0)
        #----- smoother inverse-sqrt reweighting
        #self.labelweights = np.power(np.amax(label_prob) / (label_prob + 1e-6), 0.5)
        #------ debug
        #self.labelweights = np.ones(self.num_classes, dtype=np.float32)
        #------ New: log-scaling the imbalance ratio --
        ratio = np.amax(label_prob) / (label_prob + 1e-6)   # [num_classes]
        w = np.log1p(ratio)                                 # softens large ratios
        w = w / np.mean(w)                                 # normalize so mean=1
        # clip to [0.5, 2.0] to prevent any class dominating
        self.labelweights = np.clip(w, a_min=0.5, a_max=2.0)
        

        print(f"? labelweights: min={self.labelweights.min():.2f}, "
              f"max={self.labelweights.max():.2f}, "
              f"mean={self.labelweights.mean():.2f}")

        '''
        # extra boost for the truly tiny classes:
        RARE_BOOST = {
            3:  2.0,   # pole
            4: 10.0,   # bollard
            5:  8.0,   # trash can
            6:  2.5,   # barrier
            7: 12.0,   # pedestrian
            8:  1.5,   # car
            9:  1.5,   # natural
        }
        for cls_id, boost in RARE_BOOST.items():
            weights[cls_id] *= boost
        

        # finalize: sqrt, clip to [1,15], normalize to mean=1
        weights = np.sqrt(weights)
        weights = np.clip(weights, 1.0, 15.0)
        weights /= weights.mean()
        '''

        #self.labelweights = weights.astype(np.float32)

        '''
        # === block sampling weights: proportional to avg per-point rarity ===
        # (i.e. average the class weight of each point in the block)
        block_weights = []
        for lbl_path in self.label_paths:
            lbls = np.load(lbl_path)               # shape [4096,]
            # look up the labelweight for each point, then average
            block_weights.append(self.labelweights[lbls].mean())
        # normalize to mean=1 so your sampler stays well-behaved
        bw = np.array(block_weights, dtype=np.float64)
        bw = bw / bw.mean()
        self.block_weights = torch.DoubleTensor(bw)
        '''

        print(f"Dataset init done: {len(self.data_paths)} blocks, "
              f"labelweights sum={self.labelweights.sum():.2f}")


    def __len__(self):
        return len(self.data_paths)

    '''def __getitem__(self, idx):
        xyz_file = self.data_paths[idx]
        label_file = self.label_paths[idx]

        points = np.load(xyz_file).astype(np.float32) # shape [N,3] (or [N,6] if reflectance)
        labels = np.load(label_file).astype(np.int64) # shape [N]

        if self.augment:
            points, labels = self._augment(points, labels)

        return torch.from_numpy(points), torch.from_numpy(labels)'''
    
    def __getitem__(self, idx):
        # 1) load raw block
        pts   = np.load(self.data_paths[idx]).astype(np.float32)   # (N, 3+feat)   ----- with reflectance (N, 4)!
        labels = np.load(self.label_paths[idx]).astype(np.int64)   # (N,)

        # 2) optional augment in place (rot/scale/jitter/etc + block_dropout sets some labels=255)
        if self.augment:
            pts, labels = self._augment(pts, labels)

        # 3) now sample exactly self.num_point points (so network always sees fixed-size input)
        N = pts.shape[0]
        if N >= self.num_point:
            choice = np.random.choice(N, self.num_point, replace=False)
        else:
            choice = np.random.choice(N, self.num_point, replace=True)
        pts    = pts[choice]
        labels = labels[choice]

        # 4) normalize/scale
        # pts: (4096,4)  columns = [x,y,z,reflectance]
        xyz   = pts[:, :3]        # world coords
        refl  = pts[:, 3:]        # raw 0-255

        
            # a) center the XYZ block
        centroid = xyz.mean(axis=0, keepdims=True)        # shape (1,3)
        xyz = xyz - centroid

            # b) normalize by the physical extents:
            #    we know blocks are 1 m x 1 m in XY and maybe ~5 m tall in Z
        block_size = 1.0       # meters  (must match your preproc)
        max_height = 5.0      # meters  (or whatever your data actually spans)

        xyz[:, 0:2] /= block_size        # block_size in X/Y
        xyz[:,   2] /= max_height  # pick e.g. max_height=5.0

            # c) scale reflectance from [0,255] to [0,1]
        refl = refl / 255.0

            # d) re-stitch
        pts = np.concatenate([xyz, refl], axis=1).astype(np.float32)

        # 5) convert to tensors
        pts    = torch.from_numpy(pts).float()    # (self.num_point, 3+feat)
        labels = torch.from_numpy(labels).long()  # (self.num_point,)

        return pts, labels

    def _random_occlusion(self, pts, occ_size=0.5):
        # pick a random center in the block
        center = np.random.uniform(low=pts[:, :3].min(axis=0),
                                high=pts[:, :3].max(axis=0))
        # carve out a cube of side occ_size
        mask = np.all(np.abs(pts[:, :3] - center) < (occ_size/2), axis=1)
        pts[mask, :3] = center  # collapse occluded points into the center
        return pts

    def _random_flip_xy(self, pts):
        # with 50% chance, mirror X?-X
        if np.random.rand() < .5:
            pts[:,0] = -pts[:,0]
        return pts

    def _elastic_distortion(self, pts, gran=6, mag=0.2):
        # only look at xyz for generating the distortion field
        xyz = pts[:, :3]

        # build the noise grid
        blurx = np.ones((3,1,1))/3
        blury = np.ones((1,3,1))/3
        blurz = np.ones((1,1,3))/3
        bb = (np.abs(xyz).max(axis=0).astype(int) // gran) + 3
        noise = [np.random.randn(*bb) for _ in range(3)]
        for i in range(3):
            for _ in range(2):
                noise[i] = np.apply_along_axis(lambda m: np.convolve(m, blurx.flatten(), mode='same'), 0, noise[i])
                noise[i] = np.apply_along_axis(lambda m: np.convolve(m, blury.flatten(), mode='same'), 1, noise[i])
                noise[i] = np.apply_along_axis(lambda m: np.convolve(m, blurz.flatten(), mode='same'), 2, noise[i])

        # now trilinearly interpolate the noise at each xyz
        coords = (xyz / gran) + 1  # shift into noise grid
        ix, iy, iz = coords.T.astype(int)  # integer lookup indices

        disp = np.stack([
            noise[0][ix, iy, iz],
            noise[1][ix, iy, iz],
            noise[2][ix, iy, iz],
        ], axis=1) * mag

        # apply only to the xyz channels
        pts[:, :3] += disp
        return pts


    def _random_block_dropout(self, pts, labs, drop_ratio=0.5, block_size=1.0):
        # Sample a random cube within your point-cloud�s bbox
        N = pts.shape[0]
        xs, ys = pts[:,0], pts[:,1]
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()

        # carve out one random block per call
        cx = np.random.uniform(min_x, max_x)
        cy = np.random.uniform(min_y, max_y)
        mask = (
            (pts[:,0] >= cx-block_size/2) & (pts[:,0] <= cx+block_size/2) &
            (pts[:,1] >= cy-block_size/2) & (pts[:,1] <= cy+block_size/2)
        )
        # set those to IGNORE
        labs[mask] = 255

        # now _always_ sample exactly self.num_point points from the full cloud,
        # so your network never sees a variable-sized batch.
        idxs = np.random.choice(
            np.arange(pts.shape[0]),
            self.num_point,
            replace=(pts.shape[0] < self.num_point)
        )
        return pts[idxs], labs[idxs]



    def _augment(self, pts, labs):
            
            # 1) random yaw rotation about Z
            theta = np.random.uniform(0, 2*np.pi)
            c, s  = np.cos(theta), np.sin(theta)
            R     = np.array([[ c, -s, 0],
                            [ s,  c, 0],
                            [ 0,  0, 1]], dtype=np.float32)
            pts[:, :3] = pts[:, :3] @ R.T

            # 2) random scale
            scale = np.random.uniform(0.9, 1.1)
            pts[:, :3] *= scale

            # 3) jitter (gaussian noise)
            jitter = np.clip(0.01 * np.random.randn(*pts[:, :3].shape), -0.05, 0.05)
            pts[:, :3] += jitter

            # 4) small random translation
            translation = np.random.uniform(-0.05, 0.05, size=(1,3)).astype(np.float32)
            pts[:, :3] += translation

            # 5) random point dropout (set some to zero)
            drop_rate = 0.05
            mask = np.random.rand(pts.shape[0]) < drop_rate
            pts[mask, :3] = pts[0, :3]   # �duplicate� the first point instead of zeroing
                                        # (so labels stay in range
                                        
            # 5a) horizontal flip ?
            if np.random.rand() < 0.5:
                pts = self._random_flip_xy(pts)

            ## random occlusion (20% chance)
            if np.random.rand() < 0.2:
                pts = self._random_occlusion(pts)

            ## 5b) elastic warp   ?
            #if np.random.rand()<0.3:
            #    pts = self._elastic_distortion(pts, gran=6, mag=0.1)

            # 5c) block dropout  ?
            #if np.random.rand()<0.2:
            #    pts, labs = self._random_block_dropout(pts, labs, drop_ratio=0.1, block_size=0.5)       

            return pts, labs
    