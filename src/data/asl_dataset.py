from torch.utils.data import Dataset
from tqdm import tqdm
from pytorch_lightning.utilities import rank_zero_info
import csv
import decord
from decord import VideoReader, cpu
import torch
import numpy as np
from transformers import VideoMAEImageProcessor
import os
import math
import pandas as pd


# ── Skeleton augmentation transforms ──────────────────────────────────────────

def augment_skeleton_spatial(keypoints, scale_range=(0.9, 1.1), rotation_deg=15.0,
                             translate_range=0.1):
    """Apply random scale, rotation, and translation to 2D skeleton coordinates.

    Args:
        keypoints: (T, J, C) numpy array, C >= 2 (x, y, ...)
        scale_range: (min, max) uniform scale factor
        rotation_deg: max absolute rotation in degrees
        translate_range: max absolute translation (coords are ~0-centered after norm)
    Returns:
        augmented keypoints, same shape
    """
    kp = keypoints.copy()
    valid_mask = (kp != 0.0)

    # Random uniform scale
    scale = np.random.uniform(*scale_range)
    kp[..., :2] *= scale

    # Random rotation (applied to x, y only)
    angle = np.random.uniform(-rotation_deg, rotation_deg) * math.pi / 180.0
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    x = kp[..., 0].copy()
    y = kp[..., 1].copy()
    kp[..., 0] = cos_a * x - sin_a * y
    kp[..., 1] = sin_a * x + cos_a * y

    # Random translation
    tx = np.random.uniform(-translate_range, translate_range)
    ty = np.random.uniform(-translate_range, translate_range)
    kp[..., 0] += tx
    kp[..., 1] += ty

    # Restore sentinel zeros
    kp = np.where(valid_mask, kp, 0.0)
    return kp


def augment_skeleton_temporal(keypoints, speed_range=(0.8, 1.2)):
    """Temporal speed perturbation via resampling.

    Simulates faster/slower signing by stretching or compressing the time axis
    then resampling back to the original length.

    Args:
        keypoints: (T, J, C) numpy array
        speed_range: (min, max) speed factor (>1 = faster = fewer unique frames)
    Returns:
        augmented keypoints, same shape (T, J, C)
    """
    T, J, C = keypoints.shape
    speed = np.random.uniform(*speed_range)
    new_T = max(2, int(round(T / speed)))
    # Resample to new_T frames then back to T
    indices_to_new = np.linspace(0, T - 1, new_T).astype(int)
    stretched = keypoints[indices_to_new]  # (new_T, J, C)
    indices_back = np.linspace(0, new_T - 1, T).astype(int)
    return stretched[indices_back]


def augment_skeleton_joint_noise(keypoints, noise_std=0.02):
    """Add small Gaussian noise to each joint coordinate.

    Args:
        keypoints: (T, J, C) numpy array
        noise_std: standard deviation of Gaussian noise
    Returns:
        augmented keypoints, same shape
    """
    kp = keypoints.copy()
    valid_mask = (kp != 0.0)
    noise = np.random.randn(*kp.shape).astype(kp.dtype) * noise_std
    kp = kp + noise
    kp = np.where(valid_mask, kp, 0.0)
    return kp


class RGBDSkel_Dataset(Dataset):
    def __init__(
        self,
        annotations,
        processor: VideoMAEImageProcessor,
        num_frames=16,
        modalities=("rgb", "depth", "skeleton"),
        use_tslformer_joints=False,  # Enable TSLFormer joint selection (543 → 50)
        use_z_coord=False,  # Include Z coordinate (3D) instead of just X, Y (2D)
        selected_joint_indices=None,  # Custom joint index selection (list of 543-space indices)
        augment_config=None,  # Dict of augmentation settings (None = no augmentation)
        cache_skeletons=False,  # Memoize preprocessed clips (safe only without augmentation)
        normalization_scope="global",  # "global" | "per_group" | "per_landmark"
    ):
        self.annotations = self._read_annotations(annotations)
        self.processor = processor
        self.num_frames = num_frames
        self.modalities = modalities
        self.use_tslformer_joints = use_tslformer_joints
        self.use_z_coord = use_z_coord
        self.num_coords = 3 if use_z_coord else 2
        self.selected_joint_indices = selected_joint_indices
        self.augment_config = augment_config or {}

        # Scope of the standardization statistics (see _standardize).
        #   "global"       - one mean/std per coordinate axis over all frames and all landmarks.
        #                    Original behaviour; kept as the default so every prior run stays
        #                    reproducible bit-for-bit.
        #   "per_group"    - one mean/std per coordinate axis per MediaPipe anatomical group.
        #                    Puts face, pose and hand landmarks on a common amplitude scale, so
        #                    the L0 gate compares them by their own variation rather than by raw
        #                    motion magnitude (face landmarks move ~13x less than pose landmarks).
        #   "per_landmark" - one mean/std per landmark per axis, over time only. Fully removes
        #                    amplitude differences. Diagnostic; also discards absolute location.
        #   "subset"       - global-style statistics, but pooled over ONLY the landmarks that
        #                    `selected_joint_indices` retains. Equivalent to subsetting first and
        #                    then standardizing, while leaving the pipeline order (and therefore
        #                    augmentation semantics) untouched. Without this, a K-landmark model
        #                    receives inputs centred and scaled by statistics belonging to the
        #                    543-K landmarks that were discarded: measured on the real K=24 subset
        #                    the encoder sees mean (+1.17, +1.57) and std (2.21, 1.16) instead of
        #                    0 and 1.
        assert normalization_scope in ("global", "per_group", "per_landmark", "subset"), \
            f"unknown normalization_scope: {normalization_scope}"
        assert normalization_scope != "subset" or selected_joint_indices is not None, \
            "normalization_scope='subset' requires selected_joint_indices"
        self.normalization_scope = normalization_scope

        # Skeleton preprocessing (gap-fill + normalize + joint select + frame sample) is
        # deterministic per clip, so a validation set is recomputed identically on every
        # pass. On ASL Citizen that is 51% of all clip preprocessing in the run -- its val
        # split is 10,304 clips and val_interval=0.25 replays it four times an epoch.
        # Caching is only correct when nothing random is applied, so it self-disables
        # if any augmentation is configured.
        self.cache_skeletons = bool(cache_skeletons) and not self.augment_config
        if cache_skeletons and self.augment_config:
            print("cache_skeletons requested but augmentation is active; caching disabled "
                  "(augmented clips must differ between epochs).")
        self._skel_cache = {}

        # Mutually exclusive: can't use both TSLFormer and custom selection
        assert not (use_tslformer_joints and selected_joint_indices is not None), \
            "Cannot use both use_tslformer_joints and selected_joint_indices"

        # Import joint selection utility if needed
        if self.use_tslformer_joints:
            from data.tslformer_joint_selection import select_tslformer_joints
            self.joint_selector = select_tslformer_joints
        else:
            self.joint_selector = None

    def _read_annotations(self, csv_f):
        with open(csv_f, newline='') as inf:
            rows = [tuple(r) for r in csv.reader(inf)]
        header = rows[0]
        assert header == ("rgb_path", "depth_path", "skel_path", "label")
        data = rows[1:]
        return data
    
    def _load_video(self, path, assert_frames=3):
        # print(f"DOES {path} EXIST?", os.path.exists(path))
        vr = VideoReader(path, ctx=cpu(0))
        # vr = VideoReader(path, ctx=cpu())
        total_frames = len(vr)
        indices = torch.linspace(0, total_frames - 1, self.num_frames).long()
        frames = vr.get_batch(indices).asnumpy()
        assert frames.shape[-1] == assert_frames, f"frames.shape: {frames.shape}, assert_frames: {assert_frames}"
        return list(frames), total_frames


    def interpolate_with_gaps(self, pose_data, max_gap=3, sentinel=999.0):
        pose_data = pose_data.copy()

        # Only (landmark, feature) columns that actually contain a NaN need touching.
        # Building a pandas Series for all 543x2 columns cost ~96 ms/clip and dominated
        # training wall-clock; face and pose landmarks are ~0% missing, so the vast
        # majority of that work was on NaN-free columns where interpolate + fillna are
        # both no-ops. Numerically identical to the per-column loop it replaces.
        needs_fill = np.isnan(pose_data).any(axis=0)  # (L, F)

        for lm, feat in zip(*np.nonzero(needs_fill)):
            s = pd.Series(pose_data[:, lm, feat])
            # only fill NaN runs of length <= max_gap
            s = s.interpolate(
                method='linear',
                limit=max_gap,
                limit_direction='both'
            )
            # very long gaps remain NaN → turn them into sentinel
            s = s.fillna(sentinel)
            pose_data[:, lm, feat] = s.values

        return pose_data


    def _load_skeleton(self, path):
        if self.cache_skeletons:
            hit = self._skel_cache.get(path)
            if hit is not None:
                # Clone so a downstream in-place op cannot corrupt the cached copy.
                return hit[0].clone(), hit[1]
            result = self._load_skeleton_uncached(path)
            self._skel_cache[path] = result
            return result[0].clone(), result[1]
        return self._load_skeleton_uncached(path)

    # MediaPipe Holistic landmark blocks: face, pose, left hand, right hand.
    LANDMARK_GROUPS = ((0, 468), (468, 501), (501, 522), (522, 543))

    def _standardize(self, keypoints):
        """Mean-center and scale to unit variance, at the configured scope.

        Sentinel zeros (frames MediaPipe never filled) are excluded from the statistics
        and written back as zeros, so a missing landmark stays missing.

        Args:
            keypoints: (T, J, C) array
        Returns:
            standardized (T, J, C) array
        """
        valid_mask = (keypoints != 0.0)
        if valid_mask.sum() == 0:
            return keypoints

        if self.normalization_scope == "subset":
            # Pool statistics over the retained landmarks only, then apply to the whole array;
            # the discarded columns are dropped downstream, so this is identical to subsetting
            # first and standardizing, but keeps augmentation operating on the same
            # representation as every other scope.
            sel = np.asarray(self.selected_joint_indices, dtype=int)
            sub, sub_mask = keypoints[:, sel, :], valid_mask[:, sel, :]
            if sub_mask.sum() == 0:
                return np.where(valid_mask, keypoints, 0.0)
            n = sub_mask.sum(axis=(0, 1)) + 1e-8
            mean = (np.where(sub_mask, sub, 0).sum(axis=(0, 1)) / n).reshape(1, 1, self.num_coords)
            centered_sub = np.where(sub_mask, sub - mean, 0.0)
            std = np.sqrt(np.where(sub_mask, centered_sub ** 2, 0).sum(axis=(0, 1)) / n)
            std = std.reshape(1, 1, self.num_coords) + 1e-8
            return np.where(valid_mask, (keypoints - mean) / std, 0.0)

        if self.normalization_scope == "global":
            # Original path. Statistics pooled over every frame and every landmark, so a
            # single scalar per axis rescales all J landmarks. Note this preserves their
            # relative motion magnitudes exactly -- it is a common factor.
            axes, shape = (0, 1), (1, 1, self.num_coords)
        elif self.normalization_scope == "per_landmark":
            axes, shape = (0,), (1, keypoints.shape[1], self.num_coords)
        else:  # per_group -- handled blockwise below
            out = keypoints.copy()
            for start, end in self.LANDMARK_GROUPS:
                if start >= keypoints.shape[1]:
                    break
                stop = min(end, keypoints.shape[1])
                block = keypoints[:, start:stop, :]
                mask = valid_mask[:, start:stop, :]
                if mask.sum() == 0:
                    out[:, start:stop, :] = 0.0
                    continue
                n = mask.sum(axis=(0, 1)) + 1e-8
                mean = (np.where(mask, block, 0).sum(axis=(0, 1)) / n).reshape(1, 1, self.num_coords)
                centered = np.where(mask, block - mean, 0.0)
                std = np.sqrt(np.where(mask, centered ** 2, 0).sum(axis=(0, 1)) / n)
                std = std.reshape(1, 1, self.num_coords) + 1e-8
                out[:, start:stop, :] = np.where(mask, centered / std, 0.0)
            return out

        n = valid_mask.sum(axis=axes) + 1e-8
        mean = (np.where(valid_mask, keypoints, 0).sum(axis=axes) / n).reshape(shape)
        keypoints = np.where(valid_mask, keypoints - mean, 0.0)
        std = np.sqrt(np.where(valid_mask, keypoints ** 2, 0).sum(axis=axes) / n)
        std = std.reshape(shape) + 1e-8
        return np.where(valid_mask, keypoints / std, 0.0)

    def _load_skeleton_uncached(self, path):
        """
        Load skeleton keypoints with preprocessing matching TSLFormer:
        1. Extract x, y (and optionally z) coordinates
        2. Interpolate short gaps (≤5 frames)
        3. Normalize coordinates (mean-center and scale)
        4. Sample to target number of frames
        """
        keypoints = np.load(path)

        # Filter last dimension to x, y (and optionally z) values
        # Raw format: (T, 543, 4) with [x, y, z, visibility]
        keypoints = keypoints[:, :, :self.num_coords]  # (T, 543, 2) or (T, 543, 3)

        # Interpolate short gaps with linear interpolation
        interpolated_keypoints = self.interpolate_with_gaps(keypoints, max_gap=5, sentinel=0.0)
        keypoints = interpolated_keypoints

        # Normalize coordinates (TSLFormer does this)
        # MediaPipe outputs are already in [0, 1] range, but we mean-center and scale
        # to have zero mean and unit variance per sequence for better model convergence
        keypoints = self._standardize(keypoints)

        # Apply skeleton augmentation (training only — caller sets augment_config)
        if self.augment_config.get("spatial", False):
            keypoints = augment_skeleton_spatial(
                keypoints,
                scale_range=self.augment_config.get("scale_range", (0.9, 1.1)),
                rotation_deg=self.augment_config.get("rotation_deg", 15.0),
                translate_range=self.augment_config.get("translate_range", 0.1),
            )
        if self.augment_config.get("temporal", False):
            keypoints = augment_skeleton_temporal(
                keypoints,
                speed_range=self.augment_config.get("speed_range", (0.8, 1.2)),
            )
        if self.augment_config.get("joint_noise", False):
            keypoints = augment_skeleton_joint_noise(
                keypoints,
                noise_std=self.augment_config.get("noise_std", 0.02),
            )

        # Apply TSLFormer joint selection if enabled (543 → 50 joints)
        if self.use_tslformer_joints and self.joint_selector is not None:
            keypoints = self.joint_selector(keypoints)  # (T, 50, num_coords)

        # Apply custom joint index selection if provided
        if self.selected_joint_indices is not None:
            keypoints = keypoints[:, self.selected_joint_indices, :]

        # Sample to target number of frames
        total_frames = keypoints.shape[0]
        indices = torch.linspace(0, total_frames - 1, self.num_frames).long()
        sampled = keypoints[indices]

        return torch.tensor(sampled, dtype=torch.float32), total_frames

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        rgb_path, depth_path, skel_path, label = self.annotations[idx]
        assert label.isdecimal()
        label = int(label)
        assert isinstance(label, int), "Dataset: Label {label} is not an integer (idx={idx})."
        output = {}

        rgb_len = depth_len = skel_len = None

        # RGB
        if "rgb" in self.modalities and rgb_path.strip() != "":
            # print("rgb_path:", rgb_path)
            rgb_frames, rgb_len = self._load_video(rgb_path)
            processed = self.processor(rgb_frames, return_tensors="pt")
            output["pixel_values"] = processed["pixel_values"].squeeze(0)

        # Depth
        if "depth" in self.modalities and depth_path.strip() != "":
            # print("depth path:", depth_path)
            depth_frames, depth_len = self._load_video(depth_path 
                                                    #,    assert_frames=1
                                                       )
            processed = self.processor(depth_frames, return_tensors="pt")
            output["depth_values"] = processed["pixel_values"].squeeze(0)
        
        # Skeleton
        if "skeleton" in self.modalities and skel_path.strip() != "":
            output["skeleton_keypoints"], skel_len = self._load_skeleton(skel_path)

        modality_lens = [l for l in (rgb_len, depth_len, skel_len) if l != None]
        for i, l in enumerate(modality_lens):
            assert l == modality_lens[0], f"Dataset: Not all modality lengths are equal. Modality {i} has length {l}, should be {modality_lens[0]}."
        
        # Label
        output["labels"] = torch.tensor(label, dtype=torch.long)

        return output