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
        T, L, F = pose_data.shape

        for lm in range(L):
            for feat in range(F):
                s = pd.Series(pose_data[:, lm, feat])

                if s.isna().any():
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

        # Mean-center across time and joints (per coordinate dimension)
        valid_mask = (keypoints != 0.0)  # Don't include sentinel values in stats
        if valid_mask.sum() > 0:
            # Compute mean only over valid (non-sentinel) coordinates
            mean = np.where(valid_mask, keypoints, 0).sum(axis=(0, 1)) / (valid_mask.sum(axis=(0, 1)) + 1e-8)
            mean = mean.reshape(1, 1, self.num_coords)  # (1, 1, num_coords) for broadcasting

            # Subtract mean (only from valid coordinates)
            keypoints = np.where(valid_mask, keypoints - mean, 0.0)

            # Compute std only over valid coordinates
            std = np.sqrt(np.where(valid_mask, keypoints ** 2, 0).sum(axis=(0, 1)) / (valid_mask.sum(axis=(0, 1)) + 1e-8))
            std = std.reshape(1, 1, self.num_coords) + 1e-8  # Add epsilon to avoid division by zero

            # Scale to unit variance (only valid coordinates)
            keypoints = np.where(valid_mask, keypoints / std, 0.0)

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