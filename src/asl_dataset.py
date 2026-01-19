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
import pandas as pd

class RGBDSkel_Dataset(Dataset):
    def __init__(
        self,
        annotations,
        processor: VideoMAEImageProcessor,
        num_frames=16,
        modalities=("rgb", "depth", "skeleton"),
        use_tslformer_joints=False  # NEW: Enable TSLFormer joint selection (543 → 50)
    ):
        self.annotations = self._read_annotations(annotations)
        self.processor = processor
        self.num_frames = num_frames
        self.modalities = modalities
        self.use_tslformer_joints = use_tslformer_joints

        # Import joint selection utility if needed
        if self.use_tslformer_joints:
            from tslformer_joint_selection import select_tslformer_joints
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
        1. Extract only x, y coordinates (drop z and visibility)
        2. Interpolate short gaps (≤5 frames)
        3. Normalize coordinates (mean-center and scale)
        4. Sample to target number of frames
        """
        keypoints = np.load(path)

        # Filter last dimension to only x, y values
        keypoints = keypoints[:, :, :2]  # (T, 543, 2)

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
            mean = mean.reshape(1, 1, 2)  # (1, 1, 2) for broadcasting

            # Subtract mean (only from valid coordinates)
            keypoints = np.where(valid_mask, keypoints - mean, 0.0)

            # Compute std only over valid coordinates
            std = np.sqrt(np.where(valid_mask, keypoints ** 2, 0).sum(axis=(0, 1)) / (valid_mask.sum(axis=(0, 1)) + 1e-8))
            std = std.reshape(1, 1, 2) + 1e-8  # Add epsilon to avoid division by zero

            # Scale to unit variance (only valid coordinates)
            keypoints = np.where(valid_mask, keypoints / std, 0.0)

        # Apply TSLFormer joint selection if enabled (543 → 50 joints)
        if self.use_tslformer_joints and self.joint_selector is not None:
            keypoints = self.joint_selector(keypoints)  # (T, 50, 2)

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