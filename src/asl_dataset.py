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
        modalities=("rgb", "depth", "skeleton")
    ):
        self.annotations = self._read_annotations(annotations)
        self.processor = processor
        self.num_frames = num_frames
        self.modalities = modalities

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
        #TODO (Coulson) edit Nan numbers to coordinate outside grid boundaries (or whatever is standard practice)

        #TODO (Coulson) keep only x, y values (drop depth and model values)
        keypoints = np.load(path)

        # filter last dimension to only x, y values
        keypoints = keypoints[:, :, :2]
        # should we use interpolation to fill in missing values, should we ?
        interpolated_keypoints = self.interpolate_with_gaps(keypoints, max_gap=5, sentinel=999.0)
        keypoints = interpolated_keypoints
        # change Nan values to 999 (to represent points outside grid boundaries)
        # keypoints = np.where(np.isnan(keypoints), 999, keypoints)

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