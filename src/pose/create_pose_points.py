import argparse
import os
from tqdm import tqdm

from mmpose.apis import (
    init_model,
    inference_topdown,
    inference_bottomup,
    vis_pose_result
)
from mmpose.structures import merge_data_samples
import mmcv
import cv2
import numpy as np
import os

# DEVICE="cuda:0"
DEVICE="cpu"

# 1. Load body detector (for bounding box)
det_config = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py'
det_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-6c4f3ab4_20200708.pth'
detector = init_model(det_config, det_checkpoint, device=DEVICE)

# 2. Load hand keypoint model
hand_config = 'configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/onehand10k/hrnet_w18_onehand10k_256x256.py'
hand_checkpoint = 'https://download.openmmlab.com/mmpose/hand/hrnet/hrnet_w18_onehand10k_256x256-5f9d84bf_20210909.pth'
hand_model = init_model(hand_config, hand_checkpoint, device=DEVICE)

def extract_skeleton_from_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    frame_skeletons = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Step A: detect person bbox
        det_results = inference_bottomup(detector, frame)
        det = merge_data_samples(det_results)
        if det.pred_instances.keypoints.shape[0] == 0:
            frame_skeletons.append(None)
            continue

        person_bbox = det.pred_instances.bboxes[0]  # [x1, y1, x2, y2]

        # Step B: crop hands and infer keypoints
        hand_results = inference_topdown(hand_model, frame, person_bbox)
        hand_data = merge_data_samples(hand_results)

        # keypoints is [num_hands, num_joints, 3 (x,y,score)]
        keypoints = hand_data.pred_instances.keypoints
        frame_skeletons.append(keypoints)

    cap.release()
    np.save(output_path, frame_skeletons)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", help="directory of _color.mp4 and _depth.mp4 videos. Only _color.mp4 videos will be retrieved.")
    args = parser.parse_args()
    directory = args.dir
    for f in tqdm(os.listdir(directory)):
        if f.endswith("_color.mp4"):
            f = os.path.join(directory, f)
            pp_f = f[:-4] + ".skeleton.npy"
            assert pp_f != f
            extract_skeleton_from_video(f, pp_f)
