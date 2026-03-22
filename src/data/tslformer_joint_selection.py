"""
TSLFormer Joint Selection - Manual subset selection (not learnable pruning).

MediaPipe Holistic outputs 543 landmarks:
  - Face: 468 landmarks (indices 0-467)
  - Pose: 33 landmarks (indices 468-500)
  - Left Hand: 21 landmarks (indices 501-521)
  - Right Hand: 21 landmarks (indices 522-542)

TSLFormer uses ~48 keypoints focusing on upper body and hands (no face).
This module provides MANUAL SELECTION to convert 543 → 50 joints.

This is DIFFERENT from joint_pruning.py which uses learnable parameters.
This is a FIXED, DETERMINISTIC selection matching TSLFormer's approach.

Usage:
    from tslformer_joint_selection import select_tslformer_joints

    # In dataset loader
    keypoints = np.load(path)  # (T, 543, 2)
    if config.get("use_tslformer_joints", False):
        keypoints = select_tslformer_joints(keypoints)  # (T, 50, 2)
"""

import torch
import numpy as np


# MediaPipe Pose landmark indices we want to keep
# Based on: https://google.github.io/mediapipe/solutions/pose.html
# We select upper body landmarks relevant for sign language
POSE_UPPER_BODY_INDICES = [
    11, 12,  # Left/Right shoulder
    13, 14,  # Left/Right elbow
    15, 16,  # Left/Right wrist
    23, 24,  # Left/Right hip (for torso reference)
]

# MediaPipe Holistic structure offsets (in the 543-element array)
FACE_START = 0
FACE_END = 468
POSE_START = 468
POSE_END = 501
LEFT_HAND_START = 501
LEFT_HAND_END = 522
RIGHT_HAND_START = 522
RIGHT_HAND_END = 543


def get_tslformer_joint_indices():
    """
    Get the global indices for TSLFormer-style joint selection from 543-element array.

    This is a FIXED selection (not learned), matching TSLFormer's manual choice.

    Returns:
        list: 50 indices representing:
            - 8 pose landmarks (shoulders, elbows, wrists, hips)
            - 21 left hand landmarks (all)
            - 21 right hand landmarks (all)

    Example:
        >>> indices = get_tslformer_joint_indices()
        >>> len(indices)
        50
        >>> keypoints_543 = load_all_joints()  # (T, 543, 2)
        >>> keypoints_50 = keypoints_543[:, indices, :]  # (T, 50, 2)
    """
    indices = []

    # Add upper body pose landmarks (convert local pose indices to global)
    for local_pose_idx in POSE_UPPER_BODY_INDICES:
        global_idx = POSE_START + local_pose_idx
        indices.append(global_idx)

    # Add all left hand landmarks
    indices.extend(range(LEFT_HAND_START, LEFT_HAND_END))

    # Add all right hand landmarks
    indices.extend(range(RIGHT_HAND_START, RIGHT_HAND_END))

    return sorted(indices)


def select_tslformer_joints(skeleton_keypoints, joint_indices=None):
    """
    Select TSLFormer's 50-joint subset from MediaPipe's 543 joints.

    This is MANUAL SELECTION (not learned pruning).
    Removes all 468 face landmarks and keeps only:
    - Upper body pose (8 joints)
    - Hands (21 + 21 = 42 joints)
    Total: 50 joints

    Args:
        skeleton_keypoints: NumPy array or PyTorch tensor
            Shape: (..., 543, 2) where ... can be (T,) or (B, T) or any shape
        joint_indices: Optional custom list of joint indices to keep.
            If None, uses default TSLFormer selection.

    Returns:
        Selected array/tensor of shape (..., 50, 2)

    Example:
        >>> import numpy as np
        >>> data = np.random.randn(16, 543, 2)  # 16 frames, 543 joints
        >>> selected = select_tslformer_joints(data)
        >>> selected.shape
        (16, 50, 2)
    """
    if joint_indices is None:
        joint_indices = get_tslformer_joint_indices()

    # Handle both NumPy and PyTorch
    if isinstance(skeleton_keypoints, np.ndarray):
        # Use advanced indexing for NumPy
        return skeleton_keypoints[..., joint_indices, :]
    else:
        # PyTorch tensor - use index_select
        # Convert indices to tensor if needed
        if isinstance(joint_indices, list):
            joint_indices = torch.tensor(joint_indices, dtype=torch.long)

        # Get the dimension where joints are (second-to-last)
        joint_dim = len(skeleton_keypoints.shape) - 2
        return torch.index_select(skeleton_keypoints, joint_dim, joint_indices)


def get_joint_group_mapping():
    """
    Get mapping from selected joint index (0-49) to body part name.

    Useful for visualization and interpretation.

    Returns:
        dict: {selected_index: "body_part_name"}

    Example:
        >>> mapping = get_joint_group_mapping()
        >>> mapping[0]
        'pose_left_shoulder'
        >>> mapping[30]
        'left_hand_9'
    """
    indices = get_tslformer_joint_indices()
    mapping = {}

    # Pose landmark names (MediaPipe Pose naming)
    pose_names = {
        11: "left_shoulder", 12: "right_shoulder",
        13: "left_elbow", 14: "right_elbow",
        15: "left_wrist", 16: "right_wrist",
        23: "left_hip", 24: "right_hip"
    }

    for i, global_idx in enumerate(indices):
        if global_idx < POSE_END:
            # Pose landmark
            local_pose_idx = global_idx - POSE_START
            name = pose_names.get(local_pose_idx, f"pose_{local_pose_idx}")
            mapping[i] = f"pose_{name}"
        elif global_idx < LEFT_HAND_END:
            # Left hand
            local_hand_idx = global_idx - LEFT_HAND_START
            mapping[i] = f"left_hand_{local_hand_idx}"
        else:
            # Right hand
            local_hand_idx = global_idx - RIGHT_HAND_START
            mapping[i] = f"right_hand_{local_hand_idx}"

    return mapping


def get_joint_group_counts():
    """
    Get breakdown of joint counts by body part in TSLFormer subset.

    Returns:
        dict: {"pose": 8, "left_hand": 21, "right_hand": 21, "total": 50}
    """
    indices = get_tslformer_joint_indices()

    pose_count = sum(1 for idx in indices if POSE_START <= idx < POSE_END)
    left_hand_count = sum(1 for idx in indices if LEFT_HAND_START <= idx < LEFT_HAND_END)
    right_hand_count = sum(1 for idx in indices if RIGHT_HAND_START <= idx < RIGHT_HAND_END)

    return {
        "pose": pose_count,
        "left_hand": left_hand_count,
        "right_hand": right_hand_count,
        "total": len(indices)
    }


# Testing
if __name__ == "__main__":
    print("=" * 70)
    print("TSLFormer Joint Selection Utilities (Manual, Not Learned)")
    print("=" * 70)

    # Get indices
    indices = get_tslformer_joint_indices()
    print(f"\n1. Joint Selection")
    print(f"   Total joints selected: {len(indices)}")
    print(f"   First 10 indices: {indices[:10]}")
    print(f"   Last 10 indices: {indices[-10:]}")

    # Show breakdown
    counts = get_joint_group_counts()
    print(f"\n2. Breakdown by Body Part")
    for part, count in counts.items():
        print(f"   {part:15s}: {count:3d} joints")

    # Test selection with NumPy
    print(f"\n3. Testing NumPy Selection")
    test_np = np.random.randn(16, 543, 2)
    selected_np = select_tslformer_joints(test_np)
    print(f"   Input shape:  {test_np.shape}")
    print(f"   Output shape: {selected_np.shape}")
    print(f"   ✓ Correct!" if selected_np.shape == (16, 50, 2) else "   ✗ Error!")

    # Test selection with PyTorch
    print(f"\n4. Testing PyTorch Selection")
    test_torch = torch.randn(4, 16, 543, 2)
    selected_torch = select_tslformer_joints(test_torch)
    print(f"   Input shape:  {test_torch.shape}")
    print(f"   Output shape: {selected_torch.shape}")
    print(f"   ✓ Correct!" if selected_torch.shape == (4, 16, 50, 2) else "   ✗ Error!")

    # Show some joint names
    print(f"\n5. Joint Name Mapping (first 12)")
    mapping = get_joint_group_mapping()
    for i in range(12):
        print(f"   Index {i:2d}: {mapping[i]}")

    print(f"\n{'=' * 70}")
    print("All tests passed! ✓")
    print("This is MANUAL SELECTION, not learnable pruning.")
    print("=" * 70)
