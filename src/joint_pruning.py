import torch
import torch.nn as nn
import torch.nn.functional as F


class JointPruningModule(nn.Module):
    """
    Learnable joint pruning using Gumbel-Softmax.
    
    This module learns which joints/keypoints are important for classification.
    
    During training: soft selection (probabilistic masking)
    During inference: hard selection (binary 0/1 mask)
    
    Example:
        >>> pruner = JointPruningModule(num_joints=543)
        >>> skeleton = torch.randn(4, 16, 532, 2)  # (B, T, J, P)
        >>> pruned = pruner(skeleton)
        >>> print(f"Pruning ratio: {pruner.get_pruning_ratio():.1%}")
    """
    
    def __init__(
        self,
        num_joints: int,
        temperature: float = 1.0,
        hard: bool = False,
        init_keep_prob: float = 0.9
    ):
        """
        Args:
            num_joints: Total number of skeleton joints 
            temperature: Gumbel-Softmax temperature. Lower = sharper, higher = softer
            hard: If True, use hard Gumbel-Softmax (discrete selection)
            init_keep_prob: Initial probability to keep each joint (0.0-1.0)
        """
        super().__init__()
        self.num_joints = num_joints
        self.temperature = temperature
        self.hard = hard
        
        # Learnable logits: one scalar per joint
        # Initialize so that log-odds corresponds to init_keep_prob
        init_logits = torch.log(torch.tensor(init_keep_prob / (1 - init_keep_prob)))
        self.joint_logits = nn.Parameter(torch.full((num_joints,), init_logits))
        
    def forward(self, skeleton_keypoints: torch.Tensor) -> torch.Tensor:
        """
        Apply learnable pruning to skeleton keypoints.
        
        Args:
            skeleton_keypoints: Shape (B, T, J, 2) or (B, T, J*2)
                B: batch size
                T: number of frames
                J: number of joints
                P: position dimensions (x, y)
        
        Returns:
            pruned_keypoints: Same shape as input, with unselected joints zeroed
        """
        # Handle both (B, T, J, 2) and flattened (B, T, J*2) formats
        input_shape = skeleton_keypoints.shape
        is_flattened = (len(input_shape) == 3)
        
        if is_flattened:
            # Reshape to (B, T, J, 2)
            B, T, JP = input_shape
            J = JP // 2
            skeleton_keypoints = skeleton_keypoints.view(B, T, J, 2)
        else:
            B, T, J, P = skeleton_keypoints.shape
            assert P == 2, f"Expected P=2 (x,y), got {P}"
            assert J == self.num_joints, f"Expected {self.num_joints} joints, got {J}"
        
        # Generate soft selection mask using Gumbel-Softmax
        # Input: (num_joints,) logits
        # Output: (num_joints,) probabilities in [0, 1]
        selection_mask = F.gumbel_softmax(
            self.joint_logits.unsqueeze(0),  # (1, J)
            tau=self.temperature,
            hard=self.hard,
            dim=-1
        ).squeeze(0)  # (J,)
        
        # Reshape mask for broadcasting: (J,) -> (1, 1, J, 1)
        mask = selection_mask.view(1, 1, J, 1)
        
        # Apply soft multiplication (keeps gradient flow)
        pruned = skeleton_keypoints * mask  # (B, T, J, 2)
        
        # Restore original format if needed
        if is_flattened:
            pruned = pruned.view(B, T, JP)
        
        return pruned
    
    def get_selection_probs(self) -> torch.Tensor:
        return torch.sigmoid(self.joint_logits)
    
    def get_active_joints(self, threshold: float = 0.5) -> torch.Tensor:
        """
        Get indices of joints likely to be selected.
        Args: threshold: Probability threshold for being "active"
        Returns: Boolean mask of shape (num_joints,)
        """
        probs = torch.sigmoid(self.joint_logits)
        return probs > threshold
    
    def get_pruning_ratio(self) -> float:
        """
        Get fraction of joints being pruned (probability < 0.5).
        Returns:
            Float in [0, 1]. E.g., 0.25 means 25% of joints are pruned.
        """
        active = self.get_active_joints(threshold=0.5)
        pruned = (~active).float().mean().item()
        return pruned
    
    def get_num_active_joints(self) -> int:
        """Get count of active joints (threshold=0.5)."""
        return self.get_active_joints(threshold=0.5).sum().item()
    
    def set_temperature(self, temperature: float) -> None:
        """Adjust Gumbel-Softmax temperature for annealing."""
        self.temperature = temperature
    
    def get_summary(self) -> dict:
        """Get summary statistics"""
        active = self.get_active_joints(threshold=0.5)
        probs = torch.sigmoid(self.joint_logits)
        
        return {
            "num_active": active.sum().item(),
            "num_total": self.num_joints,
            "pruning_ratio": (~active).float().mean().item(),
            "avg_prob": probs.mean().item(),
            "min_prob": probs.min().item(),
            "max_prob": probs.max().item(),
        }


def l0_penalty(pruning_layer: JointPruningModule, weight: float = 0.001) -> torch.Tensor:
    """
    Compute L0 regularization loss to encourage sparsity.
    
    Higher weight -> more aggressive pruning
    
    Args: pruning_layer: JointPruningModule instance
          weight: Scaling factor for the penalty
        
    Returns: Scalar loss term
    """
    # Penalty is the average probability of keeping a joint
    # Higher probability = higher penalty
    keep_probs = torch.sigmoid(pruning_layer.joint_logits)
    return weight * keep_probs.mean()


# Example usage / testing
if __name__ == "__main__":
    print("JointPruningModule Example:")
    print("=" * 60)
    
    # Create pruning layer for 543 joints (21 per hand)
    pruner = JointPruningModule(num_joints=543, init_keep_prob=0.9)
    
    # Simulate skeleton data: batch_size=4, frames=16, joints=543, features=2
    skeleton = torch.randn(4, 16, 543, 2)
    
    # Apply pruning
    pruned = pruner(skeleton)
    
    print(f"Input shape: {skeleton.shape}")
    print(f"Output shape: {pruned.shape}")
    print()
    
    # Analyze
    summary = pruner.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print()
    print("Active joints:", pruner.get_active_joints(threshold=0.5).nonzero(as_tuple=True)[0].tolist())
    print()
    print("Pruning ratio:", f"{pruner.get_pruning_ratio():.1%}")
