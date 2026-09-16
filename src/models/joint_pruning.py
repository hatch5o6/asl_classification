import torch
import torch.nn as nn


class JointPruningModule(nn.Module):
    """
    Learnable joint pruning using sigmoid-based independent selection.

    This module learns which joints/keypoints are important for classification.
    Each joint has an independent learnable keep/prune probability.

    During training: soft selection (continuous probabilities via sigmoid)
    During inference: hard selection (binary 0/1 mask via threshold)

    This approach enables:
      - Visualization of joint importance over training
      - Measurement of information flow per joint
      - Post-hoc ablation studies at different pruning thresholds

    Example:
        >>> pruner = JointPruningModule(num_joints=543)
        >>> skeleton = torch.randn(4, 16, 543, 2)  # (B, T, J, P)
        >>> pruned = pruner(skeleton)
        >>> print(f"Pruning ratio: {pruner.get_pruning_ratio():.1%}")
        >>> probs = pruner.get_selection_probs()  # Get importance of each joint
    """
    
    # Hard-concrete constants (Louizos et al., ICLR 2018). The stretch interval
    # (gamma, zeta) extends the concrete distribution past [0, 1] so that clamping
    # puts real probability mass on exactly 0 and exactly 1. A plain sigmoid cannot
    # reach either endpoint, which is why the sigmoid gate never closes.
    HC_GAMMA = -0.1
    HC_ZETA = 1.1

    def __init__(
        self,
        num_joints: int,
        temperature: float = 1.0,
        hard: bool = False,
        init_keep_prob: float = 0.9,
        random_init: bool = False,
        random_init_std: float = 0.1,
        gate_type: str = "sigmoid",
        hc_beta: float = 2.0 / 3.0
    ):
        """
        Args:
            num_joints: Total number of skeleton joints
            temperature: Sigmoid temperature scaling. Lower = sharper, higher = softer
            hard: Deprecated (kept for backward compatibility). Hard selection is automatic during eval.
            init_keep_prob: Initial probability to keep each joint (0.0-1.0)
            random_init: If True, add random noise to break symmetry
            random_init_std: Standard deviation of random noise (default 0.1)
        """
        super().__init__()
        assert gate_type in ("sigmoid", "hard_concrete"), \
            f"gate_type must be 'sigmoid' or 'hard_concrete', got {gate_type}"
        self.num_joints = num_joints
        self.temperature = temperature
        self.hard = hard
        self.gate_type = gate_type
        self.hc_beta = hc_beta

        # Learnable logits: one scalar per joint
        # Initialize so that log-odds corresponds to init_keep_prob
        init_logits = torch.log(torch.tensor(init_keep_prob / (1 - init_keep_prob)))

        if random_init:
            # Add random noise to break symmetry
            noise = torch.randn(num_joints) * random_init_std
            self.joint_logits = nn.Parameter(torch.full((num_joints,), init_logits) + noise)
            print(f"L0 Pruning: Using random initialization with std={random_init_std}")
            print(f"  Logit range: [{self.joint_logits.min().item():.3f}, {self.joint_logits.max().item():.3f}]")
        else:
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
        
        # Generate independent selection mask for each joint using sigmoid
        # Input: (num_joints,) learnable logits
        # Output: (num_joints,) independent keep probabilities in [0, 1]
        #
        # Key advantages of sigmoid over softmax:
        #   - Each joint selected independently (not competing for probability mass)
        #   - Clean visualization: probability directly indicates importance
        #   - Enables ablation: can threshold at different K values post-hoc
        #   - Measures information flow: prob × activation magnitude

        if self.gate_type == "hard_concrete":
            selection_mask = self._hard_concrete_mask()
        elif self.training:
            # Soft selection during training (continuous probabilities)
            # Temperature scaling controls decision sharpness
            selection_mask = torch.sigmoid(self.joint_logits / self.temperature)
        else:
            # Hard selection during inference (binary 0/1 decisions)
            selection_mask = (torch.sigmoid(self.joint_logits) > 0.5).float()

        # Reshape mask for broadcasting: (J,) -> (1, 1, J, 1)
        mask = selection_mask.view(1, 1, J, 1)

        # Apply soft multiplication (keeps gradient flow during training)
        pruned = skeleton_keypoints * mask  # (B, T, J, 2)
        
        # Restore original format if needed
        if is_flattened:
            pruned = pruned.view(B, T, JP)
        
        return pruned
    
    def _hard_concrete_mask(self) -> torch.Tensor:
        """
        Hard-concrete gate (Louizos et al. 2018).

        Training draws a stochastic sample; evaluation uses the deterministic
        estimator. Both stretch the (0, 1) sigmoid output to (gamma, zeta) and clamp,
        which is what allows a gate to take the value exactly 0 or exactly 1.
        """
        if self.training:
            u = torch.rand(self.num_joints, device=self.joint_logits.device)
            u = u.clamp(1e-6, 1 - 1e-6)
            s = torch.sigmoid(
                (torch.log(u) - torch.log1p(-u) + self.joint_logits) / self.hc_beta
            )
        else:
            s = torch.sigmoid(self.joint_logits)
        s_stretched = s * (self.HC_ZETA - self.HC_GAMMA) + self.HC_GAMMA
        return s_stretched.clamp(0.0, 1.0)

    def get_open_probs(self) -> torch.Tensor:
        """
        P(gate > 0) per joint. This is the quantity the expected-L0 penalty sums,
        and the importance score to rank by under the hard-concrete gate.
        """
        shift = self.hc_beta * torch.log(
            torch.tensor(-self.HC_GAMMA / self.HC_ZETA, device=self.joint_logits.device)
        )
        return torch.sigmoid(self.joint_logits - shift)

    def expected_l0(self) -> torch.Tensor:
        """Expected number of non-zero gates; differentiable, closed form."""
        return self.get_open_probs().sum()

    def get_selection_probs(self) -> torch.Tensor:
        if self.gate_type == "hard_concrete":
            return self.get_open_probs()
        return torch.sigmoid(self.joint_logits)
    
    def get_active_joints(self, threshold: float = 0.5) -> torch.Tensor:
        """
        Get indices of joints likely to be selected.
        Args: threshold: Probability threshold for being "active"
        Returns: Boolean mask of shape (num_joints,)
        """
        probs = self.get_selection_probs()
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
        probs = self.get_selection_probs()

        summary = {
            "num_active": active.sum().item(),
            "num_total": self.num_joints,
            "pruning_ratio": (~active).float().mean().item(),
            "avg_prob": probs.mean().item(),
            "min_prob": probs.min().item(),
            "max_prob": probs.max().item(),
        }

        if self.gate_type == "hard_concrete":
            # The diagnostic that matters: gates the deterministic estimator sends
            # to exactly 0. The sigmoid gate can never produce a nonzero count here.
            was_training = self.training
            self.eval()
            with torch.no_grad():
                mask = self._hard_concrete_mask()
            if was_training:
                self.train()
            summary["num_exact_zero"] = (mask == 0.0).sum().item()
            summary["num_exact_one"] = (mask == 1.0).sum().item()
            summary["expected_l0"] = self.expected_l0().item()

        return summary


def l0_penalty(pruning_layer: JointPruningModule, weight: float = 0.001,
               batch_size: int = 1, normalize: bool = True) -> torch.Tensor:
    """
    Compute L0 regularization loss to encourage sparsity.

    CRITICAL: Normalization is essential for proper gradient balance!
    Without normalization, the L0 penalty magnitude doesn't match the per-sample
    classification loss magnitude, causing all joints to be pushed down uniformly
    instead of selective pruning.

    Args:
        pruning_layer: JointPruningModule instance
        weight: Scaling factor for the penalty (with normalize=True, use 10-100)
        batch_size: Current batch size for normalization
        normalize: If True, normalize by batch_size and num_joints to match
                   the scale of per-sample classification loss

    Returns: Scalar loss term

    Example WITHOUT normalization (OLD - BROKEN):
        - Classification loss per sample: ~1.5
        - L0 penalty: 0.1 * 418 = 41.8
        - Per-sample L0 contribution: 41.8/8 = 5.2 >> 1.5 (too strong!)

    Example WITH normalization (NEW - CORRECT):
        - Classification loss per sample: ~1.5
        - L0 penalty: 20.0 * (418 / (8 * 543)) = 20.0 * 0.096 = 1.92 (balanced!)
    """
    if pruning_layer.gate_type == "hard_concrete":
        # Closed-form expected L0: the number of gates expected to stay open.
        l0_loss = pruning_layer.expected_l0()
        if normalize:
            # Normalize by num_joints ONLY. The gate is a structural parameter
            # shared across every sample, not a per-sample quantity, so dividing
            # by batch_size shrinks the gradient on that shared parameter by the
            # batch size for no principled reason -- with batch_size=64 that alone
            # made the old penalty 64x too weak to ever close a gate.
            l0_loss = l0_loss / pruning_layer.num_joints
        return weight * l0_loss

    keep_probs = torch.sigmoid(pruning_layer.joint_logits)
    l0_loss = keep_probs.sum()

    if normalize:
        # Legacy sigmoid path, preserved so existing configs reproduce exactly.
        # NOTE: the /batch_size term here is why the sigmoid gate cannot close;
        # see the hard_concrete branch above.
        l0_loss = l0_loss / (batch_size * pruning_layer.num_joints)

    return weight * l0_loss


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
