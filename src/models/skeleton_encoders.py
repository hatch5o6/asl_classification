"""
Modular skeleton encoders for sign language recognition.

Provides a base class and multiple encoder implementations that accept
skeleton keypoints (B, T, J, P) and output feature vectors (B, fusion_dim).

Available encoders:
    - bert:   HuggingFace BERT transformer over flattened frame embeddings
    - gru:    Bidirectional GRU over flattened frame embeddings
    - stgcn:  Spatial-Temporal Graph Convolutional Network
    - spoter: Spatial + temporal transformer (per-joint embeddings)

Usage:
    from models.skeleton_encoders import build_skeleton_encoder

    config = {"skeleton_encoder": "gru", "num_pose_points": 543, ...}
    encoder = build_skeleton_encoder(config)
    features = encoder(skeleton_keypoints)  # (B, T, J, P) -> (B, fusion_dim)
"""

import json
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from pathlib import Path


class SkeletonEncoder(nn.Module, ABC):
    """
    Base class for all skeleton encoders.

    Contract:
      - Input:  (B, T, J, P) skeleton keypoints (after L0 pruning + gating)
      - Output: (B, fusion_dim) feature vector ready for fusion

    Each subclass owns its own input projection, normalization,
    temporal encoder, and output head.
    """

    def __init__(self, num_joints: int, num_coords: int, fusion_dim: int, config: dict):
        super().__init__()
        self.num_joints = num_joints
        self.num_coords = num_coords
        self.fusion_dim = fusion_dim
        self.config = config

    @abstractmethod
    def forward(self, skeleton_keypoints: torch.Tensor) -> torch.Tensor:
        """
        Args:
            skeleton_keypoints: (B, T, J, P) after pruning/gating
        Returns:
            (B, fusion_dim) feature vector
        """
        pass

    def get_optimizer_param_groups(self, lr: float, weight_decay: float) -> list:
        """
        Return parameter groups for this encoder.
        Default: single group with all parameters.
        Subclasses can override for per-component LR.
        """
        return [{"params": self.parameters(), "lr": lr, "weight_decay": weight_decay}]


# ─────────────────────────────────────────────────────────────────────────────
# BERT Encoder
# ─────────────────────────────────────────────────────────────────────────────

class BertSkeletonEncoder(SkeletonEncoder):
    """
    Existing BERT-based skeleton encoder (extracted from lightning_asl.py).

    Flattens each frame's joints into a single vector, projects to hidden dim,
    runs BERT over the time sequence, mean-pools, and projects to fusion_dim.

    Config keys: bert_hidden_dim, bert_hidden_layers, bert_att_heads,
                 bert_intermediate_size, bert_dropout, num_frames
    """

    def __init__(self, num_joints, num_coords, fusion_dim, config):
        super().__init__(num_joints, num_coords, fusion_dim, config)
        from transformers import BertConfig, BertModel

        num_frames = config["num_frames"]
        if num_frames == "video_mae":
            from transformers import VideoMAEConfig
            num_frames = VideoMAEConfig.from_pretrained(
                "MCG-NJU/videomae-base-finetuned-kinetics"
            ).num_frames
        assert isinstance(num_frames, int)

        hidden_size = config["bert_hidden_dim"]
        bert_config = BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=config["bert_hidden_layers"],
            num_attention_heads=config["bert_att_heads"],
            intermediate_size=config["bert_intermediate_size"],
            max_position_embeddings=num_frames,
            vocab_size=1,
            type_vocab_size=1,
            attention_dropout=0.2,
            hidden_dropout_prob=config["bert_dropout"],
        )

        self.proj = nn.Linear(num_joints * num_coords, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.encoder = BertModel(bert_config)
        self.head = nn.Linear(hidden_size, fusion_dim)

        print(f"Skeleton (BERT) config:")
        print(f"  hidden_size={hidden_size}, layers={config['bert_hidden_layers']}, "
              f"heads={config['bert_att_heads']}, intermediate={config['bert_intermediate_size']}")

    def forward(self, skeleton_keypoints):
        B, T, J, P = skeleton_keypoints.shape
        x = skeleton_keypoints.view(B, T, J * P)
        x = self.proj(x)
        x = self.norm(x)
        x = self.encoder(inputs_embeds=x).last_hidden_state
        x = x.mean(dim=1)
        x = self.head(x)
        return x

    def get_optimizer_param_groups(self, lr, weight_decay):
        return [
            {"params": self.proj.parameters(), "lr": lr, "weight_decay": weight_decay},
            {"params": self.norm.parameters(), "lr": lr, "weight_decay": weight_decay},
            {"params": self.encoder.parameters(), "lr": lr, "weight_decay": weight_decay},
            {"params": self.head.parameters(), "lr": lr, "weight_decay": weight_decay},
        ]


# ─────────────────────────────────────────────────────────────────────────────
# GRU Encoder
# ─────────────────────────────────────────────────────────────────────────────

class GRUSkeletonEncoder(SkeletonEncoder):
    """
    Bidirectional GRU skeleton encoder.

    Flattens joints per frame, projects to hidden dim, runs bidirectional GRU
    over time, mean-pools, and projects to fusion_dim.

    Config keys: gru_hidden_dim (default 256), gru_num_layers (default 2),
                 gru_dropout (default 0.1), gru_bidirectional (default True)
    """

    def __init__(self, num_joints, num_coords, fusion_dim, config):
        super().__init__(num_joints, num_coords, fusion_dim, config)

        hidden_size = config.get("gru_hidden_dim", 256)
        num_layers = config.get("gru_num_layers", 2)
        dropout = config.get("gru_dropout", 0.1)
        bidirectional = config.get("gru_bidirectional", True)

        input_dim = num_joints * num_coords
        self.proj = nn.Linear(input_dim, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        gru_output_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Linear(gru_output_dim, fusion_dim)

        print(f"Skeleton (GRU) config:")
        print(f"  hidden_size={hidden_size}, layers={num_layers}, "
              f"dropout={dropout}, bidirectional={bidirectional}")

    def forward(self, skeleton_keypoints):
        B, T, J, P = skeleton_keypoints.shape
        x = skeleton_keypoints.view(B, T, J * P)
        x = self.proj(x)
        x = self.norm(x)
        x, _ = self.gru(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# ST-GCN Encoder
# ─────────────────────────────────────────────────────────────────────────────

class SpatialTemporalConv(nn.Module):
    """Single ST-GCN block: spatial graph convolution + temporal convolution."""

    def __init__(self, in_channels, out_channels, A, kernel_size=9, stride=1, residual=True):
        super().__init__()
        self.register_buffer('A', A)
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_gcn = nn.BatchNorm2d(out_channels)
        self.tcn = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=(kernel_size, 1),
                padding=(kernel_size // 2, 0),
                stride=(stride, 1),
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

        if not residual or in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        # x: (B, C, T, J)
        res = self.residual(x)
        x = torch.einsum('bctj,jk->bctk', x, self.A)
        x = self.gcn(x)
        x = self.bn_gcn(x)
        x = self.tcn(x) + res
        return self.relu(x)


class STGCNSkeletonEncoder(SkeletonEncoder):
    """
    Spatial-Temporal Graph Convolutional Network for skeleton encoding.

    Operates on the skeleton as a graph — spatial convolution respects
    joint adjacency, temporal convolution captures dynamics.

    Config keys: stgcn_channels (default [64, 128, 256]),
                 stgcn_kernel_size (default 9),
                 stgcn_edges_file (path to JSON edge list),
                 stgcn_dropout (default 0.0)
    """

    def __init__(self, num_joints, num_coords, fusion_dim, config):
        super().__init__(num_joints, num_coords, fusion_dim, config)

        channels = config.get("stgcn_channels", [64, 128, 256])
        kernel_size = config.get("stgcn_kernel_size", 9)
        dropout = config.get("stgcn_dropout", 0.0)

        A = self._build_adjacency(num_joints, config)

        self.bn_in = nn.BatchNorm1d(num_coords * num_joints)

        layers = []
        in_ch = num_coords
        for out_ch in channels:
            layers.append(SpatialTemporalConv(in_ch, out_ch, A, kernel_size))
            in_ch = out_ch
        self.st_layers = nn.ModuleList(layers)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        self.head = nn.Linear(channels[-1], fusion_dim)

        print(f"Skeleton (ST-GCN) config:")
        print(f"  channels={channels}, kernel_size={kernel_size}, "
              f"joints={num_joints}, edges={A.sum().item():.0f}")

    def forward(self, skeleton_keypoints):
        B, T, J, P = skeleton_keypoints.shape
        # ST-GCN expects (B, C, T, J) where C=num_coords
        x = skeleton_keypoints.permute(0, 3, 1, 2)  # (B, P, T, J)

        # Input batch norm (normalize across the spatial-coord dimension)
        # Reshape to (B, P*J, T) for BatchNorm1d, then back
        x_bn = x.reshape(B, P * J, T)
        x_bn = self.bn_in(x_bn)
        x = x_bn.reshape(B, P, T, J)

        for layer in self.st_layers:
            x = layer(x)

        x = self.dropout(x)
        # Global average pool over T and J
        x = x.mean(dim=(2, 3))  # (B, C_last)
        x = self.head(x)
        return x

    @staticmethod
    def _build_adjacency(num_joints, config):
        """Build normalized adjacency matrix from edge list or default to self-loops.

        When a joint_indices_file is provided (joint subset), uses shortest-path
        condensation: two selected joints are connected if any path exists between
        them in the full graph, even through unselected intermediate joints.
        This preserves skeleton topology regardless of how sparse the selection is.
        """
        edges_file = config.get("stgcn_edges_file", None)

        if edges_file is not None:
            with open(edges_file) as f:
                edges = json.load(f)

            joint_indices_file = config.get("joint_indices_file", None)
            if joint_indices_file is not None:
                with open(joint_indices_file) as f:
                    selected = json.load(f)
                selected_set = set(selected)
                global_to_local = {g: l for l, g in enumerate(sorted(selected))}

                # Build adjacency for the full graph (all 543 joints)
                from collections import defaultdict, deque
                full_adj = defaultdict(set)
                for i, j in edges:
                    full_adj[i].add(j)
                    full_adj[j].add(i)

                # For each selected joint, BFS through the full graph to find
                # all other selected joints reachable without passing through
                # another selected joint as an intermediate stop.
                condensed = set()
                for src in selected_set:
                    visited = {src}
                    queue = deque(full_adj[src])
                    while queue:
                        node = queue.popleft()
                        if node in visited:
                            continue
                        visited.add(node)
                        if node in selected_set:
                            # Found a selected joint — add condensed edge
                            pair = (min(src, node), max(src, node))
                            condensed.add(pair)
                            # Don't traverse through other selected joints
                        else:
                            queue.extend(full_adj[node] - visited)

                reindexed = [(global_to_local[i], global_to_local[j])
                             for i, j in condensed]
                print(f"  ST-GCN: {len(reindexed)} condensed edges for "
                      f"{len(selected)} joints (was {sum(1 for i,j in edges if i in selected_set and j in selected_set)} induced edges)")
                edges = reindexed
            else:
                # Full skeleton — no subsetting needed, use edges as-is
                pass

            A = torch.zeros(num_joints, num_joints)
            for i, j in edges:
                if i < num_joints and j < num_joints:
                    A[i, j] = 1.0
                    A[j, i] = 1.0
            # Add self-loops
            A = A + torch.eye(num_joints)
        else:
            # Default: identity (self-loops only, no spatial graph structure)
            A = torch.eye(num_joints)

        # Symmetric normalization: D^{-1/2} A D^{-1/2}
        D = A.sum(dim=1).clamp(min=1)
        D_inv_sqrt = D.pow(-0.5)
        A = D_inv_sqrt.unsqueeze(1) * A * D_inv_sqrt.unsqueeze(0)
        return A


# ─────────────────────────────────────────────────────────────────────────────
# SPOTER Encoder
# ─────────────────────────────────────────────────────────────────────────────

class SPOTERSkeletonEncoder(SkeletonEncoder):
    """
    SPOTER-style encoder with separate spatial and temporal attention.

    Unlike BERT which flattens all joints into one frame vector, SPOTER
    embeds each joint independently, then:
      1. Spatial transformer aggregates across joints within each frame
      2. Temporal transformer aggregates across frames

    Config keys: spoter_hidden_dim (default 256), spoter_hidden_layers (default 2),
                 spoter_att_heads (default 4), num_frames
    """

    def __init__(self, num_joints, num_coords, fusion_dim, config):
        super().__init__(num_joints, num_coords, fusion_dim, config)
        from transformers import BertConfig, BertModel

        hidden_size = config.get("spoter_hidden_dim", 256)
        num_layers = config.get("spoter_hidden_layers", 2)
        num_heads = config.get("spoter_att_heads", 4)

        # Per-joint embedding
        self.joint_embed = nn.Linear(num_coords, hidden_size)
        self.joint_norm = nn.LayerNorm(hidden_size)

        # Spatial attention: across joints within each frame
        spatial_layers = max(1, num_layers // 2)
        spatial_config = BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=spatial_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=num_joints,
            vocab_size=1,
            type_vocab_size=1,
        )
        self.spatial_encoder = BertModel(spatial_config)

        # Temporal attention: across frames
        num_frames = config["num_frames"]
        if num_frames == "video_mae":
            from transformers import VideoMAEConfig
            num_frames = VideoMAEConfig.from_pretrained(
                "MCG-NJU/videomae-base-finetuned-kinetics"
            ).num_frames
        assert isinstance(num_frames, int)

        temporal_layers = max(1, num_layers - spatial_layers)
        temporal_config = BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=temporal_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=num_frames,
            vocab_size=1,
            type_vocab_size=1,
        )
        self.temporal_encoder = BertModel(temporal_config)
        self.head = nn.Linear(hidden_size, fusion_dim)

        print(f"Skeleton (SPOTER) config:")
        print(f"  hidden_size={hidden_size}, spatial_layers={spatial_layers}, "
              f"temporal_layers={temporal_layers}, heads={num_heads}")

    def forward(self, skeleton_keypoints):
        B, T, J, P = skeleton_keypoints.shape
        # Per-joint embedding: (B, T, J, P) -> (B*T, J, hidden)
        x = skeleton_keypoints.reshape(B * T, J, P)
        x = self.joint_embed(x)
        x = self.joint_norm(x)

        # Spatial: attend across joints per frame
        x = self.spatial_encoder(inputs_embeds=x).last_hidden_state
        x = x.mean(dim=1)  # pool joints -> (B*T, hidden)

        # Temporal: attend across frames
        x = x.view(B, T, -1)
        x = self.temporal_encoder(inputs_embeds=x).last_hidden_state
        x = x.mean(dim=1)  # pool frames -> (B, hidden)

        x = self.head(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

SKELETON_ENCODERS = {
    "bert": BertSkeletonEncoder,
    "gru": GRUSkeletonEncoder,
    "stgcn": STGCNSkeletonEncoder,
    "spoter": SPOTERSkeletonEncoder,
}


def build_skeleton_encoder(config: dict) -> SkeletonEncoder:
    """
    Factory function to instantiate a skeleton encoder from config.

    Config key: skeleton_encoder (default: "bert" for backward compatibility)
    """
    encoder_name = config.get("skeleton_encoder", "bert")
    if encoder_name not in SKELETON_ENCODERS:
        raise ValueError(
            f"Unknown skeleton encoder '{encoder_name}'. "
            f"Available: {list(SKELETON_ENCODERS.keys())}"
        )

    num_coords = config.get("num_coords", 2)
    num_joints = config["num_pose_points"]
    fusion_dim = config["fusion_dim"]

    encoder_cls = SKELETON_ENCODERS[encoder_name]
    encoder = encoder_cls(
        num_joints=num_joints,
        num_coords=num_coords,
        fusion_dim=fusion_dim,
        config=config,
    )

    param_count = sum(p.numel() for p in encoder.parameters())
    print(f"Skeleton encoder: {encoder_name} ({encoder_cls.__name__})")
    print(f"  Input: (B, T, {num_joints}, {num_coords})")
    print(f"  Output: (B, {fusion_dim})")
    print(f"  Parameters: {param_count:,}")

    return encoder
