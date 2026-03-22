import csv
from collections import Counter

import torch
from torch import optim
from torch.nn import CrossEntropyLoss
import lightning as L
from torch.optim.lr_scheduler import LambdaLR
from pytorch_lightning.utilities import rank_zero_info
from transformers import VideoMAEModel, VideoMAEImageProcessor, VideoMAEConfig
from models.joint_pruning import JointPruningModule, l0_penalty
from models.skeleton_encoders import build_skeleton_encoder

def sort_modalities(x):
    assert x in ["rgb", "depth", "skeleton"]
    if x == "rgb":
        return 0
    elif x == "depth":
        return 1
    elif x == "skeleton":
        return 2

class SignClassificationLightning(L.LightningModule):
    def __init__(
        self,
        config,
        pretrained_model="MCG-NJU/videomae-base-finetuned-kinetics"
    ):
        super().__init__()
        self.config = config

        assert isinstance(self.config["modalities"], list)
        self.config["modalities"].sort(key=sort_modalities)
        for mod in self.config["modalities"]:
            assert mod in ["rgb", "depth", "skeleton"]
        if "depth" in self.config["modalities"]:
            assert "rgb" in self.config["modalities"]

        #----------------------------- CONFIGS -----------------------------#
        # RGB config
        video_mae_config = VideoMAEConfig.from_pretrained(pretrained_model)
        video_mae_config.attention_dropout = 0.0
        video_mae_config.drop_out = 0.0
        video_mae_config.attn_implementation = "sdpa"
        print("\n\n")
        print(f"RGB Config (VideoMAE from {pretrained_model}):\n________________________________________")
        self.print_config(video_mae_config)
        if "rgb" not in self.config["modalities"]:
            print(f"HOWEVER, RGB Config (VideoMAE from {pretrained_model})\n\tWILL NOT BE USED BECAUSE THE RGB ENCODER WILL NOT BE INITIALIZED. THIS CONFIG IS LIKELY BEING USED TO SET VALUES IN SKELETON OR DEPTH ENCODER CONFIGS.\n\n")

        # Depth config
        if "depth" in self.config["modalities"]:
            depth_config = VideoMAEConfig.from_pretrained(pretrained_model)
            depth_config.num_frames = video_mae_config.num_frames
            depth_config.image_size = self.config["depth_image_size"]
            depth_config.hidden_size = self.config["depth_hidden_dim"]
            depth_config.num_hidden_layers = self.config["depth_hidden_layers"]
            depth_config.num_attention_heads = self.config["depth_att_heads"]
            depth_config.intermediate_size = self.config["depth_intermediate_size"]
            depth_config.patch_size = self.config["depth_patch_size"]
            # depth_config.num_channels = 1
            depth_config.attn_implementation = "sdpa"
            print("Depth (VideoMAE) config:\n________________________________________")
            self.print_config(depth_config)

        # Skeleton config — handled by skeleton encoder factory

        #----------------------------- LOSS -----------------------------#
        # Loss function
        label_smoothing = self.config.get("label_smoothing", 0.0)
        class_weights_tensor, num_classes_in_training = self._compute_class_weights(self.config["train_csv"])
        if self.config.get("use_class_weights", True):
            self.loss_fn = CrossEntropyLoss(weight=class_weights_tensor,
                                            label_smoothing=label_smoothing)
        else:
            self.loss_fn = CrossEntropyLoss(label_smoothing=label_smoothing)
        if label_smoothing > 0:
            print(f"Label smoothing: {label_smoothing}")

        #----------------------------- TRAINABLE PARAMETERS -----------------------------#
        next_idx = 0
        # RGB encoder
        if "rgb" in self.config["modalities"]:
            self.rgb_mod_idx = next_idx
            next_idx += 1
            self.rgb_encoder = VideoMAEModel.from_pretrained(
                pretrained_model,
                config=video_mae_config
            )
            self.rgb_encoder.train()
            self.rgb_head = torch.nn.Linear(self.rgb_encoder.config.hidden_size, self.config["fusion_dim"])

        # Depth encoder
        if "depth" in self.config["modalities"]:
            self.depth_mod_idx = next_idx
            next_idx += 1
            self.depth_encoder = VideoMAEModel(depth_config)
            self.depth_encoder.train()
            self.depth_head = torch.nn.Linear(self.depth_encoder.config.hidden_size, self.config["fusion_dim"])

        # Skeleton encoder
        if "skeleton" in self.config["modalities"]:
            self.skel_mod_idx = next_idx
            next_idx += 1
            self.skel_encoder_module = build_skeleton_encoder(config)
            self.skel_encoder_module.train()

            # Skeleton pruning layer (experiment-level, outside encoder)
            if self.config["joint_pruning"] == True:
                use_random_init = self.config.get("use_random_init", False)
                self.joint_pruning = JointPruningModule(
                    num_joints=self.config["num_pose_points"],
                    init_keep_prob=self.config["init_keep_probability"],
                    random_init=use_random_init,
                    random_init_std=self.config.get("random_init_std", 0.1)
                )

            # Gating mechanism (experiment-level, outside encoder)
            if self.config.get("use_gating", False):
                num_coords = config.get("num_coords", 2)
                gate_hidden = self.config.get("gate_hidden_dim", 64)
                self.joint_gate = torch.nn.Sequential(
                    torch.nn.Linear(num_coords, gate_hidden),
                    torch.nn.ReLU(),
                    torch.nn.Linear(gate_hidden, 1),
                    torch.nn.Sigmoid()
                )
                print(f"\n\nGating Network:\n________________________________________")
                print(f"Input: {num_coords} coords per joint")
                print(f"Hidden: {gate_hidden}")
                print(f"Output: 1 gate value per joint")
                print(f"Parameters: ~{num_coords * gate_hidden + gate_hidden + gate_hidden + 1}")
                print("\n\n")
            
        assert next_idx == len(self.config["modalities"])
        self.modality_weights = torch.nn.Parameter(torch.ones(len(self.config["modalities"])))

        # Classifier
        num_classes = self._get_num_classes(self.config["class_id_csv"])
        assert num_classes_in_training == num_classes
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.config["fusion_dim"], self.config["fusion_dim"]),
            torch.nn.GELU(),
            torch.nn.Dropout(self.config["classifier_dropout"]),
            torch.nn.Linear(self.config["fusion_dim"], num_classes)
        )

    def print_config(self, config):
        for k, v in vars(config).items():
            if k in ["label2id", "id2label"]:
                print(f"-{k}=`{{SKIP}}`")
            else:
                print(f"-{k}=`{v}`")
        print("\n\n")

    def forward(self, pixel_values=None, depth_values=None, skeleton_keypoints=None):
        # assert at least one modality is passed
        # assert any([pixel_values is not None, depth_values is not None, skeleton_keypoints is not None])
        assert pixel_values is not None or depth_values is not None or skeleton_keypoints is not None

        features = []
        weights = []

        # RGB
        if pixel_values is not None:
            # rgb_output = self.rgb_encoder(pixel_values).last_hidden_state[:, 0]
            rgb_output = self.rgb_encoder(pixel_values).last_hidden_state.mean(dim=1)
            rgb_feat = self.rgb_head(rgb_output)
            features.append(rgb_feat)
            weights.append(self.modality_weights[self.rgb_mod_idx])
        
        # Depth
        if depth_values is not None:
            # depth_output = self.depth_encoder(depth_values).last_hidden_state[:, 0]
            depth_output = self.depth_encoder(depth_values).last_hidden_state.mean(dim=1)
            depth_feat = self.depth_head(depth_output)
            features.append(depth_feat)
            weights.append(self.modality_weights[self.depth_mod_idx])
            
        # Skeleton
        if skeleton_keypoints is not None:
            B, T, J, P = skeleton_keypoints.shape
            expected_coords = self.config.get("num_coords", 2)
            assert P == expected_coords, f"Expected {expected_coords} coordinates, got {P}"

            # Apply L0 pruning mask (structural, global across all samples)
            if self.config["joint_pruning"] == True:
                skeleton_keypoints = self.joint_pruning(skeleton_keypoints)

            # Apply gating (sample-specific, content-based importance)
            if self.config.get("use_gating", False):
                # Compute gates from joint coordinates: (B, T, J, P) -> (B, T, J, 1)
                gates = self.joint_gate(skeleton_keypoints)  # (B, T, J, 1)
                # Apply gates (element-wise multiplication)
                skeleton_keypoints = skeleton_keypoints * gates  # (B, T, J, P)

                # Store gates for logging (detach to avoid affecting gradients)
                self._last_gates = gates.detach()

            # Skeleton encoder (handles projection, encoding, pooling, and head)
            skel_feat = self.skel_encoder_module(skeleton_keypoints)
            features.append(skel_feat)
            weights.append(self.modality_weights[self.skel_mod_idx])

        # Fuse modalitites
        weights = torch.softmax(torch.stack(weights), dim=0)
        fused = torch.sum(torch.stack([w * f for w, f in zip(weights, features)]), dim=0)

        # Classify
        logits = self.classifier(fused)
        return logits

    def training_step(self, batch, batch_idx):
        pixel_values = batch.get("pixel_values")
        depth_values = batch.get("depth_values")
        skel_keypoints = batch.get("skeleton_keypoints")

        labels = batch["labels"]

        assert any([pixel_values is not None, depth_values is not None, skel_keypoints is not None])
        logits = self(
            pixel_values=pixel_values,
            depth_values=depth_values,
            skeleton_keypoints=skel_keypoints
        )
        loss = self.loss_fn(logits, labels)

        if "skeleton" in self.config["modalities"] and self.config["joint_pruning"] == True:
            # Temperature annealing: gradually sharpen pruning decisions
            # Research shows: start HIGH (10.0) for strong gradients, anneal FAST to force binary decisions
            #
            # Why this schedule works:
            # - High temp (10.0): Sigmoid becomes very soft, gradients flow well, model learns importance
            # - Low temp (0.01): Sigmoid becomes very sharp, forces probabilities toward 0 or 1
            # - Fast anneal (50k steps): Gives model time to learn, then forces commitment
            #
            # Previous schedule (1.0 → 0.1 over 200k) was too slow - model converged to 0.77
            # equilibrium before temperature got low enough to force binary decisions
            temp_start = self.config.get("temp_start", 10.0)
            temp_end = self.config.get("temp_end", 0.01)
            temp_anneal_steps = self.config.get("temp_anneal_steps", 50000)

            # Exponential decay: temp = start * (end/start)^progress
            temp_progress = min(1.0, self.global_step / temp_anneal_steps)
            temperature = temp_start * (temp_end / temp_start) ** temp_progress
            self.joint_pruning.set_temperature(temperature)

            # Add L0 regularization to encourage sparsity
            # Supports three modes:
            # 1. No warmup/anneal: constant l0_weight from step 0
            # 2. Warmup only: l0_weight turns on after l0_warmup_steps
            # 3. Annealing: l0_weight ramps from 0 to l0_end_weight over l0_anneal_steps (after warmup)
            l0_warmup_steps = self.config.get("l0_warmup_steps", 0)
            l0_anneal_steps = self.config.get("l0_anneal_steps", 0)
            l0_end_weight = self.config.get("l0_end_weight", self.config.get("l0_weight", 0.001))

            if self.global_step < l0_warmup_steps:
                # Warmup phase: no L0 penalty
                current_l0_weight = 0.0
                self.log("l0_active", 0.0, on_step=True)
            elif l0_anneal_steps > 0:
                # Annealing phase: linearly ramp up L0 weight
                anneal_progress = min(1.0, (self.global_step - l0_warmup_steps) / l0_anneal_steps)
                current_l0_weight = l0_end_weight * anneal_progress
                self.log("l0_active", anneal_progress, on_step=True)
            else:
                # No annealing: full L0 weight after warmup
                current_l0_weight = l0_end_weight
                self.log("l0_active", 1.0, on_step=True)

            if current_l0_weight > 0:
                # Get batch size from skeleton keypoints
                batch_size = skel_keypoints.size(0) if skel_keypoints is not None else labels.size(0)
                # Use normalized L0 penalty for proper gradient balance
                loss = loss + l0_penalty(self.joint_pruning, weight=current_l0_weight,
                                        batch_size=batch_size, normalize=True)
            self.log("l0_weight", current_l0_weight, on_step=True)

            # Log pruning statistics for visualization
            summary = self.joint_pruning.get_summary()
            self.log("pruning_ratio", summary["pruning_ratio"], on_epoch=True)
            self.log("num_active_joints", summary["num_active"], on_epoch=True)
            self.log("avg_joint_prob", summary["avg_prob"], on_epoch=True)
            self.log("temperature", temperature, on_step=True)

            # Log joint probability statistics periodically for visualization
            # Every 1000 steps, log distribution statistics
            if self.global_step % 1000 == 0:
                joint_probs = self.joint_pruning.get_selection_probs().float()  # Convert to float32 for quantile

                # Log percentiles to understand probability distribution
                self.log("joint_prob_p25", torch.quantile(joint_probs, 0.25), on_step=True)
                self.log("joint_prob_median", torch.median(joint_probs), on_step=True)
                self.log("joint_prob_p75", torch.quantile(joint_probs, 0.75), on_step=True)

                # Log top-K joint statistics (cap at number of available joints)
                k_log = min(50, len(joint_probs))
                top_50_probs = torch.topk(joint_probs, k=k_log).values
                self.log("top_50_avg_prob", top_50_probs.mean(), on_step=True)
                self.log("top_50_min_prob", top_50_probs.min(), on_step=True)

        # Log gating statistics (if using gating)
        if self.config.get("use_gating", False) and hasattr(self, '_last_gates'):
            # Average gates across batch and time: (B, T, J, 1) -> (J,)
            avg_gates = self._last_gates.mean(dim=(0, 1, 3)).float()  # Convert to float32 for quantile

            # Log gate statistics every step
            self.log("gate_mean", avg_gates.mean(), on_step=True, on_epoch=True)
            self.log("gate_std", avg_gates.std(), on_step=True, on_epoch=True)
            self.log("gate_min", avg_gates.min(), on_step=True)
            self.log("gate_max", avg_gates.max(), on_step=True)

            # Periodically log detailed gate statistics
            if self.global_step % 1000 == 0:
                self.log("gate_p25", torch.quantile(avg_gates, 0.25), on_step=True)
                self.log("gate_median", torch.median(avg_gates), on_step=True)
                self.log("gate_p75", torch.quantile(avg_gates, 0.75), on_step=True)

                # How many joints have high gates (> 0.7)?
                high_gate_count = (avg_gates > 0.7).sum()
                self.log("num_high_gate_joints", high_gate_count, on_step=True)

                # Per-body-part gate means (using original 543-space indices)
                selected_indices = self.config.get("selected_joint_indices", None)
                if selected_indices is not None:
                    orig_indices = selected_indices
                else:
                    orig_indices = list(range(self.config["num_pose_points"]))

                # Classify each joint by body part
                face_mask = torch.tensor([0 <= idx < 468 for idx in orig_indices])
                pose_mask = torch.tensor([468 <= idx < 501 for idx in orig_indices])
                lh_mask = torch.tensor([501 <= idx < 522 for idx in orig_indices])
                rh_mask = torch.tensor([522 <= idx < 543 for idx in orig_indices])

                if face_mask.any():
                    self.log("gate_face_mean", avg_gates[face_mask].mean(), on_step=True)
                if pose_mask.any():
                    self.log("gate_pose_mean", avg_gates[pose_mask].mean(), on_step=True)
                if lh_mask.any():
                    self.log("gate_left_hand_mean", avg_gates[lh_mask].mean(), on_step=True)
                if rh_mask.any():
                    self.log("gate_right_hand_mean", avg_gates[rh_mask].mean(), on_step=True)

        self.log(
            "train_loss", 
            loss, 
            on_step=True, 
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.config["batch_size"]
        )
        return loss


    def validation_step(self, batch, batch_idx):
        pixel_values = batch.get("pixel_values")
        depth_values = batch.get("depth_values")
        skel_keypoints = batch.get("skeleton_keypoints")

        labels = batch["labels"]

        assert any([pixel_values is not None, depth_values is not None, skel_keypoints is not None])
        logits = self(
            pixel_values=pixel_values,
            depth_values=depth_values,
            skeleton_keypoints=skel_keypoints
        )
        loss = self.loss_fn(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        # Top-5 accuracy
        k = min(5, logits.size(1))
        top5_preds = logits.topk(k, dim=1).indices
        top5_correct = top5_preds.eq(labels.unsqueeze(1)).any(dim=1).float().mean()

        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.config["batch_size"]
        )
        self.log(
            "val_acc",
            acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.config["batch_size"]
        )
        self.log(
            "val_top5_acc",
            top5_correct,
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=self.config["batch_size"]
        )

        # Store predictions and labels for per-class analysis at epoch end
        if not hasattr(self, '_val_preds'):
            self._val_preds = []
            self._val_labels = []
        self._val_preds.append(preds.detach().cpu())
        self._val_labels.append(labels.detach().cpu())

        return loss

    def on_validation_epoch_end(self):
        """Log per-class accuracy summary at end of validation epoch."""
        if hasattr(self, '_val_preds') and len(self._val_preds) > 0:
            all_preds = torch.cat(self._val_preds)
            all_labels = torch.cat(self._val_labels)

            # Per-class accuracy
            num_classes = self.classifier[-1].out_features
            per_class_correct = torch.zeros(num_classes)
            per_class_total = torch.zeros(num_classes)
            for c in range(num_classes):
                mask = all_labels == c
                if mask.sum() > 0:
                    per_class_total[c] = mask.sum()
                    per_class_correct[c] = (all_preds[mask] == c).sum()

            active_classes = per_class_total > 0
            per_class_acc = torch.where(
                active_classes,
                per_class_correct / per_class_total.clamp(min=1),
                torch.zeros_like(per_class_total)
            )

            # Log summary statistics
            if active_classes.sum() > 0:
                active_accs = per_class_acc[active_classes]
                self.log("val_per_class_acc_mean", active_accs.mean(), logger=True)
                self.log("val_per_class_acc_std", active_accs.std(), logger=True)
                self.log("val_per_class_acc_min", active_accs.min(), logger=True)
                self.log("val_per_class_acc_p25", torch.quantile(active_accs.float(), 0.25), logger=True)
                self.log("val_per_class_acc_median", torch.median(active_accs), logger=True)

            # Clear stored predictions
            self._val_preds = []
            self._val_labels = []
    

    # def test_step(self, batch, batch_idx):
    #     pixel_values = batch.get("pixel_values")
    #     depth_values = batch.get("depth_values")
    #     skel_keypoints = batch.get("skeleton_keypoints")
    #     labels = batch["labels"]

    #     assert any([pixel_values is not None, depth_values is not None, skel_keypoints is not None])
    #     logits = self(
    #         pixel_values=pixel_values,
    #         depth_values=depth_values,
    #         skeleton_keypoints=skel_keypoints
    #     )
    #     loss = self.loss_fn(logits, labels)

    #     preds = torch.argmax(logits, dim=1)
    #     acc = (preds == labels).float().mean()
    #     self.log(
    #         "val_loss", 
    #         loss, 
    #         on_epoch=True,
    #         prog_bar=True,
    #         logger=True,
    #         batch_size=self.config["batch_size"]
    #     )
    #     self.log(
    #         "val_acc", 
    #         acc, 
    #         on_epoch=True,
    #         prog_bar=True,
    #         logger=True,
    #         batch_size=self.config["batch_size"]
    #     )
        
    #     return {"test_loss": loss, "test_acc": acc}


    def predict_step(self, batch, batch_idx):
        pixel_values = batch.get("pixel_values")
        depth_values = batch.get("depth_values")
        skel_keypoints = batch.get("skeleton_keypoints")

        labels = batch["labels"]

        assert any([pixel_values is not None, depth_values is not None, skel_keypoints is not None])
        logits = self(
            pixel_values=pixel_values,
            depth_values=depth_values,
            skeleton_keypoints=skel_keypoints
        )

        # Return raw logits so TTA can average them; caller does argmax
        return logits, labels
    
    def configure_optimizers(self):
        optimizer_configs = []
        if "rgb" in self.config["modalities"]:
            optimizer_configs += [{
                "params": self.rgb_encoder.parameters(), 
                "lr": self.config["pretrained_learning_rate"], 
                "weight_decay": self.config["weight_decay"]
            },
            {
                "params": self.rgb_head.parameters(),
                "lr": self.config["new_learning_rate"],
                "weight_decay": self.config["weight_decay"]
            }]
        if "depth" in self.config["modalities"]:
            optimizer_configs += [{
                "params": self.depth_encoder.parameters(),
                "lr": self.config["new_learning_rate"],
                "weight_decay": self.config["weight_decay"]
            },
            {
                "params": self.depth_head.parameters(),
                "lr": self.config["new_learning_rate"],
                "weight_decay": self.config["weight_decay"]
            }]
        if "skeleton" in self.config["modalities"]:
            optimizer_configs += self.skel_encoder_module.get_optimizer_param_groups(
                lr=self.config["skel_learning_rate"],
                weight_decay=self.config["weight_decay"],
            )
            if self.config["joint_pruning"] == True:
                optimizer_configs.append({
                    "params": [self.joint_pruning.joint_logits],
                    "lr": self.config["skel_learning_rate"],
                    "weight_decay": 0.0
                })
            if self.config.get("use_gating", False):
                optimizer_configs.append({
                    "params": self.joint_gate.parameters(),
                    "lr": self.config.get("gate_learning_rate", self.config["skel_learning_rate"]),
                    "weight_decay": self.config["weight_decay"]
                })
        optimizer_configs += [{
            "params": [self.modality_weights],
            "lr": self.config["class_learning_rate"],
            "weight_decay": self.config["weight_decay"]
        },
        {
            "params": self.classifier.parameters(),
            "lr": self.config["class_learning_rate"],
            "weight_decay": self.config["weight_decay"]
        }]

        if self.config.get("use_adam_optimizer", False):
            optimizer = optim.Adam(optimizer_configs)
        else:
            optimizer = optim.AdamW(optimizer_configs)

        if self.config.get("use_reduce_lr_on_plateau", False):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=5
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_acc",
                    "interval": "epoch",
                    "frequency": 1
                }
            }
        else:
            lr_lambda = get_linear_schedule_with_warmup(
                num_warmup_steps=self.config["warmup_steps"],
                num_training_steps=self.config["max_steps"]
            )
            scheduler = LambdaLR(optimizer, lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1
                }
            }

    def on_load_checkpoint(self, checkpoint):
        """Remap old state dict key names to new modular skeleton encoder names."""
        sd = checkpoint["state_dict"]
        # Only remap if checkpoint uses old key format (pre-refactor)
        if not any(k.startswith("skel_encoder_module.") for k in sd):
            remaps = [
                ("skel_encoder.", "skel_encoder_module.encoder."),
                ("skel_proj.",    "skel_encoder_module.proj."),
                ("skel_norm.",    "skel_encoder_module.norm."),
                ("skel_head.",    "skel_encoder_module.head."),
            ]
            new_sd = {}
            for k, v in sd.items():
                new_k = k
                for old, new in remaps:
                    if k.startswith(old):
                        new_k = new + k[len(old):]
                        break
                new_sd[new_k] = v
            checkpoint["state_dict"] = new_sd

    def _compute_class_weights(self, csv_file):
        annotations = self._read_label_annotations(csv_file)
        labels = [int(label) for _, _, _, label in annotations]
        label_cts = Counter(labels)

        num_classes = len(label_cts)
        total_samples = len(labels)

        class_weights = {}
        for label, ct in label_cts.items():
            class_weights[label] = total_samples / (num_classes * ct)

        weight_list = [class_weights[i] for i in range(num_classes)]
        class_weights_tensor = torch.tensor(weight_list, dtype=torch.float32)

        return class_weights_tensor, num_classes

    def _read_label_annotations(self, csv_f):
        with open(csv_f, newline='') as inf:
            rows = [tuple(r) for r in csv.reader(inf)]
        header = rows[0]
        assert header == ("rgb_path", "depth_path", "skel_path", "label")
        data = rows[1:]
        return data
    
    def _get_num_classes(self, class_id_csv):
        with open(class_id_csv, newline='') as inf:
            rows = [tuple(r) for r in csv.reader(inf)]
        header = rows[0]
        assert header == ("ClassId", "TR", "EN")
        labels = set()
        data = rows[1:]
        for label, tr_word, en_word in data:
            assert label not in labels
            labels.add(label)
        return len(labels)

def get_linear_schedule_with_warmup(num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)),
        )
    return lr_lambda