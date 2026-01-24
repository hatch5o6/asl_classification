import csv
from collections import Counter

import torch
from torch import optim
from torch.nn import CrossEntropyLoss
import lightning as L
from torch.optim.lr_scheduler import LambdaLR
from pytorch_lightning.utilities import rank_zero_info
from transformers import VideoMAEModel, VideoMAEImageProcessor, VideoMAEConfig
from transformers import BertConfig, BertModel
from joint_pruning import JointPruningModule, l0_penalty

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

        # Skeleton config
        bert_num_frames = self.config["num_frames"]
        if bert_num_frames == "video_mae":
            bert_num_frames = video_mae_config.num_frames
        assert isinstance(bert_num_frames, int)
        if "skeleton" in self.config["modalities"]:
            bert_config = BertConfig(
                hidden_size=self.config["bert_hidden_dim"],
                num_hidden_layers=self.config["bert_hidden_layers"],
                num_attention_heads=self.config["bert_att_heads"],
                intermediate_size=self.config["bert_intermediate_size"],
                # max_position_embeddings=video_mae_config.num_frames,
                max_position_embeddings=bert_num_frames,
                vocab_size=1,
                type_vocab_size=1,
                attention_dropout=0.2,
                hidden_dropout_prob=self.config["bert_dropout"]
            )
            print("Skeleton (BERT) config:\n________________________________________")
            self.print_config(bert_config)

        #----------------------------- LOSS -----------------------------#
        # Loss function
        class_weights_tensor, num_classes_in_training = self._compute_class_weights(self.config["train_csv"])
        self.loss_fn = CrossEntropyLoss(weight=class_weights_tensor)

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
            self.skel_encoder = BertModel(bert_config)
            self.skel_encoder.train()
            num_coords = config.get("num_coords", 2)  # 2 for X,Y; 3 for X,Y,Z
            self.skel_proj = torch.nn.Linear(config["num_pose_points"] * num_coords,
                                             self.skel_encoder.config.hidden_size)
            #add layer norm
            self.skel_norm = torch.nn.LayerNorm(self.skel_encoder.config.hidden_size)
            self.skel_head = torch.nn.Linear(self.skel_encoder.config.hidden_size, self.config["fusion_dim"])
            # Skeleton pruning layer
            if self.config["joint_pruning"] == True:
                self.joint_pruning = JointPruningModule(
                    num_joints=self.config["num_pose_points"],
                    init_keep_prob=self.config["init_keep_probability"]
                )

        # # New Skeleton encoder [Batch, Frame, Joints, Points]
        # if "skeleton" in self.config["modalities"]:
            
        # modality weights [rgb, depth, skeleton]
            
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
            if self.config["joint_pruning"] == True:
                skeleton_keypoints = self.joint_pruning(skeleton_keypoints)


            # old skeleton processing    
            # skeleton_keypoints = skeleton_keypoints.view(B, T, J * P)
            # skeleton_keypoints = self.skel_proj(skeleton_keypoints)
            # skel_output = self.skel_encoder(inputs_embeds=skeleton_keypoints).last_hidden_state
            # skel_feat = self.skel_head(skel_output)

            # new skeleton processing
            skel_flat = skeleton_keypoints.view(B, T, J * P)
            skel_embeds = self.skel_proj(skel_flat)
            skel_embeds = self.skel_norm(skel_embeds)
            skel_output = self.skel_encoder(inputs_embeds=skel_embeds).last_hidden_state
            skel_pooled = skel_output.mean(dim=1)

            skel_feat = self.skel_head(skel_pooled)
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
            # Start soft (temp=1.0) to explore, end sharp (temp=0.1) to commit
            progress = self.global_step / self.config["max_steps"]
            temperature = max(0.1, 1.0 - 0.9 * progress)  # 1.0 → 0.1 over training
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
                loss = loss + l0_penalty(self.joint_pruning, weight=current_l0_weight)
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
                joint_probs = self.joint_pruning.get_selection_probs()

                # Log percentiles to understand probability distribution
                self.log("joint_prob_p25", torch.quantile(joint_probs, 0.25), on_step=True)
                self.log("joint_prob_median", torch.median(joint_probs), on_step=True)
                self.log("joint_prob_p75", torch.quantile(joint_probs, 0.75), on_step=True)

                # Log top-K joint statistics
                top_50_probs = torch.topk(joint_probs, k=50).values
                self.log("top_50_avg_prob", top_50_probs.mean(), on_step=True)
                self.log("top_50_min_prob", top_50_probs.min(), on_step=True)

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
        return loss
    

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

        preds = torch.argmax(logits, dim=1)

        return preds, labels
    
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
            optimizer_configs += [{
                "params": self.skel_proj.parameters(),
                "lr": self.config["skel_learning_rate"],
                "weight_decay": self.config["weight_decay"]
            },
            {
                "params": self.skel_norm.parameters(),
                "lr": self.config["skel_learning_rate"],
                "weight_decay": self.config["weight_decay"]
            },
            {
                "params": self.skel_encoder.parameters(),
                "lr": self.config["skel_learning_rate"],
                "weight_decay": self.config["weight_decay"]
            },
            {
                "params": self.skel_head.parameters(),
                "lr": self.config["skel_learning_rate"],
                "weight_decay": self.config["weight_decay"]
            }]
            if self.config["joint_pruning"] == True:
                optimizer_configs.append({
                    "params": [self.joint_pruning.joint_logits],
                    "lr": self.config["skel_learning_rate"],
                    "weight_decay": 0.0
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

        optimizer = optim.AdamW(optimizer_configs)

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