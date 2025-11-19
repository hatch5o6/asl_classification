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


class SignClassificationLightning(L.LightningModule):
    def __init__(
        self,
        config,
        pretrained_model="MCG-NJU/videomae-base-finetuned-kinetics"
    ):
        super().__init__()
        self.config = config

        #----------------------------- CONFIGS -----------------------------#
        # RGB config
        video_mae_config = VideoMAEConfig.from_pretrained(pretrained_model)
        video_mae_config.attention_dropout = 0.0
        video_mae_config.drop_out = 0.0
        video_mae_config.attn_implementation = "sdpa"
        print("\n\n")
        print(f"RGB Config (VideoMAE from {pretrained_model}):\n________________________________________")
        self.print_config(video_mae_config)

        # Depth config
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
        bert_config = BertConfig(
            hidden_size=self.config["bert_hidden_dim"],
            num_hidden_layers=self.config["bert_hidden_layers"],
            num_attention_heads=self.config["bert_att_heads"],
            intermediate_size=self.config["bert_intermediate_size"],
            max_position_embeddings=video_mae_config.num_frames,
            vocab_size=1,
            type_vocab_size=1
        )
        print("Skeleton (BERT) config:\n________________________________________")
        self.print_config(bert_config)

        #----------------------------- LOSS -----------------------------#
        # Loss function
        class_weights_tensor, num_classes_in_training = self._compute_class_weights(self.config["train_csv"])
        self.loss_fn = CrossEntropyLoss(weight=class_weights_tensor)

        #----------------------------- TRAINABLE PARAMETERS -----------------------------#
        # RGB encoder
        self.rgb_encoder = VideoMAEModel.from_pretrained(
            pretrained_model,
            config=video_mae_config
        )
        self.rgb_encoder.train()
        self.rgb_head = torch.nn.Linear(self.rgb_encoder.config.hidden_size, self.config["fusion_dim"])

        # Depth encoder
        self.depth_encoder = VideoMAEModel(depth_config)
        self.depth_encoder.train()
        self.depth_head = torch.nn.Linear(self.depth_encoder.config.hidden_size, self.config["fusion_dim"])

        # Skeleton encoder
        self.skel_encoder = BertModel(bert_config)
        self.skel_proj = torch.nn.Linear(config["num_pose_points"] * 2, self.skel_encoder.config.hidden_size)
        self.skel_encoder.train()
        self.skel_head = torch.nn.Linear(self.skel_encoder.config.hidden_size, self.config["fusion_dim"])

        # modality weights [rgb, depth, skeleton]
        self.modality_weights = torch.nn.Parameter(torch.ones(3))

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
            rgb_output = self.rgb_encoder(pixel_values).last_hidden_state[:, 0]
            rgb_feat = self.rgb_head(rgb_output)
            features.append(rgb_feat)
            weights.append(self.modality_weights[0])
        
        # Depth
        if depth_values is not None:
            depth_output = self.depth_encoder(depth_values).last_hidden_state[:, 0]
            depth_feat = self.depth_head(depth_output)
            features.append(depth_feat)
            weights.append(self.modality_weights[1])
            
        # Skeleton
        if skeleton_keypoints is not None:
            B, T, J, P = skeleton_keypoints.shape
            assert P == 2
            skeleton_keypoints = skeleton_keypoints.view(B, T, J * P)
            skeleton_keypoints = self.skel_proj(skeleton_keypoints)
            skel_output = self.skel_encoder(input_embeds=skeleton_keypoints).last_hidden_state[:, 0]
            skel_feat = self.skel_head(skel_output)
            features.append(skel_feat)
            weights.append(self.modality_weights[2])

        # Fuse modalitites
        weights = torch.softmax(torch.stack(weights), dim=0)
        fused = torch.sum(torch.stack([w * f for w, f in zip(weights, features)]), dim=0)

        # Classify
        logits = self.classifier(fused)
        return logits
            
    def pad_batch():
        pass
        #TODO implement frame padding

    def training_step(self, batch, batch_idx):
        pixel_values = batch.get("pixel_values")
        depth_values = batch.get("depth_values")
        skel_keypoints = batch.get("skeleton_keypoints")
        #TODO implement frame padding for skeleton keypoints

        labels = batch["labels"]

        assert any([pixel_values is not None, depth_values is not None, skel_keypoints is not None])
        logits = self(
            pixel_values=pixel_values,
            depth_values=depth_values,
            skeleton_keypoints=skel_keypoints
        )
        loss = self.loss_fn(logits, labels)
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
        #TODO implement frame padding for skeleton keypoints

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


    def prediction_step(self, batch, batch_idx):
        pixel_values = batch.get("pixel_values")
        depth_values = batch.get("depth_values")
        skel_keypoints = batch.get("skeleton_keypoints")
        #TODO implement frame padding for skeleton keypoints
        
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
        optimizer = optim.AdamW([
            {
                "params": self.rgb_encoder.parameters(), 
                "lr": self.config["pretrained_learning_rate"], 
                "weight_decay": self.config["weight_decay"]
            },
            {
                "params": self.rgb_head.parameters(),
                "lr": self.config["new_learning_rate"],
                "weight_decay": self.config["weight_decay"]
            },
            {
                "params": self.depth_encoder.parameters(),
                "lr": self.config["new_learning_rate"],
                "weight_decay": self.config["weight_decay"]
            },
            {
                "params": self.depth_head.parameters(),
                "lr": self.config["new_learning_rate"],
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
            },
            {
                "params": [self.modality_weights],
                "lr": self.config["class_learning_rate"],
                "weight_decay": self.config["weight_decay"]
            },
            {
                "params": self.classifier.parameters(),
                "lr": self.config["class_learning_rate"],
                "weight_decay": self.config["weight_decay"]
            }
        ])

        
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