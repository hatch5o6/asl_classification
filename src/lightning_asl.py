import csv
from collections import Counter

import torch
from torch import optim
from torch.nn import CrossEntropyLoss
import Lightning as L
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
        video_mae_config = VideoMAEConfig.from_pretrained(pretrained_model)
        bert_config = BertConfig()

        class_weights_tensor = self._compute_class_weights(config["data_csv"])
        self.loss_fn = CrossEntropyLoss(weight=class_weights_tensor)

        # RGB encoder
        self.rgb_encoder = VideoMAEModel.from_pretrained(
            pretrained_model,
            attn_implementation="sdpa",
            dtype=torch.float16
        )
        self.rgb_head = torch.nn.Linear(self.rgb_encoder.config.hidden_size, self.config["fusion_dim"])

        # Depth encoder
        self.depth_encoder = VideoMAEModel(
            video_mae_config,
            attn_implementation="sdpa",
            dtype=torch.float16
        )
        self.depth_head = torch.nn.Linear(self.depth_encoder.config.hidden_size, self.config["fusion_dim"])

        # Skeleton encoder
        self.skel_encoder = BertModel(bert_config)
        self.skel_head = torch.nn.Linear(self.skel_encoder.config.hidden_size, self.config["fusion_dim"])


        # modality weights [rgb, depth, skeleton]
        self.modality_weights = torch.nn.Parameter(torch.ones(3))

        # Classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.fusion_dim, self.fusion_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.fusion_dim, self.config["num_classes"])
        )


    def forward(self, pixel_values=None, depth_values=None, skeleton_keypoints=None):
        # assert at least one modality is passed
        assert any([pixel_values is not None, depth_values is not None, skeleton_keypoints is not None])

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
            skel_output = self.skel_encoder(skeleton_keypoints).last_hidden_state[:, 0]
            skel_feat = self.skel_head(skel_output)
            features.append(skel_feat)
            weights.append(self.modality_weights[2])

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


    def prediction_step(self, batch, batch_idx):
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

        return preds

    def _compute_class_weights(self, csv_file):
        annotations = self._read_label_annotations(csv_file)
        labels = [label for _, label in annotations]
        label_cts = Counter(labels)

        num_classes = len(label_cts)
        total_samples = len(labels)

        class_weights = {}
        for label, ct in label_cts.items():
            class_weights[label] = total_samples / (num_classes * ct)

        weight_list = [class_weights[i] for i in range(num_classes)]
        class_weights_tensor = torch.tensor(weight_list, dtype=torch.float32)

        return class_weights_tensor

    def _read_label_annotations(self, csv_f):
        with open(csv_f, newline='') as inf:
            rows = [tuple(r) for r in csv.reader(inf)]
        header = rows[0]
        assert header == ("rgb_path", "depth_path", "skel_path", "label")
        data = rows[1:]
        return data