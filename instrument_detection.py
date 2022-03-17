from pathlib import Path
from typing import Any
import cv2

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy
from torchvision import models
from torch import sigmoid
from torch.utils.data import DataLoader
import albumentations as A
import numpy as np
from torch.utils.data import Dataset
import json

class Model(LightningModule):
    def __init__(self, num_classes, freeze_extractor, val_loss_weights=None, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.freeze_extractor = freeze_extractor
        self.save_hyperparameters()
        
        self.model = models.efficientnet_b4(pretrained=True)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, 1)

        if val_loss_weights:
            val_loss_weights = torch.FloatTensor(val_loss_weights).to(0)
        if self.freeze_extractor:
            for param in self.model.parameters():
                param.requires_grad = False

        self.loss = nn.BCEWithLogitsLoss()
        self.criterion = nn.BCEWithLogitsLoss()

        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

        print("Model Setup Complete!")

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        inputs, labels = batch
        output = self.forward(inputs)
        loss = self.criterion(output, labels.unsqueeze(1).type_as(output))

        preds = sigmoid(output.detach()).squeeze(dim = -1)
        preds[preds >= 0.5] = 1
        preds[preds < 0.5] = 0
        preds = preds.bool()

        return loss, preds, labels

    def training_step(self, batch: Any, batch_idx: int):
        inputs, labels = batch
        output = self.forward(inputs)
        loss = self.loss(output, labels.unsqueeze(1).type_as(output))
        preds = sigmoid(output.detach()).squeeze(dim = -1)
        preds[preds >= 0.5] = 1
        preds[preds < 0.5] = 0
        preds = preds.bool()
        

        acc = self.train_accuracy(preds, labels)

        # log train metrics
        self.log("train/total_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": labels}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_accuracy(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}


    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_accuracy(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        use_optimizer = "adam"

        if use_optimizer == "adam":
            optimizer = torch.optim.AdamW(
                params=self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            from madgrad import MADGRAD

            optimizer = MADGRAD(
                params=self.parameters(),
                lr=self.hparams.lr,
                weight_decay=1e-12,
            )

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=5,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-08,
            verbose=False,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val/loss",
        }

img_transforms = A.Compose([
    A.Normalize(
        mean=(0.45211223, 0.27139644, 0.19264949),
        std=(0.31418097, 0.21088019, 0.16059452),
        max_pixel_value=255)
])

def square_img(img):
    y, x, _ = img.shape
    delta = x-y
    if delta > 0:
        _padding = [(abs(delta), 0), (0, 0), (0, 0)]
        img = np.pad(img, _padding)
    elif delta < 0:
        _padding = [(0, 0), (abs(delta),0), (0, 0)]
        img = np.pad(img, _padding)

    return img

class BinaryImageClassificationDS(Dataset):
    def __init__(self, paths, labels, scaling: int = 75):
        self.paths = paths
        self.scaling = scaling

        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        width = int(1024 * self.scaling / 100) 
        height = int(1024 * self.scaling / 100)

        img = square_img(img,)
        dim = (width, height)
        img = cv2.resize(img, dsize=dim, interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img_transforms(image=img)["image"]
        img = torch.tensor(img)
        img = torch.swapaxes(img, 0, 2)

        return img, self.labels[idx]

ckpt_path = Path("model.ckpt")
model = Model.load_from_checkpoint(ckpt_path.as_posix())

image_dir = Path("image_folder")
paths = [_ for _ in image_dir.iterdir() if _.suffix == ".jpg"]

_ids = [_.name for _ in paths]
paths = [_.as_posix() for _ in paths]

ds = BinaryImageClassificationDS(paths, _ids, 69)
loader = DataLoader(
    dataset=ds,
    batch_size=12,
    num_workers=0,
    shuffle = False
)

if eval:
    print("Load in Evaluation Mode")
    model.eval()
    model.freeze()

result = {}
for batch in loader:
    imgs, ids = batch
    pred = sigmoid(model(imgs)).numpy().squeeze()
    pred[pred>0.5]=True
    pred[pred<=0.5]=False
    for i, _id in enumerate(ids):
        result[str(_id)] = bool(pred[i])

_keys = [_ for _ in result.keys()]
_keys.sort()
result = {_: result[_] for _ in _keys}
print(result)

with open("result.json", "w") as f:
    json.dump(result, f)
