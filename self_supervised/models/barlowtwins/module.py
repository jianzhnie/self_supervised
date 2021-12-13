'''
Author: jianzhnie
Date: 2021-12-13 17:55:41
LastEditTime: 2021-12-13 18:55:13
LastEditors: jianzhnie
Description:

'''

from typing import Union

import torch
import torch.nn as nn
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pl_bolts.utils.self_supervised import torchvision_ssl_encoder
from pytorch_lightning import LightningModule
from torch.optim import Adam

from .loss import BarlowTwinLoss


class MLP(nn.Module):
    def __init__(self, input_dim=2048, hidden_size=4096, output_dim=256):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_dim, bias=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class BarlowTwinsModel(nn.Module):
    def __init__(self,
                 encoder='resnet50',
                 encoder_out_dim=2048,
                 projector_hiddex_size=4096,
                 projector_out_dim=256):
        super().__init__()

        if isinstance(encoder, str):
            encoder = torchvision_ssl_encoder(encoder)

        self.encoder = encoder
        self.projector = MLP(encoder_out_dim, projector_hiddex_size,
                             projector_out_dim)

    def forward(self, x):
        out1 = self.encoder(x)
        out2 = self.projector(out1)
        return out1, out2


class BarlowTwins(LightningModule):
    def __init__(
        self,
        num_classes,
        learning_rate: float = 0.2,
        weight_decay: float = 1.5e-6,
        input_height: int = 32,
        batch_size: int = 32,
        num_workers: int = 0,
        warmup_epochs: int = 10,
        max_epochs: int = 1000,
        base_encoder: Union[str, torch.nn.Module] = 'resnet50',
        encoder_out_dim: int = 2048,
        projector_hidden_size: int = 4096,
        projector_out_dim: int = 256,
    ):
        super().__init__()

        self.save_hyperparameters(ignore='base_encoder')

        self.model = BarlowTwinsModel(base_encoder, encoder_out_dim,
                                      projector_hidden_size, projector_out_dim)
        self.criterion = BarlowTwinLoss()

    def forward(self, x):
        out1, out2 = self.model(x)
        return out1

    def shared_step(self, batch, batch_idx):
        imgs, y = batch
        img_1, img_2 = imgs[:2]

        _, z1 = self.model(img_1)
        _, z2 = self.model(img_2)

        loss = self.criterion(z1, z2)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        # log results
        self.log_dict({'train_loss': loss})

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        # log results
        self.log_dict({'val_loss': loss})
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(),
                         lr=self.hparams.learning_rate,
                         weight_decay=self.hparams.weight_decay)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs)
        return [optimizer], [scheduler]
