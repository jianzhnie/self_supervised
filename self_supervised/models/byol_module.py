from copy import deepcopy
from typing import Any, Union

import torch
from pl_bolts.callbacks.byol_updates import BYOLMAWeightUpdate
from pl_bolts.models.self_supervised.byol.models import SiameseArm
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning import LightningModule
from torch.nn import functional as F
from torch.optim import Adam


class BYOL(LightningModule):
    """PyTorch Lightning implementation of Bootstrap Your Own Latent (BYOL_)_

    Paper authors: Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre H. Richemond, \
    Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, \
    Bilal Piot, Koray Kavukcuoglu, Rémi Munos, Michal Valko.

    Model implemented by:
        - `Annika Brundyn <https://github.com/annikabrundyn>`_

    .. warning:: Work in progress. This implementation is still being verified.

    TODOs:
        - verify on CIFAR-10
        - verify on STL-10
        - pre-train on imagenet

    Example::

        model = BYOL(num_classes=10)

        dm = CIFAR10DataModule(num_workers=0)
        dm.train_transforms = SimCLRTrainDataTransform(32)
        dm.val_transforms = SimCLREvalDataTransform(32)

        trainer = pl.Trainer()
        trainer.fit(model, datamodule=dm)

    Train::

        trainer = Trainer()
        trainer.fit(model)

    CLI command::

        # cifar10
        python byol_module.py --gpus 1

        # imagenet
        python byol_module.py
            --gpus 8
            --dataset imagenet2012
            --data_dir /path/to/imagenet/
            --meta_dir /path/to/folder/with/meta.bin/
            --batch_size 32

    .. _BYOL: https://arxiv.org/pdf/2006.07733.pdf
    """
    def __init__(self,
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
                 **kwargs):
        """
        Args:
            datamodule: The datamodule
            learning_rate: the learning rate
            weight_decay: optimizer weight decay
            input_height: image input height
            batch_size: the batch size
            num_workers: number of workers
            warmup_epochs: num of epochs for scheduler warm up
            max_epochs: max epochs for scheduler
            base_encoder: the base encoder module or resnet name
            encoder_out_dim: output dimension of base_encoder
            projector_hidden_size: hidden layer size of projector MLP
            projector_out_dim: output size of projector MLP
        """
        super().__init__()
        self.save_hyperparameters(ignore='base_encoder')

        self.online_network = SiameseArm(base_encoder, encoder_out_dim,
                                         projector_hidden_size,
                                         projector_out_dim)
        self.target_network = deepcopy(self.online_network)
        self.weight_callback = BYOLMAWeightUpdate()

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int,
                           dataloader_idx: int) -> None:
        # Add callback for user automatically since it's key to BYOL weight update
        self.weight_callback.on_train_batch_end(self.trainer, self, outputs,
                                                batch, batch_idx,
                                                dataloader_idx)

    def forward(self, x):
        y, _, _ = self.online_network(x)
        return y

    def shared_step(self, batch, batch_idx):
        imgs, y = batch
        img_1, img_2 = imgs[:2]

        # Image 1 to image 2 loss
        y1, z1, h1 = self.online_network(img_1)
        with torch.no_grad():
            y2, z2, h2 = self.target_network(img_2)
        loss_a = -2 * F.cosine_similarity(h1, z2).mean()

        # Image 2 to image 1 loss
        y1, z1, h1 = self.online_network(img_2)
        with torch.no_grad():
            y2, z2, h2 = self.target_network(img_1)
        # L2 normalize
        loss_b = -2 * F.cosine_similarity(h1, z2).mean()

        # Final loss
        total_loss = loss_a + loss_b

        return loss_a, loss_b, total_loss

    def training_step(self, batch, batch_idx):
        loss_a, loss_b, total_loss = self.shared_step(batch, batch_idx)

        # log results
        self.log_dict({
            '1_2_loss': loss_a,
            '2_1_loss': loss_b,
            'train_loss': total_loss
        })

        return total_loss

    def validation_step(self, batch, batch_idx):
        loss_a, loss_b, total_loss = self.shared_step(batch, batch_idx)

        # log results
        self.log_dict({
            '1_2_loss': loss_a,
            '2_1_loss': loss_b,
            'val_loss': total_loss
        })

        return total_loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(),
                         lr=self.hparams.learning_rate,
                         weight_decay=self.hparams.weight_decay)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs)
        return [optimizer], [scheduler]
