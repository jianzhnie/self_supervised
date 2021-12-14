import torch
import torch.nn as nn
from pl_bolts.models.self_supervised.resnets import resnet18, resnet50
from pl_bolts.models.self_supervised.simsiam.models import SiameseArm
from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from pytorch_lightning import LightningModule


class SimSiam(LightningModule):
    """PyTorch Lightning implementation of Exploring Simple Siamese
    Representation Learning (SimSiam_)

    Paper authors: Xinlei Chen, Kaiming He.

    Model implemented by:
        - `Zvi Lapp <https://github.com/zlapp>`_

    .. warning:: Work in progress. This implementation is still being verified.

    TODOs:
        - verify on CIFAR-10
        - verify on STL-10
        - pre-train on imagenet

    Example::

        model = SimSiam()

        dm = CIFAR10DataModule(num_workers=0)
        dm.train_transforms = SimCLRTrainDataTransform(32)
        dm.val_transforms = SimCLREvalDataTransform(32)

        trainer = Trainer()
        trainer.fit(model, datamodule=dm)

    Train::

        trainer = Trainer()
        trainer.fit(model)

    CLI command::

        # cifar10
        python simsiam_module.py --gpus 1

        # imagenet
        python simsiam_module.py
            --gpus 8
            --dataset imagenet2012
            --data_dir /path/to/imagenet/
            --meta_dir /path/to/folder/with/meta.bin/
            --batch_size 32

    .. _SimSiam: https://arxiv.org/pdf/2011.10566v1.pdf
    """
    def __init__(self,
                 gpus: int,
                 num_samples: int,
                 batch_size: int,
                 dataset: str,
                 num_nodes: int = 1,
                 arch: str = 'resnet50',
                 hidden_mlp: int = 2048,
                 feat_dim: int = 128,
                 warmup_epochs: int = 10,
                 max_epochs: int = 100,
                 temperature: float = 0.1,
                 first_conv: bool = True,
                 maxpool1: bool = True,
                 optimizer: str = 'adam',
                 exclude_bn_bias: bool = False,
                 start_lr: float = 0.0,
                 learning_rate: float = 1e-3,
                 final_lr: float = 0.0,
                 weight_decay: float = 1e-6,
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
        """
        super().__init__()
        self.save_hyperparameters()

        self.gpus = gpus
        self.num_nodes = num_nodes
        self.arch = arch
        self.dataset = dataset
        self.num_samples = num_samples
        self.batch_size = batch_size

        self.hidden_mlp = hidden_mlp
        self.feat_dim = feat_dim
        self.first_conv = first_conv
        self.maxpool1 = maxpool1

        self.optim = optimizer
        self.exclude_bn_bias = exclude_bn_bias
        self.weight_decay = weight_decay
        self.temperature = temperature

        self.start_lr = start_lr
        self.final_lr = final_lr
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        self.init_model()

        # compute iters per epoch
        nb_gpus = len(self.gpus) if isinstance(gpus,
                                               (list, tuple)) else self.gpus
        assert isinstance(nb_gpus, int)
        global_batch_size = self.num_nodes * nb_gpus * self.batch_size if nb_gpus > 0 else self.batch_size
        self.train_iters_per_epoch = self.num_samples // global_batch_size

        self.cosine_similarity_ = nn.CosineSimilarity(dim=1).cuda(self.gpus)

    def init_model(self):
        if self.arch == 'resnet18':
            backbone = resnet18
        elif self.arch == 'resnet50':
            backbone = resnet50

        encoder = backbone(first_conv=self.first_conv,
                           maxpool1=self.maxpool1,
                           return_all_feature_maps=False)
        self.online_network = SiameseArm(encoder,
                                         input_dim=self.hidden_mlp,
                                         hidden_size=self.hidden_mlp,
                                         output_dim=self.feat_dim)

    def forward(self, x):
        y, _, _ = self.online_network(x)
        return y

    def shard_step_(self, batch, batch_idx):
        (img_1, img_2, _), y = batch

        # Image 1 to image 2 loss
        _, z1, h1 = self.online_network(img_1)
        _, z2, h2 = self.online_network(img_2)
        loss = -(self.cosine_similarity_(h1, z2).mean() +
                 self.cosine_similarity_(h2, z1).mean()) * 0.5
        return loss

    def shard_step(self, batch, batch_idx):
        (img_1, img_2, _), y = batch

        # Image 1 to image 2 loss
        _, z1, h1 = self.online_network(img_1)
        _, z2, h2 = self.online_network(img_2)
        loss = self.cosine_similarity(h1, z2) / 2 + self.cosine_similarity(
            h2, z1) / 2
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shard_step_(batch, batch_idx)
        # log results
        self.log_dict({'train_loss': loss})
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shard_step_(batch, batch_idx)
        # log results
        self.log_dict({'val_loss': loss})
        return loss

    def exclude_from_wt_decay(self,
                              named_params,
                              weight_decay,
                              skip_list=['bias', 'bn']):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {
                'params': params,
                'weight_decay': weight_decay
            },
            {
                'params': excluded_params,
                'weight_decay': 0.0
            },
        ]

    def configure_optimizers(self):
        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay(self.named_parameters(),
                                                weight_decay=self.weight_decay)
        else:
            params = self.parameters()

        if self.optim == 'lars':
            optimizer = LARS(
                params,
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
                trust_coefficient=0.001,
            )
        elif self.optim == 'adam':
            optimizer = torch.optim.Adam(params,
                                         lr=self.learning_rate,
                                         weight_decay=self.weight_decay)

        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
        total_steps = self.train_iters_per_epoch * self.max_epochs

        scheduler = {
            'scheduler':
            torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps, total_steps, cosine=True),
            ),
            'interval':
            'step',
            'frequency':
            1,
        }

        return [optimizer], [scheduler]
