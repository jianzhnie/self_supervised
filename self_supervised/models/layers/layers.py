'''
Author: jianzhnie
Date: 2021-12-14 15:13:38
LastEditTime: 2021-12-14 15:29:53
LastEditors: jianzhnie
Description:

'''

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.utils.self_supervised import torchvision_ssl_encoder


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


class SiameseArm(nn.Module):
    def __init__(self,
                 encoder='resnet50',
                 encoder_out_dim=2048,
                 projector_hidden_size=4096,
                 projector_out_dim=256):
        super().__init__()

        if isinstance(encoder, str):
            encoder = torchvision_ssl_encoder(encoder)
        # Encoder
        self.encoder = encoder
        # Projector
        self.projector = MLP(encoder_out_dim, projector_hidden_size,
                             projector_out_dim)
        # Predictor
        self.predictor = MLP(projector_out_dim, projector_hidden_size,
                             projector_out_dim)

    def forward(self, x):
        y = self.encoder(x)[0]
        z = self.projector(y)
        h = self.predictor(z)
        return y, z, h


class ProjectorHead(nn.Module):
    def __init__(self, input_dim=2048, hidden_size=4096, output_dim=256):
        super().__init__()
        self.out_channels = 256
        self.projection = MLP(input_dim, hidden_size, output_dim)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x_pooled = self.avg_pool(x)
        h = x_pooled.view(x_pooled.shape[0],
                          x_pooled.shape[1])  # removing the last dimension
        return self.projection(h)


class SwaVPrototypes(nn.Module):
    """Prototypes used for SwaV.

    Each output feature is assigned to a prototype, SwaV solves the swapped
    predicition problem where the features of one augmentation are used to
    predict the assigned prototypes of the other augmentation.

    Examples:
        >>> # use features with 128 dimensions and 512 prototypes
        >>> prototypes = SwaVPrototypes(128, 512)
        >>>
        >>> # pass batch through backbone and projection head.
        >>> features = model(x)
        >>> features = nn.functional.normalize(features, dim=1, p=2)
        >>>
        >>> # logits has shape bsz x 512
        >>> logits = prototypes(features)
    """
    def __init__(self, input_dim: int, n_prototypes: int):
        super().__init__()
        self.layers = nn.Linear(input_dim, n_prototypes, bias=False)

    def farward(self, x):
        out = self.layers(x)
        return out


class BYOL(nn.Module):
    def __init__(self, backbone: nn.Module, target_momentum=0.996):
        super().__init__()
        self.online_network = backbone
        self.target_network = copy.deepcopy(backbone)
        # Projection Head
        self.online_projector = ProjectorHead()
        # Predictor Head
        self.predictor = MLP(self.online_projector.out_channels, 4096, 256)
        self.m = target_momentum

    def initialize_target_network(self):
        for param_q, param_k in zip(self.online_network.parameters(),
                                    self.target_network.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for param_q, param_k in zip(self.online_projector.parameters(),
                                    self.target_projector.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def update_target_network(self):
        for param_q, param_k in zip(self.online_network.parameters(),
                                    self.target_network.parameters()):
            param_k.data = self.m * param_k.data + (1 - self.m) * param_q.data

        for param_q, param_k in zip(self.online_projector.parameters(),
                                    self.target_projector.parameters()):
            param_k.data = self.m * param_k.data + (1 - self.m) * param_q.data

    @staticmethod
    def regression_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, dim=1)  # L2-normalize
        y_norm = F.normalize(y, dim=1)  # L2-normalize
        loss = 2 - 2 * (x_norm * y_norm).sum(dim=-1)  # dot product
        return loss.mean()
