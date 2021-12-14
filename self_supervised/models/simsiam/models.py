'''
Author: jianzhnie
Date: 2021-12-13 12:24:51
LastEditTime: 2021-12-14 13:15:47
LastEditors: jianzhnie
Description:

'''
import torch.nn as nn
from pl_bolts.utils.self_supervised import torchvision_ssl_encoder
from torchvision import models


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
        # compute features for one view
        y = self.encoder(x)[0]
        z = self.projector(y)
        h = self.predictor(z)
        return y, z, h


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


if __name__ == '__main__':
    base = models.resnet50(pretrained=False)
    extract_layers = {'layer4': 'feat5'}
    backbone = models._utils.IntermediateLayerGetter(base, extract_layers)
    model = SiameseArm()
    print(model)
