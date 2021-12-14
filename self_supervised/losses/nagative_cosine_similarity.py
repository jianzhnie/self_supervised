'''
Author: jianzhnie
Date: 2021-12-14 14:23:44
LastEditTime: 2021-12-14 14:52:27
LastEditors: jianzhnie
Description:

'''

import torch
import torch.nn.functional as F


def cosine_similarity(a, b):
    b = b.detach()  # stop gradient of backbone + projection mlp
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    sim = -1 * (a * b).sum(-1).mean()
    return sim


class NegativeCosineSimilarity(torch.nn.Module):
    """Implementation of the Negative Cosine Simililarity used in the
    SimSiam[0] paper.

    [0] SimSiam, 2020, https://arxiv.org/abs/2011.10566

    Examples:

        >>> # initialize loss function
        >>> loss_fn = NegativeCosineSimilarity()
        >>>
        >>> # generate two representation tensors
        >>> # with batch size 10 and dimension 128
        >>> x0 = torch.randn(10, 128)
        >>> x1 = torch.randn(10, 128)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(x0, x1)
    """
    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        """Same parameters as in torch.nn.CosineSimilarity.

        Args:
            dim (int, optional):
                Dimension where cosine similarity is computed. Default: 1
            eps (float, optional):
                Small value to avoid division by zero. Default: 1e-8
        """
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return -F.cosine_similarity(x0, x1, self.dim, self.eps).mean()
