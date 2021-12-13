'''
Author: jianzhnie
Date: 2021-12-13 18:49:43
LastEditTime: 2021-12-13 18:56:25
LastEditors: jianzhnie
Description:

'''
import torch
import torch.nn.functional as F


def off_diagonal(x):
    """
    >>> x = np.array([[1,2,3],[4,5,6],[7,8,9]])
    array([[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]])
    >>> x.flatten()
    array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> y = x.flatten()[:-1]
    >>> z = y.reshape(2,4)
    >>> z
    array([[1, 2, 3, 4],
        [5, 6, 7, 8]])
    >>> z[:, 1:]
    array([[2, 3, 4],
        [6, 7, 8]])
    """

    n, m = x.shape()
    assert n == m, ' x is not a phalanx'
    return x.flatten()[:-1].viem(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwinLoss(torch.nn.Module):
    def __init__(self, lambda_param=5e-3) -> None:
        super().__init__()

        self.lambda_param = lambda_param

    def forward(self, x, y):
        x_norm = F.normalize(x, dim=1)
        y_norm = F.normalize(y, dim=1)
        N, D = x.size()[:2]

        # cross-correlation matrix
        simmlar_mat = torch.mm(x_norm.T, y_norm) / N
        # loss
        on_diag = torch.diagonal(simmlar_mat).add(-1).pow(2).sum()
        off_diag = off_diagonal(simmlar_mat).pow_(2).sum()
        loss = on_diag + off_diag
        return loss
