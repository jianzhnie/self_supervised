'''
Author: jianzhnie
Date: 2021-12-13 18:49:43
LastEditTime: 2021-12-14 10:28:08
LastEditors: jianzhnie
Description:

'''
import torch
import torch.nn.functional as F


def off_diagonal(x):
    """Return a flattened view of the off-diagonal elements of a square matrix.

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

    n, m = x.shape
    assert n == m, 'x is not a phalanx'
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwinLoss(torch.nn.Module):
    def __init__(self, lambda_param=5e-3) -> None:
        super().__init__()

        self.lambda_param = lambda_param

    def forward(self, x, y):
        x_norm = F.normalize(x, dim=0)
        y_norm = F.normalize(y, dim=0)
        N, D = x.size()[:2]

        # cross-correlation matrix
        simmlar_mat = torch.mm(x_norm.T, y_norm) / N
        # loss
        on_diag = torch.diagonal(simmlar_mat).add(-1).pow(2).sum()
        off_diag = off_diagonal(simmlar_mat).pow_(2).sum()
        loss = on_diag + self.lambda_param * off_diag
        return loss


class BarlowTwinLoss_(torch.nn.Module):
    def __init__(self, lambda_param=5e-3) -> None:
        super().__init__()

        self.lambda_param = lambda_param

    def forward(self, x, y):
        device = x.device
        x_norm = (x - x.mean(0)) / x.std(0)
        y_norm = (y - y.mean(0)) / y.std(0)

        N, D = x.size()[:2]

        # cross-correlation matrix
        simmlar_mat = torch.mm(x_norm.T, y_norm) / N
        # loss
        diff_mat = (simmlar_mat - torch.eye(D, device=device)).pow(2)
        diff_mat[~torch.eye(D, dtype=bool)] *= self.lambda_param
        loss = diff_mat.sum()
        return loss
