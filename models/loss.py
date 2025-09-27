from typing import Optional, Sequence

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):

    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):

        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


def focal_loss(alpha: Optional[Sequence] = None,
               gamma: float = 0.,
               reduction: str = 'mean',
               ignore_index: int = -100,
               device='cpu',
               dtype=torch.float32) -> FocalLoss:

    if alpha is not None:
        if not isinstance(alpha, Tensor):
            alpha = torch.tensor(alpha)
        alpha = alpha.to(device=device, dtype=dtype)

    fl = FocalLoss(
        alpha=alpha,
        gamma=gamma,
        reduction=reduction,
        ignore_index=ignore_index)
    return fl

class FocalEMA(nn.Module):
    def __init__(self, num_classes=4, ema_alpha=0.8, ignore_index=-100, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.ema_alpha = ema_alpha
        self.ignore_index = ignore_index
        self.device = device
        self.ema_confusion = torch.zeros(num_classes, num_classes, device=device)

    @torch.no_grad()
    def get_weights(self, gt, pd):
        confusion = torch.zeros_like(self.ema_confusion)
        
        for t, p in zip(gt, pd):
            if t != self.ignore_index:
                confusion[t, p] += + 1
        
        self.ema_confusion = self.ema_alpha * confusion + (1 - self.ema_alpha) * self.ema_confusion

        diag = torch.diagonal(self.ema_confusion, dim1=0, dim2=1)
        total_counts = self.ema_confusion.sum(dim=1)
        mispred_counts = total_counts - diag
        
        max_mispred = mispred_counts.max(dim=0, keepdim=True).values
        class_weights = max_mispred / (mispred_counts + 1e-6)
        class_weights = torch.clamp(class_weights, max=1.2)
        return class_weights

    def forward(self, lg, gt):
        pd = lg.max(1).indices
        class_weights = self.get_weights(gt, pd)

        bs, C = lg.shape
        pred_flat = lg.view(bs, C)
        gt_flat = gt.view(bs)
        
        weights_flat = class_weights[gt_flat]
        
        log_probs = F.log_softmax(pred_flat, dim=1)
        ce_loss = F.nll_loss(log_probs, gt_flat.to(self.device), reduction='none')
        weighted_loss = (ce_loss * weights_flat).mean()
        
        return weighted_loss

