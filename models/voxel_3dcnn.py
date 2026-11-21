import torch
from torch import nn
import torch.nn.functional as F


class Block(nn.Sequential):
    def __init__(self, inputs, outputs):
        super().__init__(nn.Conv3d(inputs, outputs, 3, padding=1, bias=False),
                         nn.BatchNorm3d(outputs), nn.ReLU())


def nonoverlapping_max_pool(x):
    # Max over disjoint 2x2x2 cells, with the same first-index tie rule and floor sizing as MaxPool3d.
    n, c, depth, height, width = x.shape
    d, h, w = depth // 2, height // 2, width // 2
    cells = x[:, :, :2*d, :2*h, :2*w].reshape(n, c, d, 2, h, 2, w, 2)
    return cells.permute(0, 1, 2, 4, 6, 3, 5, 7).flatten(5).max(5).values


class VoxelUNet(nn.Module):
    def __init__(self, inputs, classes, widths=(8, 16, 8)):
        super().__init__()
        self.spec = dict(inputs=inputs, classes=classes, widths=list(widths))
        a, b, c = widths
        self.encoder = Block(inputs, a)
        self.pool = nn.MaxPool3d(2)
        self.bridge = Block(a, b)
        self.decoder = Block(b+a, c)
        self.head = nn.Conv3d(c, classes, 1)

    def forward(self, x):
        skip = self.encoder(x)
        if skip.is_cuda and torch.is_grad_enabled() and torch.are_deterministic_algorithms_enabled():
            pooled = nonoverlapping_max_pool(skip)
        else:
            pooled = self.pool(skip)
        low = self.bridge(pooled)
        up = F.interpolate(low, size=skip.shape[2:], mode="nearest")
        return self.head(self.decoder(torch.cat((up, skip), dim=1)))
