"""Native PyTorch submanifold convolution over explicit occupied-neighbor pairs."""
from dataclasses import dataclass
from math import sqrt
from itertools import product

import torch
from torch import nn
from torch.nn import functional as F

# Coordinates are [batch, x, y, z]. No edges cross a patch.
AXES = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
KERNEL = ((0, 0, 0),) + tuple(v for axis in AXES for v in (axis, tuple(-x for x in axis)))


def kernel_offsets(sites):
    if sites == 7:
        return KERNEL
    if sites != 27:
        raise ValueError("Expected a 7- or 27-site stencil")
    positive = [v for v in product((-1, 0, 1), repeat=3) if v > (0, 0, 0)]
    return ((0, 0, 0),) + tuple(w for v in positive for w in (v, tuple(-k for k in v)))


def neighbors(coords, sites=7):
    # A one-cell halo separates rows, planes and batches in the integer key space.
    _, sy, sz = (coords[:, 1:].amax(0) + 3).tolist()
    sx = int(coords[:, 1].max()) + 3
    batch, x, y, z = coords.unbind(1)
    keys = ((batch * sx + x + 1) * sy + y + 1) * sz + z + 1
    ordered, permutation = keys.sort()
    if torch.any(ordered[1:] == ordered[:-1]):
        raise ValueError("Duplicate occupied coordinates")
    pairs = []
    for dx, dy, dz in kernel_offsets(sites)[1::2]:
        delta = (dx * sy + dy) * sz + dz
        target = keys + delta
        p = torch.searchsorted(ordered, target).clamp_max(len(keys) - 1)
        dst = torch.where(ordered[p] == target)[0]
        src = permutation[p[dst]]
        pairs.extend(((dst, src), (src, dst)))
    return pairs


@dataclass
class Geometry:
    coords: torch.Tensor
    coarse: torch.Tensor
    parent: torch.Tensor
    count: torch.Tensor
    fine_edges: list
    coarse_edges: list


def geometry(coords, sites=7):
    if coords.dtype not in (torch.int32, torch.int64) or coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError("Expected integer [batch,x,y,z] coordinates")
    if not len(coords) or torch.any(coords < 0):
        raise ValueError("Expected occupied nonnegative coordinates")
    coords = coords.long()
    down = coords.clone()
    down[:, 1:] //= 2
    coarse, parent, count = torch.unique(down, dim=0, return_inverse=True, return_counts=True)
    return Geometry(coords, coarse, parent, count, neighbors(coords, sites), neighbors(coarse, sites))


class SparseBlock(nn.Module):
    def __init__(self, inputs, outputs, sites=7):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(sites, outputs, inputs))
        nn.init.uniform_(self.weight, -1 / sqrt(sites * inputs), 1 / sqrt(sites * inputs))
        self.norm = nn.BatchNorm1d(outputs)

    def convolve(self, x, edges):
        if len(edges) + 1 != len(self.weight):
            raise ValueError("Geometry and convolution stencil differ")
        out = F.linear(x, self.weight[0])
        for weight, (dst, src) in zip(self.weight.unbind()[1:], edges):
            out.index_add_(0, dst, F.linear(x.index_select(0, src), weight))
        return out

    def forward(self, x, edges):
        out = self.convolve(x, edges)
        if self.training and len(out) == 1:
            # A one-site patch cannot estimate a batch variance.
            out = F.batch_norm(
                out,
                self.norm.running_mean,
                self.norm.running_var,
                self.norm.weight,
                self.norm.bias,
                training=False,
                eps=self.norm.eps,
            )
        else:
            out = self.norm(out)
        return out.relu()


class SparseUNet(nn.Module):
    def __init__(self, widths=(16, 32, 16), sites=7, inputs=3):
        super().__init__()
        self.sites = len(kernel_offsets(sites))
        if inputs not in (3,6):
            raise ValueError('Expected RGB or RGB + local XYZ')
        self.inputs = inputs
        self.widths = tuple(widths)
        a, b, c = widths
        self.encoder = SparseBlock(inputs, a, sites)
        self.bridge = SparseBlock(a, b, sites)
        self.decoder = SparseBlock(b + a, c, sites)
        self.head = nn.Linear(c, 6)

    def forward(self, g, rgb):
        if rgb.shape != (len(g.coords), self.inputs) or g.coords.device != rgb.device:
            raise ValueError("Coordinate-feature mismatch")
        skip = self.encoder(rgb, g.fine_edges)
        # FP32 accumulation avoids half-precision reduction error.
        low = skip.new_zeros((len(g.coarse), skip.shape[1]), dtype=torch.float32)
        low.index_add_(0, g.parent, skip.float())
        low = self.bridge(low / g.count[:, None], g.coarse_edges)
        decoded = self.decoder(torch.cat((low[g.parent], skip), dim=1), g.fine_edges)
        return g.coords, self.head(decoded)

    def macs(self, g):
        a, b, c = self.widths
        fine = len(g.coords) + sum(len(src) for _, src in g.fine_edges)
        coarse = len(g.coarse) + sum(len(src) for _, src in g.coarse_edges)
        return fine * (self.inputs * a + (a + b) * c) + coarse * a * b + len(g.coords) * c * 6


@torch.no_grad()
def prune_channels(net, ratio):
    if not 0 <= ratio < 1:
        raise ValueError("Expected ratio in [0,1)")
    blocks = (net.encoder, net.bridge, net.decoder)
    keep = []
    for width, block in zip(net.widths, blocks):
        score = block.weight.abs().sum((0, 2))
        keep.append(
            score.argsort(descending=True, stable=True)[: max(1, round(width * (1 - ratio)))]
            .sort()
            .values
        )
    a, b, c = keep
    device = a.device
    new = SparseUNet(tuple(map(len, keep)), sites=net.sites, inputs=net.inputs).to(device)
    inputs = (torch.arange(net.inputs, device=device), a, torch.cat((b, a + net.widths[1])))
    for old, dest, rows, cols in zip(blocks, (new.encoder, new.bridge, new.decoder), keep, inputs):
        dest.weight.copy_(old.weight[:, rows][:, :, cols])
        for name in ("weight", "bias", "running_mean", "running_var"):
            getattr(dest.norm, name).copy_(getattr(old.norm, name)[rows])
        dest.norm.num_batches_tracked.copy_(old.norm.num_batches_tracked)
    new.head.weight.copy_(net.head.weight[:, c])
    new.head.bias.copy_(net.head.bias)
    return new
