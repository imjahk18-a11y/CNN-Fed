from itertools import product

import pytest
import torch
from torch.nn import functional as F

from models.active_voxel import SparseUNet, geometry, kernel_offsets


@pytest.mark.parametrize("sites", [7, 27])
def test_sparse_matches_dense_oracle_and_gradient(sites):
    # Two patches, all 26 neighbors around the center, a hole, and isolated corners.
    coords = torch.tensor(
        [
            (b, *v)
            for b in range(2)
            for v in product(range(4), repeat=3)
            if v != (0, 3, 0) and (max(v) < 3 or v == (3, 3, 3))
        ]
    )
    coords = coords[torch.randperm(len(coords), generator=torch.Generator().manual_seed(4))]
    g = geometry(coords, sites)
    model = SparseUNet(sites=sites).double()
    x = torch.randn(len(coords), 3, dtype=torch.double, requires_grad=True)
    block = model.encoder
    got = block.convolve(x, g.fine_edges)
    cube = x.new_zeros(2, 4, 4, 4, 3)
    idx = coords.unbind(1)
    cube[idx] = x
    k = x.new_zeros(16, 3, 3, 3, 3)
    for i, (dx, dy, dz) in enumerate(kernel_offsets(sites)):
        k[:, :, dx + 1, dy + 1, dz + 1] = block.weight[i]
    ref = F.conv3d(cube.permute(0, 4, 1, 2, 3), k, padding=1).permute(0, 2, 3, 4, 1)[idx]
    torch.testing.assert_close(got, ref)
    dy = torch.randn_like(got)
    actual = torch.autograd.grad(got, (x, block.weight), dy)
    expected = torch.autograd.grad(ref, (x, block.weight), dy)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    wrong = SparseUNet(sites=34 - sites)
    with pytest.raises(ValueError, match="stencil"):
        wrong(g, x.float())


def test_missing_classes_and_model_field():
    import numpy as np
    from experiments.sparse_reproduction import scores_from_confusion
    from experiments.sparse_diagnosis import layer_support

    cm = np.zeros((6, 6), dtype=int)
    cm[0, 0] = 3
    cm[0, 1] = 1
    score = scores_from_confusion(cm)
    assert score["iou"] == [0.75, 0, None, None, None, None]
    assert score["miou"] == 0.375
    assert score["accuracy"] == 0.75
    assert score["macro_f1"] == pytest.approx(3 / 7)
    for sites, n in ((7, 336), (27, 1000)):
        support = layer_support(sites)
        for corner in product((0, 1), repeat=3):
            assert support[str(corner)] == dict(points=n, bbox_cells=[10, 10, 10])
