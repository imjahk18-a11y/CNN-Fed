import numpy as np
import torch
from torch import nn

from experiments.phase4_evaluate import choices
from experiments.phase4_federated import hold_stats, moments, recalibrate, shared
from experiments.phase4_pretrain import distant
from experiments.sparse_reproduction import cpu_state
from models.active_voxel import SparseUNet, geometry


def test_calibration_uses_pooled_occupied_sites_without_changing_weights():
    model = SparseUNet((3, 4, 3))
    data = []
    for n in (3, 9):
        coords = torch.tensor([[0, i, 1, 0] for i in range(n)])
        data.append((geometry(coords), torch.arange(n * 3).reshape(n, 3).float() / 20))
    orig = cpu_state(model)
    features = torch.cat([model.encoder.convolve(x, g.fine_edges).detach() for g, x in data])
    ck = recalibrate(model, data)
    torch.testing.assert_close(ck["encoder.norm.running_mean"], features.mean(0))
    torch.testing.assert_close(ck["encoder.norm.running_var"], features.var(dim=0, unbiased=False))
    assert all(torch.equal(orig[k], v) for k, v in ck.items() if not moments(k))
    model.train()
    hooks = [
        m.register_forward_pre_hook(hold_stats)
        for m in model.modules()
        if isinstance(m, nn.BatchNorm1d)
    ]
    try:
        model(*data[0])[1].sum().backward()
    finally:
        for h in hooks:
            h.remove()
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in ck.items() if moments(k))
    assert model.encoder.norm.weight.grad is not None


def test_boundaries_for_splits_and_stopping():
    xy = np.array([[i, j] for i in range(5) for j in range(5)])
    assert all(xy[i].max() >= 3 for i in distant(xy, [0]))
    assert choices([0.32, 0.33, 0.32, 0.32, 0.32, 0.40]) == (1, 4, 5)
    assert not shared("encoder.norm.weight", "private")
    assert shared("encoder.norm.weight", "frozen")
    assert not shared("encoder.norm.running_mean", "frozen")
