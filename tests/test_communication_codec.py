import pytest
import torch

from federated.communication_codec import decode, encode
from federated.fedavg import fedavg


def test_independent_scales_and_numerical_aggregation():
    states = [
        {
            "w": torch.tensor([a, -a, a / 3]),
            "zero": torch.zeros(3),
            "norm.num_batches_tracked": torch.tensor(counter),
        }
        for a, counter in ((0.01, 2), (10.0, 7))
    ]
    restored = []
    for state in states:
        raw, (_, units) = encode(state, "INT8")
        assert units <= 0.50002
        recv = decode(raw, "INT8", dict(reversed(list(state.items()))))
        torch.testing.assert_close(
            recv["w"], state["w"], atol=state["w"].abs().max().item() / 254 + 1e-7, rtol=0
        )
        assert torch.equal(recv["zero"], state["zero"])
        restored.append(recv)
    averaged = fedavg(restored, [1, 3])
    assert averaged["w"][0].item() == pytest.approx(7.5025, abs=1e-6)
    assert averaged["norm.num_batches_tracked"].item() == 7
    raw, errors = encode(states[1], "FP32")
    assert errors == (0, 0)
    assert all(torch.equal(v, decode(raw, "FP32", states[1])[k]) for k, v in states[1].items())


def test_reject_incompatible_template():
    raw, _ = encode({"x": torch.ones(3)}, "INT8")
    with pytest.raises(ValueError):
        decode(raw, "INT8", {"x": torch.ones(4)})
