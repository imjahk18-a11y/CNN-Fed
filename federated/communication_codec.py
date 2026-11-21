"""Encode complete CPU model states for bidirectional FedAvg communication."""
import torch

from federated.process_fedavg import serialize, unpack_state


def decode(raw, mode, template):
    blob = unpack_state(raw)
    if mode == "INT8":
        state, scales = blob
        expected = sum(t.is_floating_point() for t in template.values())
        if (
            scales.dtype != torch.float32
            or scales.shape != (expected,)
            or not (torch.isfinite(scales) & (scales > 0)).all()
        ):
            raise ValueError("Invalid INT8 scales")
        factors = iter(scales)
    elif mode == "FP32":
        state = blob
    else:
        raise ValueError(mode)
    # A scale is associated with each floating key in canonical sorted order.
    if state.keys() != template.keys():
        raise ValueError("Mismatched state keys")
    decoded = {}
    for key in sorted(template):
        t, ref = state[key], template[key]
        if t.shape != ref.shape:
            raise ValueError(key)
        if mode == "INT8" and ref.is_floating_point():
            if t.dtype != torch.int8 or (t == -128).any():
                raise ValueError(key)
            t = t.float() * next(factors)
        if t.dtype != ref.dtype or not torch.isfinite(t).all():
            raise ValueError(key)
        decoded[key] = t
    return decoded


def encode(state, mode):
    if mode == "FP32":
        raw = serialize(state)
        restored = decode(raw, mode, state)
        if not all(torch.equal(t, restored[k]) for k, t in state.items()):
            raise ArithmeticError("FP32 round trip")
        return raw, (0.0, 0.0)
    if mode != "INT8":
        raise ValueError(mode)
    quant, scales = {}, {}
    for key in sorted(state):
        t = state[key]
        if t.is_floating_point():
            if t.dtype != torch.float32 or not torch.isfinite(t).all():
                raise ValueError(key)
            bound = t.abs().max().item()
            scale = torch.tensor(bound / 127 if bound else 1.0, dtype=torch.float32)
            scales[key] = scale
            quant[key] = (t / scale).round().clamp(-127, 127).to(torch.int8)
        else:
            quant[key] = t
    raw = serialize((quant, torch.stack(list(scales.values()))))
    restored = decode(raw, mode, state)
    max_abs = max_units = 0.0
    for key, t in state.items():
        if key in scales:
            error = (t - restored[key]).abs().max().item()
            max_abs = max(max_abs, error)
            max_units = max(max_units, error / scales[key].item())
        elif not torch.equal(t, restored[key]):
            raise ArithmeticError("Integer buffer changed")
    if max_units > 0.50002:
        raise ArithmeticError(f"Quantization bound exceeded: {max_units}")
    return raw, (max_abs, max_units)
