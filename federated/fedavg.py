import torch


@torch.no_grad()
def fedavg(states, sizes):
    if len(states) != 2 or len(sizes) != 2 or min(sizes) < 1:
        raise ValueError("Expected two nonempty clients")
    first, second = states
    if first.keys() != second.keys():
        raise ValueError("Client state keys differ")
    averaged = {}
    for key, a in first.items():
        b = second[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"Incompatible client tensor {key}")
        if a.is_floating_point():
            averaged[key] = ((a.double()*sizes[0]+b.double()*sizes[1])/sum(sizes)).to(a.dtype)
        elif key.endswith("num_batches_tracked"):
            averaged[key] = torch.maximum(a,b)
        elif torch.equal(a,b):
            averaged[key] = a.clone()
        else:
            raise ValueError(f"Unsupported non-floating buffer {key}")
    return averaged
