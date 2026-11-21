import torch

from models.voxel_3dcnn import VoxelUNet


@torch.no_grad()
def prune(model, ratio, example, *, target_widths=None):
    if not 0 < ratio < 1:
        raise ValueError("Expected a fractional pruning ratio")
    blocks = [model.encoder, model.bridge, model.decoder]
    keep = []
    if target_widths is not None and len(target_widths) != len(blocks):
        raise ValueError('Specify one target width per block')
    for index, block in enumerate(blocks):
        weight = block[0].weight
        width = weight.shape[0]
        n = max(1, width - int(width*ratio+.5))
        if target_widths is not None:
            n = target_widths[index]
            if not 1 <= n <= width:
                raise ValueError("Target width outside the source block")
        order = weight.abs().flatten(1).sum(1).argsort(descending=True, stable=True)
        keep.append(order[:n].sort().values)
    a, b, c = keep
    spec = {**model.spec, "widths":list(map(len, keep))}
    small = VoxelUNet(**spec).to(example.device)
    in_axis = [torch.arange(model.spec["inputs"], device=example.device), a,
               torch.cat((b, a+model.spec["widths"][1]))]
    for src, dst, outputs, inputs in zip(blocks, [small.encoder, small.bridge, small.decoder], keep, in_axis):
        dst[0].weight.copy_(src[0].weight[outputs][:,inputs])
        for key in ("weight","bias","running_mean","running_var"):
            getattr(dst[1],key).copy_(getattr(src[1],key)[outputs])
        dst[1].num_batches_tracked.copy_(src[1].num_batches_tracked)
    small.head.weight.copy_(model.head.weight[:,c])
    small.head.bias.copy_(model.head.bias)
    prev_training = model.training
    baseline_shape = model.eval()(example).shape
    model.train(prev_training)
    result = small.eval()(example)
    if result.shape != baseline_shape or not result.isfinite().all():
        raise RuntimeError("Invalid pruned forward")
    return small, {k:v.tolist() for k,v in zip(("encoder","bridge","decoder"),keep)}
