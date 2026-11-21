"""Store only occupied voxels. Materialize channels-first Z,Y,X on demand."""
import numpy as np
import torch

IGNORE = -100


def aggregate(local_xyz, rgb, labels, side, classes):
    """local_xyz is in voxel units relative to the lower patch corner."""
    cells = np.floor(local_xyz).astype(np.int64)
    if not np.all((cells >= 0) & (cells < side)):
        raise ValueError("Point outside its patch")
    flat = np.ravel_multi_index(cells[:, ::-1].T, (side,)*3)
    slots, inverse, counts = np.unique(flat, return_inverse=True, return_counts=True)
    means = [np.bincount(inverse, weights=local_xyz[:, 2]/side)/counts]
    if rgb is not None:
        means.extend(np.bincount(inverse, weights=c)/(255*counts) for c in rgb.T)
    valid = labels != IGNORE
    votes = np.bincount(inverse[valid]*classes + labels[valid], minlength=len(slots)*classes)
    votes = votes.reshape(-1, classes)
    target = votes.argmax(axis=1)
    target[votes.max(axis=1) == 0] = IGNORE
    return slots.astype("int32"), np.asarray(means, dtype="float32"), target.astype("int16")


def dense(slots, features, target, side, variant):
    channels = {"A":2, "B":5}[variant]
    if channels-1 > len(features):
        raise ValueError("Variant B requires an RGB point cloud")
    x = np.zeros((channels, side**3), dtype=np.float32)
    y = np.full(side**3, IGNORE, dtype=np.int64)
    x[0, slots] = 1
    x[1:, slots] = features[:channels-1]
    y[slots] = target
    return torch.from_numpy(x.reshape(channels, side, side, side)), torch.from_numpy(y.reshape((side,)*3))
