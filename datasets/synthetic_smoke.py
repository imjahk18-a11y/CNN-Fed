"""Deterministic toy ground/raised-object voxels, strictly for smoke testing."""
import torch
from torch.utils.data import Dataset

from .voxelization import IGNORE


class SyntheticSmoke(Dataset):
    classes = ["toy_surface", "toy_object"]

    def __init__(self, split, n, side, variant, seed):
        self.samples, self.rows = [], []
        generator = torch.Generator().manual_seed(seed+{"train":0,"val":1000,"test":2000}[split])
        grid = torch.arange(side)
        z,y,x = torch.meshgrid(grid, grid, grid, indexing="ij")
        for i in range(n):
            region = i % 2
            height = int(torch.randint(1, side//3+1, (1,), generator=generator))
            center = int(torch.randint(side//4, side*3//4, (1,), generator=generator))
            surface = z == height
            obj = ((x-center).abs() <= side//(4+region)) & ((y-side//2).abs() <= side//4)
            obj = obj & (z > height) & (z <= height+side//(3+region))
            occupied = surface | obj
            target = torch.full((side,)*3, IGNORE, dtype=torch.long)
            target[surface], target[obj] = 0, 1
            features = [occupied.float(), (z+.5)/side*occupied]
            if variant == "B":
                features.extend((obj.float(), surface.float(), .25*occupied))
            self.samples.append((torch.stack(features), target))
            self.rows.append(dict(id=f"toy/{split}/{i}", region=f"{split}-{region}",
                                  hist=[int(surface.sum()),int(obj.sum())]))

    def __getitem__(self, i):
        return self.samples[i]

    def __len__(self):
        return len(self.samples)
