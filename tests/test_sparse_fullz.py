import unittest
import numpy as np
import torch

from datasets.fullz import common, representatives
from models.active_voxel import KERNEL, SparseUNet, geometry, prune_channels


class SparseTests(unittest.TestCase):
    def test_mapping(self):
        raw = np.r_[0:16, 17:20]
        np.testing.assert_array_equal(
            common(raw), [0, 1, 2, 2, 2, 3, 3, 1, 3, 1, 1, 4, 4, 1, 5, 0, 1, 0, 0]
        )
        for unknown in (16, 20, -1, 256):
            with self.assertRaises(ValueError):
                common([unknown])

    def test_independent_votes(self):
        groups = np.array([0] * 5 + [1] * 5)
        raw = np.array([2, 3, 4, 1, 1, 0, 0, 1, 1, 2])
        red = np.array([16, 31, 32, 33, 255, 0, 1, 15, 16, 16])
        rgb = np.stack((red, red * 0, red * 0), axis=1).astype("u1")
        d = np.array([0.2, 0.3, 0.4, 0.1, 0.01, 0.2, 0.3, 0.4, 0.1, 0.01])
        winners, colors = representatives(groups, raw, rgb, d)
        np.testing.assert_array_equal(winners, [1, 1])
        np.testing.assert_allclose(colors, [[32.5, 0, 0], [16 / 3, 0, 0]])
        _, other = representatives(groups, raw[::-1], rgb, d)
        np.testing.assert_array_equal(colors, other)

    def test_sparse_reference_and_pruning(self):
        coords = torch.tensor(
            [[0, 0, 0, 166], [1, 0, 0, 166], [0, 0, 0, 165], [0, 1, 0, 166], [1, 0, 0, 0]]
        )
        g = geometry(coords)
        torch.testing.assert_close(g.coarse[g.parent][:, 1:], coords[:, 1:] // 2)
        torch.testing.assert_close(g.coarse[g.parent][:, 0], coords[:, 0])
        model = SparseUNet()
        x = torch.randn(len(coords), 3)
        actual = model.encoder.convolve(x, g.fine_edges)
        ref = torch.zeros_like(actual)
        for d, row in enumerate(coords):
            for k, delta in enumerate(KERNEL):
                target = row + torch.tensor((0, *delta))
                for s in torch.where((coords == target).all(1))[0]:
                    ref[d] += model.encoder.weight[k] @ x[s]
        torch.testing.assert_close(actual, ref)
        order = torch.tensor([3, 1, 2, 0, 4])
        for ratio in (0, 0.2, 0.4):
            net = prune_channels(model, ratio).eval()
            pos, scores = net(g, x)
            torch.testing.assert_close(pos, coords)
            torch.testing.assert_close(net(geometry(coords[order]), x[order])[1], scores[order])
            if ratio == 0:
                torch.testing.assert_close(scores, model.eval()(g, x)[1])
            else:
                self.assertLess(net.macs(g), model.macs(g))
            scores.sum().backward()
            self.assertTrue(all(p.grad is not None for p in net.parameters()))
        with self.assertRaises(ValueError):
            geometry(coords[[0, 0]])

    def test_scene_roundtrip(self):
        import tempfile
        from pathlib import Path
        from datasets.fullz import Scene, convert

        dtype = np.dtype(
            [(k, "<f8") for k in "xyz"] + [(k, "u1") for k in ("red", "green", "blue", "class")]
        )
        pts = np.array(
            [
                (-0.29, 0.01, -0.29, 16, 48, 255, 3),
                (-0.14, 0.14, -0.14, 31, 60, 244, 1),
                (-0.25, 0.12, 49.55, 17, 50, 243, 8),
                (9.31, 0.01, -0.29, 2, 4, 6, 14),
            ],
            dtype=dtype,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file = root / "test.ply"
            with file.open("wb") as f:
                fields = ["property double " + k for k in "xyz"] + [
                    "property uchar " + k for k in dtype.names[3:]
                ]
                f.write(
                    (
                        "\n".join(
                            [
                                "ply",
                                "format binary_little_endian 1.0",
                                "element vertex 4",
                                *fields,
                                "end_header",
                            ]
                        )
                        + "\n"
                    ).encode()
                )
                f.write(pts.tobytes())
            convert(file, root / "cache")
            scene = Scene(root / "cache")
            np.testing.assert_allclose(scene.meta["origin"], [-0.3, 0, -0.3])
            self.assertEqual(scene.meta["shape"][2], 167)
            self.assertEqual(scene.meta["voxels"], 3)
            self.assertEqual(len(scene), 2)
            first, rgb, labels = scene[0]
            np.testing.assert_array_equal(first, [[0, 0, 0], [0, 0, 166]])
            np.testing.assert_array_equal(scene.raw, [1, 8, 14])
            np.testing.assert_array_equal(labels, [1, 3])
            np.testing.assert_array_equal(rgb[0], [23.5, 54, 249.5])
            np.testing.assert_array_equal(scene.tile_xy, [[0, 0], [1, 0]])
            np.testing.assert_array_equal(scene[1][0], [[0, 0, 0]])

    def test_pruned_skip_matches_explicit_mask(self):
        torch.manual_seed(5)
        net = SparseUNet().eval()
        coords = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0], [0, 3, 4, 100]])
        g = geometry(coords)
        x = torch.randn(4, 3)
        for ratio in (0.2, 0.4):
            hooks = []
            for block in (net.encoder, net.bridge, net.decoder):
                scores = block.weight.abs().sum((0, 2))
                keep = round(len(scores) * (1 - ratio))
                mask = torch.zeros_like(scores)
                mask[scores.argsort(descending=True, stable=True)[:keep]] = 1
                hooks.append(block.register_forward_hook(lambda m, args, out, mask=mask: out * mask))
            try:
                ref = net(g, x)[1]
            finally:
                for handle in hooks:
                    handle.remove()
            small = prune_channels(net, ratio).eval()
            torch.testing.assert_close(small(g, x)[1], ref)

    def test_single_occupied_voxel(self):
        g = geometry(torch.tensor([[0, 1, 2, 80]]))
        net = SparseUNet().train()
        pos, logits = net(g, torch.ones(1, 3))
        self.assertEqual(pos.tolist(), [[0, 1, 2, 80]])
        logits.sum().backward()
        self.assertTrue(all(p.grad is not None for p in net.parameters()))

    def test_rgb_joint_bins_and_exact_distance_tie(self):
        g = np.zeros(5, dtype=np.int64)
        raw = [14, 14, 11, 11, 7]
        rgb = np.array([[32, 0, 15], [33, 0, 16], [39, 0, 17], [32, 16, 15], [32, 17, 14]], np.uint8)
        labels, mean = representatives(g, raw, rgb, np.array([0.2, 0.5, 0.1, 0.1, 0.0]))
        # Raw tie 11 vs 14 selects 11. RGB tie selects green 16..31, blue 0..15.
        self.assertEqual(labels.tolist(), [11])
        np.testing.assert_array_equal(mean, [[32, 16.5, 14.5]])
        labels, mean = representatives(g, raw, rgb, np.ones(5))
        self.assertEqual(labels.tolist(), [14])
        np.testing.assert_array_equal(mean, [[36, 0, 16.5]])
