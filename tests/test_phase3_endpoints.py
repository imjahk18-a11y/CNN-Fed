import multiprocessing as mp
import socket

import numpy as np
import pytest
import torch

from federated import fl_server
from federated.uav_client import serve_client
from federated.uav_runtime import (
    apply_reference_norms,
    bundle,
    early_update,
    vector,
    wire_decode,
    wire_encode,
)
from models.active_voxel import SparseUNet


def test_wire_formats_and_stochastic_rounding():
    state = {"x": torch.tensor([127.0, 0.0, 1.5, 2.5, -1.5]), "num_batches_tracked": torch.tensor(3)}
    decoded = lambda mode, seed: wire_decode(wire_encode(state, mode, seed), mode, state)
    assert decoded("INT8", 1)["x"].tolist() == [127.0, 0.0, 2.0, 2.0, -2.0]
    for mode in ("FP32", "FP16", "INT8", "INT8_SR"):
        a = decoded(mode, 42)
        assert torch.equal(a["num_batches_tracked"], state["num_batches_tracked"])
        assert torch.equal(a["x"], decoded(mode, 42)["x"])
    many = {"x": torch.cat([torch.tensor([127.0]), torch.full((20000,), 0.25)])}
    vals = wire_decode(wire_encode(many, "INT8_SR", 45), "INT8_SR", many)["x"][1:]
    assert abs(vals.mean().item() - 0.25) < 0.015
    base = {"weight": torch.tensor([2.0, 1.0]), "a.running_mean": torch.ones(2)}
    local = {k: v + 1 for k, v in base.items()}
    wanted = {"parameters": 0.3, "bn": 1.1}
    matched = apply_reference_norms(base, local, wanted)
    for k in wanted:
        assert float((vector(matched, k) - vector(base, k)).norm()) == pytest.approx(
            wanted[k], abs=1e-6
        )
    assert early_update(0.4, 1, 0.4, 4, 3) == (0.4, 1, True)


def test_server_restart_replays_uncommitted_round(tmp_path, monkeypatch):
    torch.manual_seed(42)
    start = tmp_path / "start.pt"
    torch.save(dict(state=SparseUNet().state_dict()), start)
    token = b"0123456789abcdef"
    coords = np.tile(np.array([[x, 0, 0] for x in range(8)], np.int32), (2, 1))

    def data(role):
        path = tmp_path / (role + ".npz")
        np.savez(
            path,
            role=role,
            bounds=[0, 8, 16],
            coords=coords,
            rgb=np.full((16, 3), 128, np.float32),
            labels=np.arange(16) % 6,
        )
        return path

    validation = data("validation")
    assert bundle(validation, "validation", torch.device("cpu"), training_labels=False)[0][0][2] is None
    folder = tmp_path / "server"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(4)
        listener.settimeout(40)
        host, port = listener.getsockname()
        ctx = mp.get_context("spawn")
        clients = [
            ctx.Process(
                target=serve_client,
                args=(host, port, i, data(f"client{i+1}"), tmp_path / f"client{i+1}", token, "cpu"),
            )
            for i in (0, 1)
        ]

        def run():
            fl_server.run_server(
                listener, token, start, validation, folder, 42, "INT8", "cpu", 2, trace=True
            )

        original = fl_server.save_commit

        def interrupted(path, blob):
            if path.name == "server.pt" and len(blob["rows"]) == 3:
                raise RuntimeError("Injected server restart")
            original(path, blob)

        try:
            for c in clients:
                c.start()
            with monkeypatch.context() as m:
                m.setattr(fl_server, "save_commit", interrupted)
                with pytest.raises(RuntimeError, match="Injected server restart"):
                    run()
            assert len(torch.load(folder / "server.pt", weights_only=True)["rows"]) == 2
            expected = torch.load(folder / "global_02.pt", weights_only=True)["state"]
            run()
            final = torch.load(folder / "server.pt", weights_only=True)
            assert len(final["rows"]) == 3
            assert all(torch.equal(v, final["state"][k]) for k, v in expected.items())
            for c in clients:
                c.join(15)
                assert c.exitcode == 0
        finally:
            for c in clients:
                if c.is_alive():
                    c.terminate()
                    c.join()
