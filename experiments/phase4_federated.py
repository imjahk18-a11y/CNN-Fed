"""Paired FL comparisons and BN interventions using serialized, measured loopback frames."""
import hashlib
import argparse
import json
import os
import socket
import time
from contextlib import ExitStack

import numpy as np
import torch

from datasets.compact_cache import atomic_json, sha
from datasets.fullz import Scene
from experiments.distributed_ai import (
    CACHE,
    SCENES,
    batchify,
    init,
    round_seed,
    train_local,
    startfile as phase3_start,
)
from experiments.phase4_pretrain import OUT, SEEDS, FL_SEED, start_path, artifact_hashes
from experiments.sparse_reproduction import cpu_state, evaluate, scores_from_confusion
from federated.fedavg import fedavg
from federated.process_fedavg import read_exact, receive, send, save_commit
from federated.uav_runtime import apply_reference_norms, group, vector, wire_decode, wire_encode
from models.active_voxel import SparseUNet

POLICIES = ("aggregate", "frozen", "private")
MODES = ("INT8", "FP32", "FP32_MATCH")


def folder(f, seed, bn, mode):
    return OUT / "fl" / f / f"s{seed}" / bn / mode


def read_state(p):
    return torch.load(p, map_location="cpu", weights_only=True)["state"]


def moments(k):
    return k.endswith(("running_mean", "running_var", "num_batches_tracked"))


def shared(k, bn):
    if bn == "frozen":
        return not moments(k)
    if bn == "private":
        return ".norm." not in k
    return True


def stage_state(net, state):
    net.load_state_dict(state)
    return net


def hold_stats(module, args):
    module.eval()


@torch.inference_mode()
def recalibrate(net, data):
    """Layerwise exact population moments over occupied sites in calibration inputs."""
    net.eval()
    for layer in (net.encoder.norm, net.bridge.norm, net.decoder.norm):
        count = 0
        total = torch.zeros_like(layer.running_mean, dtype=torch.float64)
        square = torch.zeros_like(total)

        def observe(module, args):
            nonlocal count
            x = args[0].double()
            count += len(x)
            total.add_(x.sum(0))
            square.add_(x.square().sum(0))

        hook = layer.register_forward_pre_hook(observe)
        try:
            for g, x in data:
                net(g, x)
        finally:
            hook.remove()
        mean = total / count
        layer.running_mean.copy_(mean)
        layer.running_var.copy_((square / count - mean.square()).clamp_min(0))
    return cpu_state(net)


class Links:
    """Two TCP connections, two logical clients and server in one CUDA process."""

    def __init__(self, stack):
        listener = stack.enter_context(socket.socket())
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        self.pairs = []
        self.hello = 0
        for idx in range(2):
            client = stack.enter_context(socket.create_connection(listener.getsockname(), timeout=20))
            server = stack.enter_context(listener.accept()[0])
            server.settimeout(20)
            header = os.urandom(16) + bytes([idx])
            client.sendall(header)
            assert read_exact(server, len(header)) == header
            self.hello += len(header)
            self.pairs.append((server, client))

    def frame(self, idx, upload, meta, raw):
        server, client = self.pairs[idx]
        tx, rx = (client, server) if upload else (server, client)
        n = send(tx, meta, raw)
        m, data, got = receive(rx)
        assert n == got and all(m[k] == v for k, v in meta.items())
        return data, n

    def closing_bytes(self):
        return sum(
            self.frame(i, up, dict(kind="closed" if up else "stop", client=i), b"")[1]
            for i in (0, 1)
            for up in (False, True)
        )


def cosine(a, b):
    size = a.norm() * b.norm()
    return float(a.dot(b) / size) if size else None


def result_summary(net, state, data, private=None, links=None, r=0):
    if private is None:
        return evaluate(stage_state(net, state), data)["miou"], 0, []
    matrices = []
    traffic = 0
    for i, p in enumerate(private):
        conf = np.asarray(evaluate(stage_state(net, {**state, **p}), data)["confusion"], dtype="<i8")
        if links:
            payload, n = links.frame(
                i, True, dict(kind="validation_confusion", client=i, round=r), conf.tobytes()
            )
            conf = np.frombuffer(payload, dtype="<i8").reshape(6, 6)
            traffic += n
        matrices.append(conf)
    return (
        scores_from_confusion(sum(matrices))["miou"],
        traffic,
        [scores_from_confusion(c)["miou"] for c in matrices],
    )


def train_fold(f, seed, bn, mode, sets, reference=None):
    dest = folder(f, seed, bn, mode)
    done = dest / "done.json"
    if done.exists():
        return json.loads(done.read_text())
    dest.mkdir(parents=True, exist_ok=True)
    net = SparseUNet().cuda()
    source = phase3_start(42, 0) if f == "phase3_replay" else start_path(f, seed)
    initial = read_state(source)
    state = dict(initial)
    private = [{k: t for k, t in initial.items() if not shared(k, bn)} for _ in range(2)]
    val = sets["validation"]
    cal = sets["calibration"]
    model_miou = evaluate(stage_state(net, state), val)["miou"]
    rec0 = None
    if bn == "aggregate":
        rec0 = recalibrate(net, cal)
    rows = [
        dict(
            round=0,
            miou=model_miou,
            model_bytes=0,
            evaluation_bytes=0,
            hello_bytes=0,
            cumulative_bytes=0,
            recal_miou=evaluate(stage_state(net, rec0), val)["miou"] if rec0 is not None else None,
        )
    ]

    def snapshot(r, rec):
        save_commit(
            dest / f"r{r:02}.pt",
            dict(
                state=state,
                private=private if bn == "private" else None,
                recal_moments={k: v for k, v in rec.items() if moments(k)} if rec else None,
            ),
        )

    snapshot(0, rec0)
    with ExitStack() as stack:
        links = Links(stack)
        for r in range(1, 31):
            template = {k: v for k, v in state.items() if shared(k, bn)}
            down = wire_encode(template, mode, 0)
            parent = hashlib.sha256(down).hexdigest()
            dirs = []
            post = []
            ups = []
            losses = []
            upload = download = 0
            for i in (0, 1):
                raw, n = links.frame(i, False, dict(kind="global", client=i, round=r, codec=mode), down)
                download += n
                recv = wire_decode(raw, mode, template)
                start = {**recv, **private[i]}
                stage_state(net, start)
                hooks = [
                    m.register_forward_pre_hook(hold_stats)
                    for m in net.modules()
                    if bn == "frozen" and isinstance(m, torch.nn.BatchNorm1d)
                ]
                try:
                    loss, _ = train_local(net, sets[f"client{i+1}"], 1, round_seed(FL_SEED, r, i))
                finally:
                    for h in hooks:
                        h.remove()
                losses.append(loss)
                local = cpu_state(net)
                if bn == "frozen":
                    assert all(torch.equal(local[k], v) for k, v in private[i].items())
                if bn == "private":
                    private[i] = {k: v for k, v in local.items() if not shared(k, bn)}
                raw_local = {k: local[k] for k in template}
                w, n = links.frame(
                    i,
                    True,
                    dict(kind="update", client=i, round=r, parent=parent, samples=96),
                    wire_encode(raw_local, mode, 0),
                )
                upload += n
                restored = wire_decode(w, mode, template)
                ups.append(restored)
                origin = vector(recv, "parameters")
                dirs.append(vector(raw_local, "parameters") - origin)
                post.append(vector(restored, "parameters") - origin)
            candidate = {**state, **fedavg(ups, [96, 96])}
            old = state
            if mode == "FP32_MATCH":
                target = {g: reference["rows"][r][f"global_{g}_l2"] for g in ("parameters", "bn")}
                state = apply_reference_norms(old, candidate, target)
            else:
                state = candidate
            # With private BN, the state placeholder for unshared BN remains the initialization.
            delta = {g: vector(state, g) - vector(old, g) for g in ("parameters", "bn")}
            shared_direction = vector({k: state[k] for k in template}, "parameters") - vector(
                template, "parameters"
            )
            if mode == "FP32_MATCH":
                assert all(abs(float(delta[g].norm()) - t) < 1e-5 for g, t in target.items())
            value, extra, client_scores = result_summary(
                net, state, val, private if bn == "private" else None, links, r
            )
            rec = None
            if bn == "aggregate":
                rec = recalibrate(stage_state(net, state), cal)
                assert all(torch.equal(v, rec[k]) for k, v in state.items() if not moments(k))
            handshake = links.hello if r == 1 else 0
            rows.append(
                dict(
                    round=r,
                    miou=value,
                    recal_miou=evaluate(stage_state(net, rec), val)["miou"] if rec else None,
                    client1_loss=losses[0],
                    client2_loss=losses[1],
                    upload_bytes=upload,
                    download_bytes=download,
                    model_bytes=upload + download,
                    evaluation_bytes=extra,
                    hello_bytes=handshake,
                    cumulative_bytes=rows[-1]["cumulative_bytes"]
                    + upload
                    + download
                    + extra
                    + handshake,
                    global_parameters_l2=float(delta["parameters"].norm()),
                    global_bn_l2=float(delta["bn"].norm()),
                    raw_fedavg_parameters_l2=float(
                        (vector(candidate, "parameters") - vector(old, "parameters")).norm()
                    ),
                    client1_update_l2=float(dirs[0].norm()),
                    client2_update_l2=float(dirs[1].norm()),
                    client_update_cos=cosine(*dirs),
                    client_decoded_cos=cosine(*post),
                    client1_global_cos=cosine(dirs[0], shared_direction),
                    client2_global_cos=cosine(dirs[1], shared_direction),
                    personalized_miou=client_scores,
                )
            )
            snapshot(r, rec)
        closing = links.closing_bytes()
    data = dict(
        fold=f, seed=seed, bn=bn, mode=mode, start_sha=sha(source), closing_bytes=closing, rows=rows
    )
    atomic_json(done, data)
    print(f, seed, bn, mode, "30R", round(value, 5), flush=True)
    return data


def main(follow=False):
    init()
    splits = json.loads((OUT / "splits.json").read_text())
    scenes = {i: Scene(CACHE / s) for i, s in SCENES.items()}
    for f, roles in splits.items():
        sets = {
            k: batchify(roles[k], scenes) for k in ("client1", "client2", "validation", "calibration")
        }
        sets["calibration"] = [(g, x) for g, x, _ in sets["calibration"]]
        if f == "reference":
            archive = OUT.parent / "distributed_ai_phase3" / "runs"
            for mode in MODES:
                replay = train_fold(
                    "phase3_replay", 42, "aggregate", mode, sets, ref if mode == "FP32_MATCH" else None
                )
                if mode == "INT8":
                    ref = replay
                old = archive / f"s42_{mode}"
                scores = json.loads((old / "done.json").read_text())["rows"]
                assert [r["miou"] for r in scores] == [r["miou"] for r in replay["rows"]]
                expected = read_state(old / "server.pt")
                actual = read_state(folder("phase3_replay", 42, "aggregate", mode) / "r30.pt")
                assert all(torch.equal(expected[k], v) for k, v in actual.items())
        for seed in SEEDS[f]:
            ready = start_path(f, seed).with_name("done.json")
            deadline = time.monotonic() + 600
            while follow and not ready.exists() and time.monotonic() < deadline:
                time.sleep(2)
            if not ready.exists():
                raise FileNotFoundError(f"Complete pretraining first: {ready}")
            for bn in POLICIES:
                ref = train_fold(f, seed, bn, "INT8", sets)
                train_fold(f, seed, bn, "FP32", sets)
                train_fold(f, seed, bn, "FP32_MATCH", sets, ref)
    assert artifact_hashes() == json.loads((OUT / "prior_hashes.json").read_text())
    atomic_json(
        OUT / "training_complete.json",
        dict(
            trajectories=54,
            phase3_replays=3,
            independent_checkpoints=6,
            checkpoint_hashes={str(p): sha(p) for p in sorted((OUT / "fl").rglob("*.pt"))},
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--follow-pretraining", action="store_true")
    main(parser.parse_args().follow_pretraining)
