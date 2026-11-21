"""Measured bidirectional compression and application-shaped loopback FedAvg."""
import csv
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import random
import socket
import time
from contextlib import closing
from pathlib import Path

import torch

from datasets.compact_cache import atomic_json, sha
from datasets.fullz import Scene
from experiments.distributed_ai import (
    OUT as OLD,
    BASE,
    CACHE,
    SCENES,
    SEEDS,
    SIZES,
    batchify,
    config_id,
    init,
    round_seed,
    startfile,
    train_local,
)
from experiments.sparse_reproduction import cpu_state, evaluate
from federated.communication_codec import decode, encode
from federated.fedavg import fedavg
from federated.process_fedavg import accept_update, read_exact, receive, send, save_commit
from models.active_voxel import SparseUNet

OUT = Path("results/distributed_ai_phase2")
PROFILES = ("loopback", "bandwidth", "latency", "disconnect", "straggler")
MODES = ("FP32", "INT8")
RATE = 256_000  # application bytes per second, each direction
ONE_WAY = 0.025
OUTAGE = 0.15
SLOW = 0.15


class ShapedSender:
    def __init__(self, sock, profile):
        self.sock = sock
        self.profile = profile

    def sendall(self, frame):
        if self.profile == "latency":
            time.sleep(ONE_WAY)
        if self.profile != "bandwidth":
            self.sock.sendall(frame)
            return
        start = time.monotonic()
        for at in range(0, len(frame), 4096):
            part = frame[at : at + 4096]
            deadline = start + (at + len(part)) / RATE
            time.sleep(max(0, deadline - time.monotonic()))
            self.sock.sendall(part)


def setup(sock):
    sock.settimeout(120)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


def new_model(pct):
    widths = {0: (16, 32, 16), 20: (13, 26, 13), 40: (10, 19, 10)}
    return SparseUNet(widths[pct])


def dataset(refs):
    return batchify(refs, {s: Scene(CACHE / SCENES[s]) for s in {i for i, _ in refs}})


def drone(address, token, idx, stop):
    init()
    refs = json.loads((OLD / "split.json").read_text())["NonIID"][idx]
    data = dataset(refs)
    injected = set()
    while not stop.is_set():
        with socket.create_connection(address, timeout=120) as sock:
            setup(sock)
            sock.sendall(token + bytes([idx]))
            while not stop.is_set():
                try:
                    meta, raw, n = receive(sock)
                except EOFError:
                    break
                seed, pct, mode, profile_index = meta["config"]
                profile = PROFILES[profile_index]
                rid, v = meta["run"], meta["version"]
                if meta["type"] != "global" or meta["client"] != idx:
                    raise ValueError("Wrong destination")
                if profile == "disconnect" and idx == 1 and v == 2 and rid not in injected:
                    injected.add(rid)
                    sock.shutdown(socket.SHUT_RDWR)
                    time.sleep(OUTAGE)
                    break
                begin = time.perf_counter()
                net = new_model(pct)
                net.load_state_dict(decode(raw, mode, net.state_dict()))
                net.to("cuda:0")
                if profile == "straggler" and idx == 1:
                    time.sleep(SLOW)
                train_start = time.perf_counter()
                _, peak = train_local(net, data, 1, round_seed(seed, v + 1, idx))
                torch.cuda.synchronize()
                train_time = time.perf_counter() - train_start
                local, errors = encode(cpu_state(net), mode)
                # Fixed widths keep telemetry formatting from affecting byte comparisons.
                reply = dict(
                    type="update",
                    run=rid,
                    client=idx,
                    version=v,
                    parent=meta["sha256"],
                    samples=len(refs),
                    received=n,
                    proc=f"{time.perf_counter()-begin:012.6f}",
                    train=f"{train_time:012.6f}",
                    peak=f"{peak:012.5f}",
                    error=f"{errors[0]:015.8e}",
                    units=f"{errors[1]:015.8e}",
                )
                send(ShapedSender(sock, profile), reply, local)


def accept(listener, token):
    sock, _ = listener.accept()
    setup(sock)
    hello = read_exact(sock, 17)
    if hello[:16] != token or hello[16] not in (0, 1):
        sock.close()
        raise ValueError("Invalid client handshake")
    return hello[16], sock


def integrity():
    files = [p for p in OLD.rglob("*") if p.is_file()] + [BASE]
    return {str(p): sha(p) for p in sorted(files)}


def config_name(seed, pct, mode, profile):
    return f"s{seed}_p{pct:02}_{mode}_{PROFILES.index(profile)}"


def verify_prior(seed, pct, version, state, score, log):
    if abs(score["miou"] - float(log[version - 1]["global_miou"])) > 1e-12:
        raise ArithmeticError("Validation differs from the historical FP32 run")
    if version in (10, 30):
        path = OLD / "runs" / config_id("FedAvg", pct, "NonIID", 1, seed) / f"r{version}.pt"
        ref = torch.load(path, map_location="cpu", weights_only=True)["state"]
        if any(not torch.equal(state[k], v) for k, v in ref.items()):
            raise ArithmeticError("Historical FP32 checkpoint mismatch")


def run_one(cfg, peers, listener, token, val, warm_handshake):
    seed, pct, mode, profile = cfg
    rid = config_name(*cfg)
    folder = OUT / "runs" / rid
    folder.mkdir(parents=True, exist_ok=True)
    blob = torch.load(startfile(seed, pct), map_location="cpu", weights_only=True)
    state = blob["state"]
    model = new_model(pct).to("cuda:0")
    model.load_state_dict(state)
    val0 = evaluate(model, val)
    oldfile = OLD / "runs" / config_id("FedAvg", pct, "NonIID", 1, seed) / "rounds.csv"
    with oldfile.open() as f:
        history = list(csv.DictReader(f))
    assert abs(val0["miou"] - float(history[0]["initial_miou"])) < 1e-12
    rounds = 30 if profile == "loopback" else 5
    session_start = time.perf_counter()
    cumulative = warm_handshake
    rows = []
    for v in range(rounds):
        tic = time.perf_counter()
        raw, down_error = encode(state, mode)
        parent = hashlib.sha256(raw).hexdigest()
        accepted, details = {}, []
        up = down = failure = recovery = replay = retry_bytes = hello = 0
        for idx in range(2):
            while True:
                request = dict(
                    type="global",
                    run=rid,
                    client=idx,
                    version=v,
                    config=[seed, pct, mode, PROFILES.index(profile)],
                )
                # Reuse the length-prefixed, hashed model frames of the original transport.
                sent = send(ShapedSender(peers[idx], profile), request, raw)
                down += sent
                try:
                    meta, data, size = receive(peers[idx])
                except EOFError:
                    if (profile, idx, v, failure) != ("disconnect", 1, 2, 0):
                        raise
                    failure += 1
                    peers[idx].close()
                    reconnect_id, peers[idx] = accept(listener, token)
                    if reconnect_id != idx:
                        raise ValueError("Unexpected reconnect")
                    hello += 17
                    replay += 1
                    retry_bytes += sent
                    continue
                up += size
                if meta["received"] != sent or not accept_update(meta, rid, idx, v, parent, accepted):
                    raise ValueError("Invalid or repeated client update")
                accepted[idx] = decode(data, mode, state)
                details.append(meta)
                if failure and idx == 1:
                    recovery += 1
                break
        state = fedavg([accepted[0], accepted[1]], [d["samples"] for d in details])
        model.load_state_dict(state)
        score = evaluate(model, val)
        if mode == "FP32":
            verify_prior(seed, pct, v + 1, state, score, history)
        total = up + down + hello
        cumulative += total
        row = dict(
            round=v + 1,
            miou=score["miou"],
            macro_f1=score["macro_f1"],
            delta=score["miou"] - val0["miou"],
            up_bytes=up,
            down_bytes=down,
            handshake_bytes=hello + (warm_handshake if not v else 0),
            total_bytes=total + (warm_handshake if not v else 0),
            cumulative_bytes=cumulative,
            failures=failure,
            recovered=recovery,
            retransmissions=replay,
            retry_frame_bytes=retry_bytes,
            down_max_error=down_error[0],
            down_max_step_fraction=down_error[1],
            up_max_error=max(float(d["error"]) for d in details),
            up_max_step_fraction=max(float(d["units"]) for d in details),
        )
        for idx, d in enumerate(details, 1):
            for key in ("proc", "train", "peak"):
                row[f"client{idx}_{key}"] = float(d[key])
        row["round_s"] = time.perf_counter() - tic
        row["elapsed_s"] = time.perf_counter() - session_start
        rows.append(row)
    elapsed = time.perf_counter() - session_start
    save_commit(folder / "final.pt", dict(state=state, widths=blob["widths"]))
    atomic_json(folder / "done.json", dict(config=cfg, initial=val0, rows=rows, elapsed_s=elapsed))
    print(rid, rounds, f"{score['miou']:.4f}", f"{elapsed:.1f}s", flush=True)


def main():
    init()
    OUT.mkdir(exist_ok=True)
    original = integrity()
    split = json.loads((OLD / "split.json").read_text())
    for client in split["NonIID"]:
        assert len(set(map(tuple, client))) == 96
    assert {s for s, _ in split["val"]} == {12} and len(split["val"]) == 144
    assert {s for s, _ in split["NonIID"][0]} == {3} and {s for s, _ in split["NonIID"][1]} == {4}
    for seed, pct in itertools.product(SEEDS, SIZES):
        old_job = OLD / "runs" / config_id("FedAvg", pct, "NonIID", 1, seed) / "done.json"
        previous = json.loads(old_job.read_text())
        assert previous["split_sha256"] == sha(OLD / "split.json") and previous["start_sha256"] == sha(
            startfile(seed, pct)
        )
    assert sha(BASE) == json.loads((OLD / "protocol.json").read_text())["checkpoint"]
    protocol = dict(
        old_hashes=original,
        target="initial mIoU + 0.01 for 3 consecutive rounds",
        profiles=PROFILES,
        bytes_per_second=RATE,
        one_way_delay_s=ONE_WAY,
        outage_s=OUTAGE,
        client2_slow_s=SLOW,
        primary="NonIID E1",
        network_probe_rounds=5,
        clean_rounds=30,
        seeds=SEEDS,
        gpu=torch.cuda.get_device_name(0),
        torch=str(torch.__version__),
        split_sha256=sha(OLD / "split.json"),
    )
    manifest = OUT / "protocol.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()) == json.loads(json.dumps(protocol))
    else:
        atomic_json(manifest, protocol)
    configs = list(itertools.product(SEEDS, SIZES, MODES, PROFILES))
    random.Random(2026).shuffle(configs)
    # First compare both codecs on a known checkpoint, then interleave the remaining conditions.
    smoke = [(42, 0, m, "loopback") for m in MODES]
    configs = smoke + [c for c in configs if c not in smoke]
    configs = [c for c in configs if not (OUT / "runs" / config_name(*c) / "done.json").exists()]
    if not configs:
        atomic_json(OUT / "integrity.json", dict(old_files_verified=len(original), split=sha(OLD / "split.json")))
        return
    val = dataset(split["val"])
    ctx = mp.get_context("spawn")
    stop = ctx.Event()
    token = os.urandom(16)
    with closing(socket.socket()) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(4)
        server.settimeout(120)
        procs = [
            ctx.Process(target=drone, args=(server.getsockname(), token, i, stop)) for i in range(2)
        ]
        peers = {}
        try:
            for p in procs:
                p.start()
            atomic_json(
                OUT / "processes.json", dict(server=os.getpid(), clients=[p.pid for p in procs])
            )
            for cfg in configs:
                for _ in range(2):
                    idx, sock = accept(server, token)
                    if idx in peers:
                        raise ValueError("Duplicate connection")
                    peers[idx] = sock
                run_one(cfg, peers, server, token, val, 34)
                for sock in peers.values():
                    sock.close()
                peers.clear()
        finally:
            stop.set()
            for sock in peers.values():
                sock.close()
            server.close()
            for p in procs:
                p.join(3)
                if p.is_alive():
                    p.terminate()
                    p.join()
    assert integrity() == original
    atomic_json(
        OUT / "integrity.json", dict(old_files_verified=len(original), split=sha(OLD / "split.json"))
    )


if __name__ == "__main__":
    main()
