"""Localhost socket transport for a CPU server and two independent CUDA client processes."""
import argparse
import csv
import hashlib
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import socket
import struct
import time

import torch

from datasets.compact_cache import atomic_json, sha
from datasets.fullz import Scene
from experiments.distributed_ai import (
    OUT,
    SCENES,
    CACHE,
    SIZES,
    batchify,
    init,
    round_seed,
    startfile,
    train_local,
    config_id,
)
from experiments.sparse_reproduction import cpu_state, write_csv
from federated.fedavg import fedavg
from models.active_voxel import SparseUNet

PREFIX = struct.Struct("!II")
MAX_BYTES = 2**20


def serialize(state):
    buf = io.BytesIO()
    torch.save(state, buf)
    return buf.getvalue()


def unpack_state(data):
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)


def read_exact(sock, length):
    data = bytearray()
    while len(data) < length:
        piece = sock.recv(length - len(data))
        if not piece:
            raise EOFError("Socket closed before the next complete frame")
        data.extend(piece)
    return bytes(data)


def send(sock, meta, data):
    head = json.dumps(
        dict(**meta, sha256=hashlib.sha256(data).hexdigest()), sort_keys=True, separators=(",", ":")
    ).encode()
    frame = PREFIX.pack(len(head), len(data)) + head + data
    sock.sendall(frame)
    return len(frame)


def receive(sock):
    a, b = PREFIX.unpack(read_exact(sock, PREFIX.size))
    if a > 4096 or b > MAX_BYTES:
        raise ValueError("Oversized model frame")
    meta = json.loads(read_exact(sock, a))
    data = read_exact(sock, b)
    if hashlib.sha256(data).hexdigest() != meta["sha256"]:
        raise ValueError("Corrupt model frame")
    return meta, data, PREFIX.size + a + b


def drone(address, token, idx, pct, duplicate_round):
    init()
    split = json.loads((OUT / "split.json").read_text())
    refs = split["NonIID"][idx]
    scenes = {i: Scene(CACHE / SCENES[i]) for i in {s for s, p in refs}}
    data = batchify(refs, scenes)
    with socket.create_connection(address, timeout=120) as sock:
        sock.settimeout(120)
        # Count the fixed handshake separately from model frames.
        sock.sendall(token + bytes([idx]))
        last = -1
        while True:
            try:
                meta, raw, down = receive(sock)
            except EOFError:
                return
            r = meta["version"]
            if (
                meta["run"] != f"p{pct}"
                or meta["client"] != idx
                or meta["type"] != "global"
                or r <= last
            ):
                raise ValueError("Stale or misdirected global model")
            last = r
            net = SparseUNet(tuple(meta["widths"])).to("cuda:0")
            net.load_state_dict(unpack_state(raw))
            train_local(net, data, 1, round_seed(42, r + 1, idx))
            local = serialize(cpu_state(net))
            reply = dict(
                run=meta["run"],
                client=idx,
                version=r,
                type="update",
                parent=meta["sha256"],
                samples=len(refs),
                received=down,
            )
            send(sock, reply, local)
            if duplicate_round == r + 1 and idx == 0:
                send(sock, reply, local)


def accept_update(meta, run, idx, version, parent, seen):
    if meta["type"] != "update" or meta["run"] != run or meta["client"] != idx:
        raise ValueError("Update identity mismatch")
    if meta["version"] < version or idx in seen:
        return False
    if meta["version"] != version or meta["parent"] != parent or meta["samples"] != 96:
        raise ValueError("Update base version or sample count mismatch")
    return True


def save_commit(path, blob):
    tmp = path.with_suffix(".writing")
    torch.save(blob, tmp)
    tmp.replace(path)


class ProbeInterruption(Exception):
    pass


def session(pct, folder, interrupt, duplicate):
    # Only a commit with two valid contributions advances the global version.
    commit = folder / "committed.pt"
    start = torch.load(startfile(42, pct), map_location="cpu", weights_only=True)
    current = (
        torch.load(commit, map_location="cpu", weights_only=True)
        if commit.exists()
        else dict(version=0, state=start["state"], rows=[])
    )
    signature = [sha(startfile(42, pct)), sha(OUT / "split.json")]
    if commit.exists() and current.get("signature") != signature:
        raise ValueError("Stored commit has a different model/split")
    if current["version"] == 10:
        return current
    initial_version = current["version"]
    handshake = 0
    token = os.urandom(16)
    ctx = mp.get_context("spawn")
    procs = []
    peers = {}
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(2)
        listener.settimeout(120)
        address = listener.getsockname()
        try:
            for idx in range(2):
                p = ctx.Process(target=drone, args=(address, token, idx, pct, 4 if duplicate else -1))
                p.start()
                procs.append(p)
            for _ in range(2):
                sock, _ = listener.accept()
                sock.settimeout(120)
                hello = read_exact(sock, 17)
                handshake += len(hello)
                if hello[:16] != token or hello[16] not in (0, 1) or hello[16] in peers:
                    raise ValueError("Bad client handshake")
                peers[hello[16]] = sock
            for version in range(current["version"], 10):
                tic = time.perf_counter()
                raw = serialize(current["state"])
                parent = hashlib.sha256(raw).hexdigest()
                accepted = {}
                up = down = dup_bytes = 0
                for idx in range(2):
                    sent = send(
                        peers[idx],
                        dict(
                            type="global",
                            run=f"p{pct}",
                            client=idx,
                            version=version,
                            widths=start["widths"],
                        ),
                        raw,
                    )
                    down += sent
                    while True:
                        meta, local, got = receive(peers[idx])
                        up += got
                        if accept_update(meta, f"p{pct}", idx, version, parent, accepted):
                            break
                        dup_bytes += got
                    if meta["received"] != sent:
                        raise ValueError("Endpoint byte counts differ")
                    update = unpack_state(local)
                    if update.keys() != current["state"].keys() or any(
                        v.shape != current["state"][k].shape
                        or v.dtype != current["state"][k].dtype
                        or not torch.isfinite(v).all()
                        for k, v in update.items()
                    ):
                        raise ValueError("Incompatible update state")
                    accepted[idx] = update
                    if interrupt and version == 2 and idx == 0:
                        atomic_json(
                            folder / "rollback.json",
                            dict(
                                last_committed=2,
                                discarded_partial_round=3,
                                accepted_clients=1,
                                download_bytes=down,
                                upload_bytes=up,
                                old_commit_sha256=sha(commit),
                            ),
                        )
                        raise ProbeInterruption()
                state = fedavg([accepted[0], accepted[1]], [96, 96])
                row = dict(
                    pruning_pct=pct,
                    round=version + 1,
                    server_pid=os.getpid(),
                    client1_pid=procs[0].pid,
                    client2_pid=procs[1].pid,
                    upload_bytes=up,
                    download_bytes=down,
                    total_bytes=up + down,
                    duplicate_bytes=dup_bytes,
                    handshake_bytes=handshake if version == initial_version else 0,
                    round_s=time.perf_counter() - tic,
                )
                current = dict(
                    version=version + 1, state=state, signature=signature, rows=current["rows"] + [row]
                )
                save_commit(commit, current)
        finally:
            for sock in peers.values():
                sock.close()
            for p in procs:
                p.join(15)
                if p.is_alive():
                    p.terminate()
                    p.join()
    return current


def main():
    rows = []
    checks = {}
    for pct in SIZES:
        folder = OUT / "transport" / f"p{pct}"
        folder.mkdir(parents=True, exist_ok=True)
        # Retry after unexpected disconnect by starting the command again at the last atomic commit.
        try:
            result = session(pct, folder, pct == 0 and not (folder / "rollback.json").exists(), False)
        except ProbeInterruption:
            if (
                sha(folder / "committed.pt")
                != json.loads((folder / "rollback.json").read_text())["old_commit_sha256"]
            ):
                raise AssertionError("Partial round was committed")
            result = session(pct, folder, False, True)
        ref = OUT / "runs" / config_id("FedAvg", pct, "NonIID", 1, 42) / "r10.pt"
        sim = torch.load(ref, map_location="cpu", weights_only=True)["state"]
        delta = max(float((sim[k] - v).abs().max()) for k, v in result["state"].items())
        for k, v in result["state"].items():
            torch.testing.assert_close(sim[k], v, rtol=1e-5, atol=1e-6)
        rows.extend(result["rows"])
        checks[str(pct)] = dict(max_delta=delta, rounds=10, simulation_checkpoint_sha256=sha(ref))
    write_csv(OUT / "transport_rounds.csv", rows)
    atomic_json(
        OUT / "transport_checks.json",
        dict(
            models=checks,
            protocol="loopback TCP with length-prefixed JSON and torch.save weights; both directions include framing",
            handshakes="17 application bytes per client per connection; excluded from model-round rows",
            server="CPU process",
            clients="two spawned CUDA processes, sequential on RTX A6000",
            rollback_test="drop partial round 3 and recover from round 2",
            duplicate_test="replayed client 1 round 4 update ignored at round 5",
        ),
    )


if __name__ == "__main__":
    main()
