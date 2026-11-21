"""Two-client synchronous server. It only loads the common validation bundle."""
import argparse
import hashlib
import json
import socket
import time
import os
from pathlib import Path

import torch
from datasets.compact_cache import atomic_json, sha
from federated.fedavg import fedavg
from federated.process_fedavg import receive, send, read_exact, save_commit
from federated.uav_runtime import (
    MODES,
    apply_reference_norms,
    bundle,
    device_setup,
    early_update,
    load_checkpoint,
    rounding_seed,
    score,
    wire_decode,
    wire_encode,
)
from experiments.sparse_reproduction import cpu_state


def accept(listener, token):
    sock, _ = listener.accept()
    sock.settimeout(90)
    try:
        hello = read_exact(sock, len(token) + 1)
        if hello[:-1] != token or hello[-1] not in (0, 1):
            raise ValueError("Bad handshake")
        return hello[-1], sock
    except BaseException:
        sock.close()
        raise


def run_server(
    listener,
    token,
    start,
    val_path,
    work,
    seed,
    mode,
    device="cuda:0",
    rounds=30,
    patience=0,
    targets=None,
    trace=False,
    edge_batch=None,
    after_commit=None,
):
    if len(token) != 16 or rounds < 1 or patience < 0:
        raise ValueError("Bad session settings")
    device = device_setup(device)
    work.mkdir(parents=True, exist_ok=True)
    if val_path is None and patience:
        raise ValueError("Stopping by validation requires labeled data")
    val = bundle(val_path, "validation", device)[0] if val_path is not None else None
    net = load_checkpoint(start, device)
    config = dict(
        seed=seed,
        mode=mode,
        start=sha(start),
        val=sha(val_path) if val_path is not None else None,
        rounds=rounds,
        patience=patience,
        targets=targets,
    )
    if edge_batch is not None:
        config["edge_batch"] = edge_batch
    rid = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    cp = work / "server.pt"
    if cp.exists():
        ck = torch.load(cp, map_location="cpu", weights_only=True)
        if ck["config"] != config:
            raise ValueError("Checkpoint configuration changed")
    else:
        state = cpu_state(net)
        initial = score(net, val) if val is not None else None
        ck = dict(
            config=config,
            state=state,
            best_state=state,
            best=initial,
            best_round=0,
            stopped=False,
            rows=[
                dict(round=0, miou=initial, up_bytes=0, down_bytes=0, hello_bytes=0, cumulative_bytes=0)
            ],
        )
        save_commit(cp, ck)
    peers = {}
    hello = 0
    try:
        for _ in range(2):
            i, s = accept(listener, token)
            if i in peers:
                raise ValueError("Duplicate client")
            peers[i] = s
            hello += len(token) + 1
        for r in range(len(ck["rows"]), rounds + 1):
            if ck["stopped"]:
                break
            begin = time.perf_counter()
            recovery_s = 0.0
            state = ck["state"]
            raw = wire_encode(state, mode, rounding_seed(seed, r, 0))
            parent = hashlib.sha256(raw).hexdigest()
            states, sizes, losses = [], [], []
            up = down = retries = 0
            for idx in (0, 1):
                for attempt in range(4):
                    try:
                        sent = send(
                            peers[idx],
                            dict(type="global", run=rid, round=r, mode=mode, seed=seed, client=idx,
                                 **({"widths":net.widths,"batch":edge_batch,"start_sha":config["start"]} if edge_batch is not None else {})),
                            raw,
                        )
                        down += sent
                        msg, wire, n = receive(peers[idx])
                        up += n
                        break
                    except (EOFError, OSError):
                        if attempt == 3:
                            raise
                        dropped_at=time.perf_counter()
                        peers[idx].close()
                        who, s = accept(listener, token)
                        if who != idx:
                            raise ValueError("Unexpected reconnect")
                        peers[idx] = s
                        hello += len(token) + 1
                        retries += 1
                        recovery_s += time.perf_counter()-dropped_at
                if (
                    msg["type"],
                    msg["run"],
                    msg["round"],
                    msg["client"],
                    msg["parent"],
                    msg["received"],
                ) != ("update", rid, r, idx, parent, sent) or msg["samples"] < 1:
                    raise ValueError("Wrong update")
                states.append(wire_decode(wire, mode, state))
                sizes.append(msg["samples"])
                losses.append(float(msg["loss"]))
            result = fedavg(states, sizes)
            if mode == "FP32_MATCH":
                result = apply_reference_norms(state, result, targets[r - 1])
            net.load_state_dict(result)
            value = score(net, val) if val is not None else None
            best, br, stop = (early_update(ck["best"], ck["best_round"], value, r, patience or rounds + 1)
                              if val is not None else (None,r,False))
            row = dict(
                round=r,
                miou=value,
                up_bytes=up,
                down_bytes=down,
                hello_bytes=hello,
                retries=retries,
                cumulative_bytes=ck["rows"][-1]["cumulative_bytes"] + up + down + hello,
                client1_loss=losses[0],
                client2_loss=losses[1],
                client1_patches=sizes[0],
                client2_patches=sizes[1],
            )
            if edge_batch is not None:
                update=sum((result[k].double()-state[k].double()).square().sum() for k,_ in net.named_parameters())
                row.update(round_s=time.perf_counter()-begin,update_l2=float(update.sqrt()),
                           reconnect_wait_s=recovery_s,server_pid=os.getpid(),aggregation_device=str(next(iter(result.values())).device))
            hello = 0
            ck.update(
                state=result,
                best_state=result if br == r else ck["best_state"],
                best=best,
                best_round=br,
                stopped=stop,
                rows=ck["rows"] + [row],
            )
            if trace:
                save_commit(work / f"global_{r:02}.pt", dict(state=result))
            save_commit(cp, ck)
            if after_commit is not None:
                after_commit(r)
        control = hello
        for idx, sock in peers.items():
            control += send(sock, dict(type="stop", run=rid, client=idx), b"")
            ack, data, n = receive(sock)
            if (ack["type"], ack["run"], ack["client"], data) != ("stopped", rid, idx, b""):
                raise ValueError("Missing shutdown acknowledgement")
            control += n
        save_commit(work / "best.pt", dict(state=ck["best_state"], widths=net.widths))
        atomic_json(
            work / "done.json",
            dict(
                run=rid,
                mode=mode,
                seed=seed,
                best_round=ck["best_round"],
                stopped=ck["stopped"],
                rows=ck["rows"],
                termination_bytes=control,
            ),
        )
    finally:
        for sock in peers.values():
            sock.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bind", default="127.0.0.1")
    p.add_argument("--port", type=int, default=6543)
    p.add_argument("--validation",type=Path)
    p.add_argument("--edge-batch",type=int,choices=(1,2,4))
    for name in ("start", "work", "token-file"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--mode", choices=[m for m in MODES if m != "FP32_MATCH"], default="INT8")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--patience", type=int, default=3)
    a = p.parse_args()
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((a.bind, a.port))
        server.listen(4)
        server.settimeout(100)
        run_server(
            server,
            a.token_file.read_bytes(),
            a.start,
            a.validation,
            a.work,
            a.seed,
            a.mode,
            a.device,
            a.rounds,
            a.patience if a.validation else 0,edge_batch=a.edge_batch,
        )


if __name__ == "__main__":
    main()
