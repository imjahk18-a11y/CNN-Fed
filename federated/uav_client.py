"""A standalone client with one private patch bundle, offline inference and replayable updates."""
import argparse
import socket
import os
import time
from pathlib import Path

import numpy as np
import torch

from datasets.compact_cache import sha, atomic_json
from federated.edge_batches import EdgeBatches, cap_allocator, memory, measure_inference, train_epoch
from experiments.distributed_ai import round_seed
from experiments.sparse_reproduction import cpu_state
from federated.process_fedavg import receive, send, save_commit
from federated.uav_runtime import (
    MODES,
    bundle,
    device_setup,
    load_checkpoint,
    local_epoch,
    rounding_seed,
    wire_decode,
    wire_encode,
)
from models.active_voxel import SparseUNet


def serve_client(
    host, port, idx, data_path, work, token, device="cuda:0", trace=False, drop_once=False, edge_batch=None, allocator_gib=2
):
    device = device_setup(device)
    if edge_batch is None:
        data,count=bundle(data_path,f"client{idx+1}",device)
    else:
        if device.type!="cuda": raise ValueError("Edge training requires CUDA")
        cap_allocator(allocator_gib)
        data=EdgeBatches(data_path,f"client{idx+1}",edge_batch,device)
        count=len(data.files)
    data_hash = sha(data_path)
    work.mkdir(parents=True, exist_ok=True)
    cache = work / "last.pt"
    last = torch.load(cache, map_location="cpu", weights_only=True) if cache.exists() else {}
    failures = 0
    while failures < 20:
        try:
            with socket.create_connection((host, port), timeout=30) as sock:
                sock.settimeout(90)
                sock.sendall(token + bytes([idx]))
                while True:
                    msg, raw, down = receive(sock)
                    if msg["client"] != idx:
                        raise ValueError("Client ID mismatch")
                    if msg["type"] == "stop":
                        send(sock, dict(type="stopped", run=msg["run"], client=idx), b"")
                        return
                    if msg["type"] != "global" or msg["mode"] not in MODES:
                        raise ValueError("Unexpected frame")
                    r, seed, mode = msg["round"], msg["seed"], msg["mode"]
                    key = [msg["run"], r, msg["sha256"], data_hash, mode, seed]
                    if edge_batch is not None:
                        if msg.get("batch")!=edge_batch: raise ValueError("Server/client batch mismatch")
                        key += [edge_batch,msg["start_sha"]]
                    replayed = last.get("key") == key
                    if not replayed:
                        if last and last["key"][0] == key[0] and last["key"][1] >= r:
                            raise ValueError("Conflicting parent or stale round")
                        net = SparseUNet(tuple(msg.get("widths",(16,32,16))))
                        net.load_state_dict(wire_decode(raw, mode, net.state_dict()))
                        net = net.to(device)
                        if edge_batch is None:
                            loss=local_epoch(net,data,round_seed(seed,r,idx),device)
                        else:
                            torch.cuda.reset_peak_memory_stats()
                            logits=measure_inference(net,data[0])
                            voxels=len(logits)
                            del logits
                            metrics=train_epoch(net,data,round_seed(seed,r,idx))
                            loss=metrics["loss"]
                            metrics.update(**memory(),pid=os.getpid(),device=str(device),gpu=torch.cuda.get_device_name(0),
                                           inference_voxels=voxels,client=idx+1,round=r,codec=mode,
                                           checkpoint_sha=msg["start_sha"],manifest_sha=data_hash,
                                           widths=net.widths,allocator_limit_gib=allocator_gib)
                            atomic_json(work/f"metrics_{r:02}.json",metrics)
                        state = cpu_state(net)
                        wire = wire_encode(state, mode, rounding_seed(seed, r, idx + 1))
                        last = dict(key=key, reply=wire, loss=loss, state=state, widths=net.widths)
                        save_commit(cache, last)
                        if trace:
                            save_commit(work / f"local_{r:02}.pt", dict(state=state))
                    if drop_once and r == 2 and not (work / "dropped").exists():
                        (work / "dropped").touch()
                        sock.shutdown(socket.SHUT_RDWR)
                        raise ConnectionResetError("Injected loss before update")
                    if edge_batch is not None and replayed:
                        atomic_json(work/f"replay_{r:02}.json",dict(round=r,parent=msg["sha256"],client=idx+1))
                    send(
                        sock,
                        dict(
                            type="update",
                            run=msg["run"],
                            round=r,
                            parent=msg["sha256"],
                            client=idx,
                            samples=count,
                            received=down,
                            loss=f"{last['loss']:.10f}",
                        ),
                        last["reply"],
                    )
                    failures = 0
        except (OSError, EOFError):
            failures += 1
            time.sleep(min(0.2 * failures, 2))
    raise ConnectionError("Server unavailable after retry limit")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--id", type=int, choices=(1, 2), required=True)
    p.add_argument("--device", default="cuda:0")
    sub = p.add_subparsers(dest="action", required=True)
    c = sub.add_parser("connect")
    c.add_argument("--host", default="127.0.0.1")
    c.add_argument("--port", type=int, default=6543)
    c.add_argument("--token-file", type=Path, required=True)
    c.add_argument("--work", type=Path, required=True)
    c.add_argument("--edge-batch",type=int,choices=(1,2,4))
    c.add_argument("--allocator-gib",type=float,default=2)
    inf = sub.add_parser("infer")
    inf.add_argument("--checkpoint", type=Path, required=True)
    inf.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.action == "connect":
        serve_client(
            args.host,
            args.port,
            args.id - 1,
            args.data,
            args.work,
            args.token_file.read_bytes(),
            args.device,edge_batch=args.edge_batch,allocator_gib=args.allocator_gib,
        )
    else:
        device = device_setup(args.device)
        batches, _ = bundle(args.data, f"client{args.id}", device, training_labels=False)
        model = load_checkpoint(args.checkpoint, device)
        with torch.inference_mode():
            predictions = [model(g, x)[1].argmax(1).cpu().numpy() for g, x, _ in batches]
        np.save(args.output, np.concatenate(predictions))


if __name__ == "__main__":
    main()
