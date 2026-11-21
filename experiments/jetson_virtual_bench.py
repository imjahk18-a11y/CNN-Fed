"""Paired single-client A6000 benchmarks on the provisioned Non-IID patches."""
import gc
import json
import multiprocessing as mp
import threading
import time

import numpy as np
import torch

from datasets.compact_cache import atomic_json, sha
from experiments.distributed_ai import startfile
from experiments.jetson_virtual import CONFIG, OUT
from federated.edge_batches import EdgeBatches, cap_allocator, measure_inference, memory, train_epoch
from federated.uav_runtime import device_setup, load_checkpoint


def budget_check(sample, nominal, reserve, cap):
    driver = sample["driver_sample_mib"]
    overhead = None if driver is None else max(0, driver - sample["current_reserved_mib"])
    accounted = None if overhead is None else sample["peak_reserved_mib"] + sample["peak_rss_mib"] + overhead
    status = "INCOMPLETE" if accounted is None else (
        "PASS" if accounted <= (nominal - reserve) * 1024 and sample["peak_reserved_mib"] <= cap * 1024 else "EXCEEDS_BUDGET"
    )
    return dict(working_set_proxy_mib=accounted, budget_status=status)


def summarize(samples):
    return dict(inference_count=len(samples), p50_ms=float(np.percentile(samples, 50)),
                p95_ms=float(np.percentile(samples, 95)))


def inference(net, patch):
    g, x, _ = patch
    start = time.perf_counter()
    net(g, x)
    torch.cuda.current_stream().synchronize()
    return (time.perf_counter() - start) * 1000


def paired(checkpoint, data, probe, seed):
    # Build common geometry and warm the libraries before timing.
    warm = load_checkpoint(checkpoint, data.device)
    measure_inference(warm, probe)
    torch.optim.AdamW(warm.parameters())
    g, x, _ = probe
    warm.train()
    warm(g, x)[1].sum().backward()
    del warm
    gc.collect()
    torch.cuda.empty_cache()
    rows = []
    for condition in ("inference_only", "training_only", "concurrent"):
        train = load_checkpoint(checkpoint, data.device) if condition != "inference_only" else None
        infer = load_checkpoint(checkpoint, data.device) if condition != "training_only" else None
        if infer is not None:
            with torch.inference_mode():
                for _ in range(5):
                    inference(infer, probe)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        metrics = {}
        if condition == "inference_only":
            with torch.inference_mode():
                metrics.update(summarize([inference(infer, probe) for _ in range(80)]))
        elif condition == "training_only":
            metrics.update(train_epoch(train, data, seed))
        else:
            ready, go, stop = threading.Event(), threading.Event(), threading.Event()
            intervals, errors = [], []
            def monitor():
                try:
                    with torch.inference_mode(), torch.cuda.stream(torch.cuda.Stream()):
                        inference(infer, probe)
                        ready.set()
                        go.wait()
                        while not stop.is_set():
                            a = time.perf_counter()
                            latency = inference(infer, probe)
                            intervals.append((a, time.perf_counter(), latency))
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    ready.set()
            thread = threading.Thread(target=monitor)
            thread.start()
            try:
                ready.wait()
                if errors: raise errors[0]
                start = time.perf_counter()
                go.set()
                metrics.update(train_epoch(train, data, seed))
                end = time.perf_counter()
            finally:
                stop.set()
                go.set()
                thread.join()
            if errors: raise errors[0]
            inside = [ms for a,b,ms in intervals if a >= start and b <= end]
            if not inside: raise RuntimeError("No inference completion during local training")
            metrics.update(summarize(inside))
        rows.append(dict(condition=condition, probe_voxels=len(probe[1]), **metrics, **memory()))
        del train, infer
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def worker(cfg, idx, pct, budget, path):
    dev = device_setup("cuda:0")
    cap = cfg["allocator_cap_gib"][str(budget)]
    cap_allocator(cap)
    role = f"client{idx}"
    manifest = OUT/"data"/role/"manifest.json"
    probe = EdgeBatches(manifest, role, 1, dev)[0]
    ck = startfile(cfg["seed"], pct)
    rows = []
    for batch in cfg["batch_sweep"]:
        data = EdgeBatches(manifest, role, batch, dev)
        for metrics in paired(ck, data, probe, cfg["seed"]):
            rows.append(dict(client=idx, pruning_pct=pct, budget_gib=budget, allocator_gib=cap,
                             batch=batch, start_sha=sha(ck), **metrics,
                             **budget_check(metrics, budget, cfg["os_reserve_gib"], cap)))
    atomic_json(path, rows)


def main():
    cfg = json.loads(CONFIG.read_text())
    dest = OUT/"benchmark"
    dest.mkdir(exist_ok=True)
    for budget in cfg["budget_gib"]:
        for pct in cfg["pruning_percent"]:
            for idx in (1, 2):
                path = dest/f"client{idx}_p{pct}_{budget}g.json"
                if path.exists(): continue
                proc = mp.get_context("spawn").Process(target=worker, args=(cfg, idx, pct, budget, path))
                proc.start()
                proc.join()
                if proc.exitcode: raise RuntimeError(f"A6000 benchmark failed: {path}")
                print(path, flush=True)


if __name__ == "__main__":
    main()
