from collections import defaultdict
from copy import deepcopy

import numpy as np
from torch.utils.data import Subset

from training.train import train
from training.evaluate import evaluate
from .fedavg import fedavg


def partition(rows, seed):
    """Reserve central training patches per region; divide the remaining local pool twice."""
    rng = np.random.default_rng(seed)
    by_region = defaultdict(list)
    for i,r in enumerate(rows):
        by_region[r["region"]].append(i)
    if len(by_region) < 2:
        raise ValueError("Need two spatial regions for Non-IID")
    central, local = [], {}
    for region, indices in sorted(by_region.items()):
        shuffled = rng.permutation(indices).tolist()
        cut = min(len(indices)-1, (len(indices)+1)//2)
        central.extend(shuffled[:cut])
        local[region] = shuffled[cut:]
    if not central:
        raise ValueError("Need more training patches to reserve central and local pools")
    non = [[],[]]
    for region in sorted(local, key=lambda r:(-len(local[r]), r)):
        non[int(len(non[1]) < len(non[0]))].extend(local[region])
    strata = defaultdict(list)
    for region, indices in local.items():
        for i in indices:
            strata[region, int(np.argmax(rows[i]["hist"]))].append(i)
    iid = [[],[]]
    for indices in strata.values():
        rng.shuffle(indices)
        start = int(len(iid[1]) < len(iid[0]))
        iid[start].extend(indices[::2])
        iid[1-start].extend(indices[1::2])
    return central, {"IID":iid,"Non-IID":non}


def simulate(model, dataset, pair, val, device, cfg, epoch_seed):
    sizes = list(map(len,pair))
    payload = sum(v.numel()*v.element_size() for v in model.state_dict().values())/1e6
    rounds = []
    for r in range(1,cfg["rounds"]+1):
        states, logs = [], []
        for i, indices in enumerate(pair):
            client = deepcopy(model).to(device)
            logs.append(train(client, Subset(dataset,indices),device,cfg,cfg["local_epochs"],
                              cfg["local_lr"],epoch_seed+2*r+i))
            states.append(client.cpu().state_dict())
            del client
        model.load_state_dict(fedavg(states,sizes),strict=True)
        scores = evaluate(model.to(device),val,device,cfg["batch"])
        model.cpu()
        rounds.append(dict(round=r, **scores, samples_a=sizes[0], samples_b=sizes[1],
                           train_a_s=sum(l["seconds"] for l in logs[0]), train_b_s=sum(l["seconds"] for l in logs[1]),
                           local_loss_a=logs[0][-1]["loss"], local_loss_b=logs[1][-1]["loss"],
                           uplink_mb=2*payload, downlink_mb=2*payload, cumulative_mb=4*r*payload))
        if cfg.get("verify_cuda"):
            rounds[-1].update(peak_training_mb=max(e["peak_training_mb"] for log in logs for e in log),
                              allocated_mb=max(e["allocated_mb"] for log in logs for e in log),
                              device=logs[0][-1]["device"])
    return rounds
