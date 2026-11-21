import json
import struct
import csv
import subprocess
import sys
from pathlib import Path
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from datasets.stpls3d_loader import prepare, VoxelPatches, assign_splits
from datasets.synthetic_smoke import SyntheticSmoke
from datasets.voxelization import IGNORE
from models.voxel_3dcnn import VoxelUNet
from pruning.structured_pruning import prune
from training.train import load, save, seed_all, train
from training.evaluate import evaluate, profile
from federated.fedavg import fedavg
from federated.server import partition, simulate

torch.set_num_threads(1)


def ply(path, mode, extra_label=None):
    points = []
    for i in range(12):
        # Six real raw IDs collapse to the official benchmark groups.
        points.extend([(i*8.,0,0,0,0,0,0),(i*8.+.2,0,0,255,0,0,5),
                       (i*8.+.3,0,0,255,0,0,8),(i*8.,1,1,0,255,0,14),
                       (i*8.,2,1,0,0,255,11),(i*8.,1,2,0,0,0,-100)])
    if extra_label is not None:
        points.append((0,0,0,0,0,0,extra_label))
    header = ["ply",f"format {mode} 1.0",f"element vertex {len(points)}",
              *["property double "+k for k in "xyz"],
              *["property uchar "+k for k in ("red","green","blue")],"property int class","end_header"]
    with path.open("wb") as f:
        f.write(("\n".join(header)+"\n").encode())
        for p in points:
            if mode == "ascii":
                f.write((" ".join(map(str,p))+"\n").encode())
            else:
                f.write(struct.pack(("<" if mode=="binary_little_endian" else ">")+"dddBBBi",*p))


@pytest.mark.parametrize("mode",["ascii","binary_little_endian","binary_big_endian"])
def test_ply_voxel_split(tmp_path,mode):
    path, out = tmp_path/"WMSC_points.ply",tmp_path/"cache"
    ply(path,mode)
    m = prepare([path],out,side=8,resolution=1,ignore_raw=[-100])
    rows = m["rows"]
    sets = {s:[r for r in rows if r["split"]==s] for s in ("train","val","test")}
    for a,b in (("train","val"),("train","test"),("test","val")):
        assert min(abs(p["tile"][0]-q["tile"][0]) for p in sets[a] for q in sets[b]) >= 2
    assert len({r["region"] for r in sets["train"]}) == 2
    item = VoxelPatches(out,rows,8,"B")
    x,y = item[0]
    assert x.shape == (5,8,8,8) and y.shape == (8,8,8)
    assert x[2,0,0,0] == pytest.approx(2/3)
    assert y[0,0,0] == 3 and y[1,1,0] == 5 and y[1,2,0] == 4
    assert x[1,1,1,0] == .125 and y[2,1,0] == IGNORE and x[0,2,1,0] == 1
    assert not x[0,7,7,7] and y[7,7,7] == IGNORE
    torch.testing.assert_close(VoxelPatches(out,rows,8,"A")[0][0],x[:2])
    assert json.loads((out/"manifest.json").read_text())["sources"][0]["raw_counts"]["8"] == 12


def test_unknown_label(tmp_path):
    path = tmp_path/"fixture.ply"
    ply(path,"ascii",37)
    with pytest.raises(ValueError, match="Unknown semantic IDs"):
        prepare([path],tmp_path/"cache",ignore_raw=[-100])


def test_multisite():
    rows = [dict(scene=s,tile=[i,0,0]) for s in ("A","B","C","WMSC_points") for i in range(4)]
    assign_splits(rows)
    assert {r["scene"] for r in rows if r["split"]=="test"} == {"WMSC_points"}
    assert {r["region"] for r in rows if r["split"]=="train"} == {"A","B"}
    for site in {r["scene"] for r in rows}:
        assert len({r["split"] for r in rows if r["scene"] == site}) == 1


@pytest.mark.parametrize("ratio",[.2,.4])
def test_skip_pruning_identical_to_gated_reference_and_reload(tmp_path,ratio):
    seed_all(42)
    x = torch.randn(2,5,9,8,7)
    base = VoxelUNet(5,6).eval()
    net,keep = prune(base,ratio,x)
    handles = []
    def gate(indices):
        def apply(module,args,out):
            mask = torch.zeros(out.shape[1])
            mask[indices] = 1
            return out*mask[None,:,None,None,None]
        return apply
    for name,keep_axis in keep.items():
        handles.append(getattr(base,name).register_forward_hook(gate(keep_axis)))
    torch.testing.assert_close(base(x),net(x))
    for h in handles:
        h.remove()
    a,b = [profile(model,x[:1],"cpu",2) for model in (base,net)]
    assert b["parameters"] < a["parameters"] and b["macs"] < a["macs"]
    net.train()
    loss = net(x).square().mean()
    loss.backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in net.parameters())
    net.eval()
    path = tmp_path/"small.pt"
    save(path,net,{"dataset":"fixture"})
    restored,meta = load(path)
    torch.testing.assert_close(net(x),restored(x))
    assert meta["dataset"] == "fixture"


def test_segmentation_metrics():
    class Fixed(nn.Identity):
        spec = dict(classes=2)
    x = torch.tensor([4.,-4.,4.,4.,-4.,4.,-4.,-4.]).reshape(1,2,1,1,4)
    y = torch.tensor([0,1,1,IGNORE]).reshape(1,1,1,4)
    scores = evaluate(Fixed(),TensorDataset(x,y),"cpu",1)
    assert scores["accuracy"] == pytest.approx(2/3)
    assert scores["miou"] == pytest.approx(.5)
    assert scores["per_class_iou"] == [.5,.5]
    assert scores["macro_f1"] == pytest.approx(2/3)
    assert scores["valid_voxels"] == 3


def test_fedavg_and_bn():
    a = {"param":torch.tensor([2.]),"bn.running_var":torch.tensor([1.]),"bn.num_batches_tracked":torch.tensor(3)}
    b = {"param":torch.tensor([6.]),"bn.running_var":torch.tensor([5.]),"bn.num_batches_tracked":torch.tensor(7)}
    state = fedavg([a,b],[3,1])
    assert state["param"].item() == 3 and state["bn.running_var"].item() == 2
    assert state["bn.num_batches_tracked"].item() == 7
    with pytest.raises(ValueError):
        fedavg([a,{**b,"param":torch.ones(2)}],[3,1])


def test_synthetic_fl_and_best_selection(tmp_path):
    seed_all(42)
    d = SyntheticSmoke("train",19,12,"B",42)
    val = SyntheticSmoke("val",4,12,"B",42)
    central, pairs = partition(d.rows,42)
    assert (central,pairs) == partition(d.rows,42)
    assert next(iter(DataLoader(d,batch_size=2)))[0].shape == (2,5,12,12,12)
    cfg = dict(batch=3,amp=False,local_lr=1e-4,local_epochs=1,rounds=2)
    net = VoxelUNet(5,2)
    best = tmp_path/"best.pt"
    logs = train(net,Subset(d,central),"cpu",cfg,4,.001,42,val,best,{})
    assert logs[-1]["loss"] < logs[0]["loss"]
    loaded,meta = load(best)
    assert meta["val_miou"] == max(e["val_miou"] for e in logs)
    assert evaluate(loaded,val,"cpu",3)["miou"] == pytest.approx(meta["val_miou"])
    for case,(a,b) in pairs.items():
        assert len(set(central+a+b)) == len(central+a+b) == len(d)
        if case=="Non-IID":
            assert {d.rows[i]["region"] for i in a}.isdisjoint(d.rows[j]["region"] for j in b)
        restored = deepcopy(loaded)
        history = simulate(restored,d,[a,b],val,"cpu",cfg,42)
        assert [r["round"] for r in history] == [1,2]
        assert history[1]["cumulative_mb"] == 2*history[0]["cumulative_mb"]
        assert not torch.equal(restored.head.weight,loaded.head.weight)

def test_prepared_ply_cli(tmp_path):
    # This is a parser/CLI fixture in pytest's temporary directory.
    # It does not contain any source STPLS3D points.
    root = Path(__file__).resolve().parents[1]
    src, cache = tmp_path/"fixture.ply", tmp_path/"cache"
    ply(src,"binary_little_endian")
    prepare([src],cache,side=8,resolution=1,ignore_raw=[-100])
    cfg = json.loads((root/"configs/default.json").read_text())
    cfg.update(pretrain_epochs=1,finetune_epochs=1,rounds=2,batch=2,latency_repeats=1)
    settings = tmp_path/"config.json"
    settings.write_text(json.dumps(cfg))
    out = tmp_path/"fixture-run"
    subprocess.run([sys.executable,"-m","experiments.run_experiments","stpls3d","--variant","B",
                    "--cpu-threads","1","--device","cpu","--out",str(out),"--data",str(cache),
                    "--config",str(settings)], cwd=root,check=True)
    with (out/"results/experiment_results.csv").open() as f:
        records = list(csv.DictReader(f))
    assert {r["id"] for r in records} == {f"E{i}" for i in range(1,7)}
    assert len(records)==11 and all(r["variant"]=="B" for r in records)
    assert len(list((out/"plots").glob("*.png"))) == 4
    for record in records:
        assert len(json.loads(record["per_class_iou"])) == 6
    with (out/"results/federated_round_metrics.csv").open() as f:
        assert len(list(csv.DictReader(f))) == 12
