"""Validate decoded voxels, classes, all spatial partitions and generate five figures."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets.compact_cache import CompactPatches,SIDE,RES,CLASSES,sha,spatial_clients


def stats(rs):
    h=np.array([r["hist"] for r in rs])
    n=len(rs)*SIDE**3
    return dict(patches=len(rs),voxels=h.sum(0).tolist(),patches_per_class=(h>0).sum(0).tolist(),
                absent=[c for c,f in zip(CLASSES,h.sum(0)) if not f],
                empty_fraction=1-sum(r["occupied"] for r in rs)/n,valid_fraction=float(h.sum()/n))


def audit(root,out):
    out.mkdir(parents=True,exist_ok=False)
    m=json.loads((root/"manifest.json").read_text())
    meta=json.loads((root/"plan.json").read_text())["meta"]
    rows=m["rows"]
    for path,expected in m["shards"].items():
        if sha(root/path)!=expected: raise ValueError(f"Shard SHA-256 mismatch: {path}")
    seen={tuple(r["tile"]):r["region"] for r in rows}
    if len(seen)!=len(rows): raise ValueError("Duplicate tile")
    for (x,y,z),zone in seen.items():
        for dx,dy,dz in np.ndindex(3,3,3):
            if seen.get((x+dx-1,y+dy-1,z+dz-1),zone)!=zone:
                raise ValueError("Adjacent patches cross a spatial boundary")
    tr=[r for r in rows if r["split"]=="train"]
    central,pairs=spatial_clients(tr)
    for pair in pairs.values():
        if sorted(central+pair[0]+pair[1])!=list(range(len(tr))):raise ValueError("Client assignment overlap")
    ds=CompactPatches(root,rows,"B" if meta["rgb"] else "A")
    samples=[]
    rng=np.random.default_rng(42)
    minval=np.full(5 if meta["rgb"] else 2,np.inf)
    maxval=-minval.copy()
    for i,row in enumerate(rows):
        x,y=ds[i]
        occupied=x[0].bool()
        hist=torch.bincount(y[y>=0],minlength=len(CLASSES)).tolist()
        if hist!=row["hist"] or int(occupied.sum())!=row["occupied"] or torch.any(y[~occupied]!=-100):
            raise ValueError(f"Invalid decoded patch {row['id']}")
        if not torch.all((x>=0)&(x<=1)): raise ValueError("Feature outside normalized range")
        minval=np.minimum(minval,x.flatten(1).amin(1).numpy())
        maxval=np.maximum(maxval,x.flatten(1).amax(1).numpy())
        idx=occupied.nonzero()
        zyx=idx[rng.choice(len(idx),min(30,len(idx)),replace=False)].numpy()
        labels=y[tuple(zyx.T)].numpy()
        samples.append(np.column_stack([(np.array(row["tile"])*SIDE+zyx[:,::-1]+.5)*RES+meta["origin"],labels]))
    splits={k:stats([r for r in rows if r["split"]==k]) for k in ("train","val","test")}
    splits["central"]=stats([tr[i] for i in central])
    for case,pair in pairs.items():
        for client,idx in zip(("A","B"),pair): splits[case+"_"+client]=stats([tr[i] for i in idx])
    result=dict(classes=CLASSES,all=stats(rows),splits=splits,range_min=minval.tolist(),range_max=maxval.tolist(),
                max_abs_quantization_error=np.max([r["quant_error"] for r in rows],axis=0).tolist(),
                cache_bytes=sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),
                adjacency_violations=0,duplicates=0,raw_ids=meta["raw_counts"],manifest_sha256=sha(root/"manifest.json"))
    (out/"stats.json").write_text(json.dumps(result,indent=2))
    parts=dict(central=[tr[i]["id"] for i in central],clients={c:[[tr[i]["id"] for i in v] for v in pair]
               for c,pair in pairs.items()},val=[r["id"] for r in rows if r["split"]=="val"],
               test=[r["id"] for r in rows if r["split"]=="test"])
    (out/"partitions.json").write_text(json.dumps(parts))
    origin=np.asarray(meta["origin"])
    raw=np.asarray(meta["preview"])
    sample=np.concatenate(samples)
    for title,coords,color in (("raw_points",raw[:,:3],raw[:,3:6]/255 if meta["rgb"] else raw[:,2]),
                               ("occupied_voxels",sample[:,:3],sample[:,2]),("semantic_voxels",sample[:,:3],sample[:,3])):
        fig=plt.figure(figsize=(8,5))
        ax=fig.add_subplot(projection="3d")
        if title=="semantic_voxels":
            obj=ax.scatter(*(coords-origin).T,c=color,s=1,cmap="tab10",vmin=-.5,vmax=9.5)
            fig.colorbar(obj,ax=ax,ticks=range(6)).ax.set_yticklabels(CLASSES)
        else: ax.scatter(*(coords-origin).T,c=color,s=1)
        ax.set(xlabel="X (m)",ylabel="Y (m)",zlabel="Z (m)",title=title.replace("_"," "))
        ax.set_box_aspect(np.maximum(np.ptp(coords,axis=0),1))
        fig.savefig(out/(title+".png"),dpi=140,bbox_inches="tight")
        plt.close(fig)
    fig,ax=plt.subplots()
    for region in sorted(set(seen.values())):
        tiles=np.array([r["tile"] for r in rows if r["region"]==region])
        ax.scatter(tiles[:,0],tiles[:,1],s=5,label=region)
    ax.set(xlabel="X patch column",ylabel="Y patch column",aspect="equal")
    ax.legend()
    fig.savefig(out/"spatial_splits.png",dpi=140)
    plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(8,6),sharex=True)
    for ax,case in zip(axes,("IID","Non-IID")):
        for i,c in enumerate(("A","B")):
            h=np.array(splits[case+"_"+c]["voxels"])
            ax.bar(np.arange(6)+i*.38,h/h.sum(),width=.38,label="Drone "+c)
        ax.set_ylabel(case+" fraction")
        ax.legend()
    axes[-1].set_xticks(np.arange(6)+.19,CLASSES,rotation=12)
    fig.tight_layout()
    fig.savefig(out/"client_distributions.png",dpi=140)
    plt.close(fig)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cache",type=Path)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    torch.set_num_threads(1)
    audit(args.cache,args.out)
