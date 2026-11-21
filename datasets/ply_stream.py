"""Bounded PLY vertex reads. Never map or load the complete point table."""
from dataclasses import dataclass
import itertools
import numpy as np
from .stpls3d_loader import PLY_TYPES,RAW_TO_CLASS,CLASSES
from .voxelization import IGNORE


@dataclass
class Header:
    count: int
    offset: int
    fmt: str
    dtype: np.dtype


def header(path):
    fields=[]
    count=fmt=element=None
    with path.open("rb") as f:
        if f.readline(128).strip()!=b"ply":
            raise ValueError("Invalid PLY signature")
        while f.tell()<1024**2:
            line=f.readline(8192)
            if not line: break
            part=line.decode("ascii").split()
            if not part: continue
            if part[0]=="format":
                fmt=part[1]
            elif part[0]=="element":
                element=part[1]
                if count is None:
                    if element!="vertex": raise ValueError("Expected vertex as first PLY element")
                    count=int(part[2])
            elif part[0]=="property" and element=="vertex":
                if part[1] not in PLY_TYPES: raise ValueError("Unsupported vertex property")
                fields.append((part[2],PLY_TYPES[part[1]]))
            elif part[0]=="end_header":
                if fmt not in {"ascii","binary_big_endian","binary_little_endian"} or not count or count<0:
                    raise ValueError("Invalid PLY header")
                dtype=np.dtype(fields).newbyteorder(">" if fmt=="binary_big_endian" else "<")
                return Header(count,f.tell(),fmt,dtype)
    raise ValueError("No valid PLY header")


def chunks(path,h,size,phase):
    if size<1: raise ValueError("Chunk size must be positive")
    mark=-1
    with path.open("rb") as f:
        f.seek(h.offset)
        for start in range(0,h.count,size):
            n=min(size,h.count-start)
            if h.fmt=="ascii":
                rows=itertools.islice(f,n)
                data=np.loadtxt(rows,dtype=h.dtype,ndmin=1)
            else:
                data=np.fromfile(f,dtype=h.dtype,count=n)
            if len(data)!=n: raise ValueError("Truncated PLY vertices")
            progress=int(10*(start+n)/h.count)
            if progress!=mark:
                print(f"{phase}: {start+n:,}/{h.count:,} vertices",flush=True)
                mark=progress
            yield data


def values(points,field,mapping,ignore):
    xyz=np.column_stack([points[k] for k in "xyz"]).astype("float64",copy=False)
    raw=points[field]
    if not np.isfinite(xyz).all() or not np.isfinite(raw).all() or (raw.dtype.kind=="f" and not np.all(raw==np.floor(raw))):
        raise ValueError("Nonfinite coordinates or invalid semantic IDs")
    valid=(raw>=0)&(raw<len(mapping))
    ignored=np.isin(raw,ignore) if ignore else np.zeros(len(raw),dtype=bool)
    unknown=~(valid|ignored)
    if unknown.any(): raise ValueError(f"Unmapped raw IDs {np.unique(raw[unknown]).tolist()}")
    labels=np.full(len(raw),IGNORE,dtype="int8")
    labels[valid&~ignored]=np.asarray(mapping,dtype="int8")[raw[valid&~ignored].astype(int)]
    return xyz,labels


def inspect(path,h,size,field,schema,ignore):
    if not {"x","y","z",field}<=set(h.dtype.names):
        raise ValueError(f"Missing XYZ or semantic field {field}: {h.dtype.names}")
    rgb={"red","green","blue"}<=set(h.dtype.names)
    mapping=RAW_TO_CLASS if schema=="raw" else list(range(len(CLASSES)))
    low=np.full(3,np.inf)
    high=-low.copy()
    counts={}
    sample=[]
    offset=0
    for pts in chunks(path,h,size,"inspect"):
        xyz,y=values(pts,field,mapping,ignore)
        low=np.minimum(low,xyz.min(0))
        high=np.maximum(high,xyz.max(0))
        ids,freq=np.unique(pts[field],return_counts=True)
        for i,n in zip(ids,freq): counts[str(int(i))]=counts.get(str(int(i)),0)+int(n)
        if rgb:
            for name in ("red","green","blue"):
                c=pts[name]
                if not np.isfinite(c).all() or c.min()<0 or c.max()>255: raise ValueError("Invalid RGB")
        stride=max(1,(h.count+1999)//2000)
        ix=np.arange((-offset)%stride,len(pts),stride)
        colors=np.column_stack([pts[k][ix] for k in ("red","green","blue")]) if rgb else np.zeros((len(ix),3))
        sample.extend(np.column_stack([xyz[ix],colors,y[ix]]).tolist())
        offset+=len(pts)
    return dict(count=h.count,format=h.fmt,properties=h.dtype.descr,raw_counts=counts,
                origin=low.tolist(),maximum=high.tolist(),rgb=rgb,mapping=mapping,preview=sample)
