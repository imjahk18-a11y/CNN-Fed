"""Run from the project root with python -m deployment.jetson."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def read(path):
    try:
        return Path(path).read_text().strip("\x00\n ")
    except OSError:
        return None


def query(args):
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=8)
        return proc.stdout.strip() if not proc.returncode else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def environment():
    model = read("/proc/device-tree/model")
    arch = platform.machine()
    info = dict(model=model, machine=arch, python=sys.version.split()[0], host=platform.node(),
                os=platform.freedesktop_os_release().get("PRETTY_NAME"), l4t=read("/etc/nv_tegra_release"),
                l4t_package=query(["dpkg-query","-W","-f=${Version}","nvidia-l4t-core"]),
                jetpack=query(["dpkg-query","-W","-f=${Version}","nvidia-jetpack"]),
                power_mode=query(["nvpmodel","-q"]), tegrastats=shutil.which("tegrastats"),
                disk_free_bytes=shutil.disk_usage(".").free,thermal_millic={},
                meminfo_kib={k:int(v.split()[0]) for line in (read("/proc/meminfo") or "").splitlines()
                             for k,v in [line.split(":",1)] if k in ("MemAvailable","MemTotal")})
    info["orin_nano"] = bool(arch == "aarch64" and model and "orin nano" in model.lower())
    for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        info["thermal_millic"][read(zone/"type") or zone.name]=read(zone/"temp")
    try:
        import torch
        info.update(torch=str(torch.__version__), cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                    cuda_test=torch.cuda.is_available() and torch.tensor(2, device="cuda:0").square().item()==4)
        if info["cuda_test"]:
            info["gpu"]=torch.cuda.get_device_name(0)
            info["total_gpu_bytes"]=torch.cuda.get_device_properties(0).total_memory
    except (ImportError, RuntimeError, OSError) as exc:
        info.update(cuda_test=False, torch_error=str(exc))
    return info


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="role", required=True)
    sub.add_parser("preflight").add_argument("--output",type=Path)
    for role in ("client","server"):
        a=sub.add_parser(role)
        a.add_argument("--port",type=int,default=6543)
        a.add_argument("--token-file",type=Path,required=True)
        a.add_argument("--work",required=True)
        a.add_argument("--batch",type=int,choices=(1,2,4),default=1)
        if role=="server":
            a.add_argument("--start",required=True)
            a.add_argument("--rounds",type=int,default=10)
            a.add_argument("--mode",choices=("INT8","FP32"),default="INT8")
            a.add_argument("--validation")
        else:
            a.add_argument("--id",type=int,choices=(1,2),required=True)
            a.add_argument("--host",default="127.0.0.1")
            a.add_argument("--data",required=True)
            a.add_argument("--allocator-gib",type=float,default=2)
    args=p.parse_args()
    if args.role=="preflight":
        info=json.dumps(environment(),indent=2)
        if args.output:
            with args.output.open("x") as file:
                file.write(info+"\n")
        print(info)
        return
    if len(args.token_file.read_bytes())!=16:
        p.error("Expected a shared 16-byte token file")
    shared=["--port",str(args.port),"--work",args.work,"--token-file",str(args.token_file),"--edge-batch",str(args.batch)]
    if args.role=="server":
        cli=["federated.fl_server","--start",args.start,"--device","cpu","--patience","0",
             "--mode",args.mode,"--rounds",str(args.rounds)]+shared
        if args.validation: cli+=["--validation",args.validation]
    else:
        info=environment()
        if not (info["orin_nano"] and info["cuda_test"]):
            p.error("Client deployment requires detected Jetson Orin Nano and a working CUDA torch build")
        cli=["federated.uav_client","--id",str(args.id),"--data",args.data,"--device","cuda:0","connect",
             "--host",args.host,"--allocator-gib",str(args.allocator_gib)]+shared
    os.execv(sys.executable,[sys.executable,"-m",*cli])


if __name__=="__main__":
    main()
