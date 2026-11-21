"""Byte-level budget for project data on a quota-limited filesystem."""
import os
from pathlib import Path
import shutil
import subprocess


def budget(path, quota_free=None):
    path = Path(path)
    while not path.exists():
        path = path.parent
    free = shutil.disk_usage(path).free
    if quota_free is None:
        fs = subprocess.check_output(["df","--output=source",str(path)],text=True).splitlines()[-1].strip()
        result = subprocess.run(["quota","-w"],capture_output=True,text=True)
        for line in result.stdout.splitlines():
            parts = line.split()
            if parts and parts[0] == fs:
                used,soft,hard = (int(v.rstrip("*")) for v in parts[1:4])
                limits = [v for v in (soft,hard) if v]
                if limits:
                    quota_free = max(0,min(limits)-used)*1024
                else:
                    quota_free = free
                break
        if quota_free is None:
            raise ValueError("Cannot verify quota on this mount. Supply a freshly measured --quota-free-bytes.")
    available = min(free,quota_free)
    reserve = max(5_000_000_000,(available+4)//5)
    return dict(filesystem_free=free,quota_free=quota_free,available=available,reserve=reserve,
                writable=max(0,available-reserve))


def require_space(plan, needed):
    if needed > plan["writable"]:
        raise ValueError(f"Need {needed:,} additional bytes. Budget after reserve is {plan['writable']:,}.")
