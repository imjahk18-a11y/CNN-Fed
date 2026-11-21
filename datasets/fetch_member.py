"""Read a public Drive ZIP central directory and stream just one semantic PLY."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
from urllib.parse import urlparse
import zipfile

import requests
from bs4 import BeautifulSoup

from .storage_budget import budget, require_space


class RangeZip(io.RawIOBase):
    def __init__(self, url, session, consent):
        self.session, self.url, self.pos, self.transferred = session,url,0,0
        self.etag = None
        res = self.request("bytes=0-0")
        if res.status_code==200 and "text/html" in res.headers.get("Content-Type",""):
            with res:
                page = BeautifulSoup(next(res.iter_content(32768)),"html.parser")
            form = page.select_one("form#download-form")
            if not consent or form is None or "can't scan" not in page.get_text():
                raise ValueError("Drive requires explicit user input")
            action = form.get("action")
            if urlparse(action).hostname != "drive.usercontent.google.com":
                raise ValueError("Unsupported confirmation endpoint")
            self.url = requests.Request("GET",action,params={e["name"]:e.get("value","") for e in
                                                          form.select('input[type=hidden][name]')}).prepare().url
            res = self.request("bytes=0-0")
        with res:
            m = re.fullmatch(r"bytes 0-0/(\d+)",res.headers.get("Content-Range",""))
            if res.status_code != 206 or not m:
                raise ValueError(f"Drive does not offer verified byte ranges (HTTP {res.status_code}); full download stopped")
            self.size=int(m.group(1))
            self.etag=res.headers.get("ETag")
        self.pending=None

    def request(self, span):
        headers={"Range":span,"Accept-Encoding":"identity"}
        if self.etag:
            headers["If-Range"]=self.etag
        return self.session.get(self.url,headers=headers,timeout=(10,90),stream=True)

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self,offset,whence=0):
        self.pos = [0,self.pos,self.size][whence]+offset
        if self.pos<0:
            raise OSError("Negative seek")
        return self.pos

    def read(self,size=-1):
        n=min(self.size-self.pos,size if size>=0 else self.size-self.pos)
        if n<=0:
            return b""
        if n>16*1024**2:
            raise ValueError("Unexpected ZIP metadata request over 16 MiB")
        start=self.pos
        if self.pending is not None and self.pending[0]==start:
            response,end = self.pending[1:]
            if start+n>end:
                raise ValueError("Read beyond selected member")
            data=response.raw.read(n)
            self.pending=(start+n,response,end)
        else:
            with self.request(f"bytes={start}-{start+n-1}") as r:
                self.check(r,start,start+n)
                data=r.content
        if len(data)!=n:
            raise IOError("Truncated HTTP range")
        self.pos+=n
        self.transferred+=n
        return data

    def check(self,r,start,end):
        if r.status_code!=206 or r.headers.get("Content-Range")!=f"bytes {start}-{end-1}/{self.size}":
            raise IOError(f"Range not honored (status {r.status_code}); response body discarded")

    def member_stream(self,length):
        end=self.pos+length
        r=self.request(f"bytes={self.pos}-{end-1}")
        try:
            self.check(r,self.pos,end)
        except Exception:
            r.close()
            raise
        self.pending=(self.pos,r,end)
        return r


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file_id")
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--report",type=Path,required=True)
    ap.add_argument("--consent-scan-warning",action="store_true")
    ap.add_argument("--quota-free-bytes",type=int)
    args=ap.parse_args()
    plan=budget(args.out,args.quota_free_bytes)
    with requests.Session() as session:
        src=RangeZip("https://drive.usercontent.google.com/download?export=download&id="+args.file_id,
                     session,args.consent_scan_warning)
        with zipfile.ZipFile(src) as z:
            files=[i for i in z.infolist() if i.filename.lower().endswith(".ply") and "__MACOSX" not in i.filename]
            candidates=[i for i in files if "realworlddata" in i.filename.lower().split("/")]
            synthetic=[i for i in files if "synthetic" in i.filename.lower() and "instance" not in i.filename.lower()]
            ordered=sorted(candidates,key=lambda i:(Path(i.filename).name.lower()!="wmsc_points.ply",i.file_size))
            ordered+=sorted(synthetic,key=lambda i:i.file_size)
            overhead=4_000_000_000+1_100_000_000+100_000_000
            info=dict(budget=plan,archive_bytes=src.size,cache_workspace_results_reservation=overhead,
                      members=[dict(name=i.filename,compressed=i.compress_size,bytes=i.file_size) for i in ordered])
            with args.report.open("x") as f:
                json.dump(info,f,indent=2)
            chosen=next((i for i in ordered if i.file_size+overhead<=plan["writable"]),None)
            if chosen is None:
                raise ValueError("No eligible individual PLY fits the budget")
            print(chosen.filename,chosen.compress_size,chosen.file_size,flush=True)
            require_space(plan,chosen.file_size+overhead)
            args.out.mkdir(exist_ok=True,parents=True)
            dest=args.out/Path(chosen.filename).name
            if dest.exists(): raise FileExistsError(dest)
            tmp=dest.with_suffix(".ply.partial")
            digest=hashlib.sha256()
            with z.open(chosen) as member, src.member_stream(chosen.compress_size), tmp.open("xb") as f:
                copied=0
                while block:=member.read(1024**2):
                    copied+=len(block)
                    if copied>chosen.file_size: raise IOError("Member exceeds declared length")
                    digest.update(block)
                    f.write(block)
                if copied!=chosen.file_size: raise IOError("Incomplete file")
            os.link(tmp,dest)
            tmp.unlink()
            info.update(selected=chosen.filename,sha256=digest.hexdigest(),download_bytes=src.transferred,
                        stored_bytes=copied,zip_crc32=f"{chosen.CRC:08x}")
            args.report.write_text(json.dumps(info,indent=2))
            print("Verified",dest,flush=True)


if __name__=="__main__":
    main()
