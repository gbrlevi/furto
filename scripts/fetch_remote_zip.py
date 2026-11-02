"""Extract chosen members of a remote zip with HTTP range requests.

Usage: fetch_remote_zip.py URL OUT_DIR SUBSTRING [SUBSTRING ...]
"""

import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


class RemoteFile(io.RawIOBase):
    def __init__(self, url: str) -> None:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        ) as r:
            self.url = url  # the redirect target of Dropbox is single use
            self.size = int(r.headers["Content-Range"].split("/")[-1])
        self.pos = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, buffer) -> int:  # type: ignore[no-untyped-def, override]  # buffer protocol
        end = min(self.pos + len(buffer), self.size)
        if end <= self.pos:
            return 0
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end - 1}"})
        with urllib.request.urlopen(request) as r:
            data: bytes = r.read()
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


def main() -> None:
    url, out, wanted = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
    out.mkdir(parents=True, exist_ok=True)
    remote = io.BufferedReader(RemoteFile(url), 1 << 25)  # type: ignore[arg-type]  # RawIOBase subclass
    with zipfile.ZipFile(remote) as z:
        names = [n for n in z.namelist() if not n.endswith("/") and any(w in n for w in wanted)]
        print(f"{len(names)} of {len(z.namelist())} members match", flush=True)
        for name in names:
            target = out / Path(name).name
            if target.exists() and target.stat().st_size == z.getinfo(name).file_size:
                continue
            with z.open(name) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            print(target.name, target.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
