#!/usr/bin/env python3
"""Install pinned public HLSL translation tools into Q0's ignored local directory."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import urllib.request

DEST = Path(__file__).resolve().parent / ".local"
BOTTLES = {
    "glslang": (
        "16.5.0",
        "b14c788c5c9c3fd1b115a1ace94e0c88394830906b303415f4099ed6ee34dec4",
    ),
    "spirv-cross": (
        "1.4.357.0",
        "3a02d41b640b88adb966762ee8a489e4afed26c6140f58fcdbbe4068273cf991",
    ),
    "spirv-tools": (
        "1.4.357.0",
        "a00b484190a546a8b5284da248deebb92ab6fac0688f3f3ba1b334fe8f127f74",
    ),
}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for name, (version, sha) in BOTTLES.items():
        target = DEST / name / version
        if target.is_dir():
            print(name + ": already present")
            continue
        token_url = (
            "https://ghcr.io/token?service=ghcr.io&scope=repository:homebrew/core/"
            + name
            + ":pull"
        )
        token = json.load(urllib.request.urlopen(token_url, timeout=30))["token"]
        url = "https://ghcr.io/v2/homebrew/core/" + name + "/blobs/sha256:" + sha
        data = urllib.request.urlopen(
            urllib.request.Request(url, headers={"Authorization": "Bearer " + token}),
            timeout=120,
        ).read()
        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError("shader tool checksum mismatch: " + name)
        archive = DEST / (name + ".tar.gz")
        archive.write_bytes(data)
        with tarfile.open(archive) as tar:
            for m in tar.getmembers():
                p = PurePosixPath(m.name)
                if (
                    p.is_absolute()
                    or ".." in p.parts
                    or not p.parts
                    or p.parts[0] != name
                ):
                    raise ValueError("unsafe tool archive path")
                if m.islnk() or m.issym():
                    link = (DEST / p.parent / m.linkname).resolve()
                    if not link.is_relative_to(DEST.resolve()):
                        raise ValueError("unsafe tool archive link")
                if not (m.isfile() or m.isdir() or m.issym() or m.islnk()):
                    raise ValueError("unsupported archive member")
            tar.extractall(DEST)
        archive.unlink()
        print(name + ": " + version + " verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
