"""Content-addressed build/artifact storage with atomic writes and integrity checks."""

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Cache:
    def __init__(self, root):
        self.root = Path(root)
        self.events = []

    def artifact(self, kind, identity, build):
        key = digest(canonical(identity))
        directory = self.root / kind / key
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            record = directory / "record.json"
            target = directory / "payload"
            if record.exists():
                try:
                    info = json.loads(record.read_text())
                except (ValueError, OSError) as e:
                    raise ValueError(f"corrupt {kind} cache record: {key}") from e
                if (
                    info.get("identity") != identity
                    or not target.is_file()
                    or file_hash(target) != info.get("sha256")
                ):
                    raise ValueError(
                        f"corrupt {kind} cache: {key}; remove this entry explicitly to rebuild"
                    )
                for name,checksum in info.get("auxiliary",{}).items():
                    aux=directory/name
                    if not aux.is_file() or file_hash(aux)!=checksum:raise ValueError(f"corrupt {kind} auxiliary cache: {key}")
                self.events.append({"kind": kind, "key": key, "hit": True})
                return target
            if target.exists():
                raise ValueError(f"incomplete {kind} cache: {key}")
            temporary = directory / "pending"
            try:
                build(temporary)
                if not temporary.is_file():
                    raise ValueError(f"{kind} builder produced no artifact")
                if kind == "geometry":
                    from packet_store import compact_packet

                    compact_packet(temporary, self.root / "content")
                    os.replace(
                        Path(str(temporary) + ".blobs"), Path(str(target) + ".blobs")
                    )
                auxiliary={}
                source_meta=Path(str(temporary)+".source.json")
                if source_meta.exists():
                    final_meta=Path(str(target)+".source.json");os.replace(source_meta,final_meta)
                    auxiliary[final_meta.name]=file_hash(final_meta)
                checksum = file_hash(temporary)
                os.replace(temporary, target)
                temp_record = directory / "pending.json"
                temp_record.write_bytes(
                    canonical({"identity": identity, "sha256": checksum,"auxiliary":auxiliary})
                )
                os.replace(temp_record, record)
            finally:
                if temporary.exists():
                    temporary.unlink()
            self.events.append({"kind": kind, "key": key, "hit": False})
            return target
