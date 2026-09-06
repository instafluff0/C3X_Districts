"""Packet-v2 content references: deduplicate immutable mip and buffer payloads."""

import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
from cache import canonical, file_hash


def compact_packet(path, store):
    path = Path(path)
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    references = Path(str(path) + ".blobs")
    references.mkdir(exist_ok=True)
    target = path.with_suffix(".compact")
    with path.open("rb") as source, target.open("wb") as output:

        def read(n):
            data = source.read(n)
            if len(data) != n:
                raise ValueError("truncated packet during compaction")
            return data

        def u32(copy=True):
            data = read(4)
            if copy:
                output.write(data)
            return struct.unpack("<I", data)[0]

        def blob():
            size = u32(False)
            if size & 0x80000000:
                raise ValueError("packet already compacted")
            if size > 512 * 1024 * 1024:
                raise ValueError("invalid packet blob size")
            content = read(size)
            key = hashlib.sha256(content).hexdigest()
            final = store / key
            if final.exists():
                if file_hash(final) != key:
                    raise ValueError("corrupt content cache: " + key)
            else:
                fd, temp = tempfile.mkstemp(dir=store)
                try:
                    with os.fdopen(fd, "wb") as f:
                        f.write(content)
                    try:
                        os.link(temp, final)
                    except FileExistsError:
                        pass
                finally:
                    os.unlink(temp)
            ref = references / key
            if not ref.exists():
                os.link(final, ref)
            output.write(struct.pack("<I", size | 0x80000000))
            output.write(key.encode())

        if u32() != 0x32514C43:
            raise ValueError("invalid packet magic")
        version = u32(False)
        if version not in (1, 2, 3, 4, 5, 6):
            raise ValueError("unsupported packet version")
        output.write(struct.pack("<I", max(2, version)))
        for _ in range(3):
            u32()
        if version >= 3:
            output.write(read(28 if version>=4 else 24))
        textures = u32()
        if textures > 512:
            raise ValueError("invalid texture count")
        for _ in range(textures):
            for _ in range(3):
                u32()
            mips = u32()
            if mips > 15:
                raise ValueError("invalid mip count")
            for _ in range(mips):
                u32()
                blob()
        buffers = u32()
        if buffers > 256:
            raise ValueError("invalid buffer count")
        for _ in range(buffers):
            blob()
        output.write(source.read())
    os.replace(target, path)
    return {
        "resource_count": len(list(references.iterdir())),
        "packet_bytes": path.stat().st_size,
    }
