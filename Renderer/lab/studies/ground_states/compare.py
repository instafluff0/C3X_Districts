"""Before/after page for study.py renders (local, self-contained)."""
import base64
import struct
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "Renderer/lab/out/ground_states"


def bmp_to_png(path: Path) -> bytes:
    data = path.read_bytes()
    offset, = struct.unpack_from("<I", data, 10)
    width, height = struct.unpack_from("<ii", data, 18)
    bpp, = struct.unpack_from("<H", data, 28)
    stride = ((width * bpp // 8) + 3) & ~3
    rows = []
    for y in range(abs(height)):
        row = data[offset + (y if height < 0 else abs(height) - 1 - y) * stride:][:width * bpp // 8]
        step = bpp // 8
        rgb = bytearray(width * 3)
        rgb[0::3], rgb[1::3], rgb[2::3] = row[2::step], row[1::step], row[0::step]
        rows.append(b"\0" + bytes(rgb))
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, abs(height), 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(b"".join(rows), 6)) + chunk(b"IEND", b""))


def main():
    views = sorted({p.parent.name.rsplit("_", 1)[0] for p in OUT.glob("z*_*/*.bmp")})
    parts = ["<!doctype html><html><head><meta charset=utf-8><title>Ground States Lab</title><style>",
             "body{margin:0;background:#1b1a18;color:#ecebe7;font:15px -apple-system,Helvetica,Arial}main{padding:16px}",
             ".row{display:flex;gap:10px;flex-wrap:wrap}figure{margin:0}img{max-width:100%;display:block}",
             "figcaption{color:#a8a59e;font-size:13px;margin:4px 0 14px}</style></head><body><main>",
             "<h1>Ground states and site sizes: production renderer, 1498 AD save</h1>",
             "<p>Before is the production site pack; after is the candidate pack. Eruption pollution beside the volcano, "
             "Osaka's real pollution, craters, ruins, a goody hut and a barbarian camp were added on chosen tiles.</p>"]
    for view in views:
        parts.append(f"<h2>{view}</h2><div class=row>")
        for label in ("before", "after"):
            images = sorted((OUT / f"{view}_{label}").glob("*.bmp"))
            images = [p for p in images if "removed" not in p.name]
            if images:
                uri = "data:image/png;base64," + base64.b64encode(bmp_to_png(images[0])).decode()
                parts.append(f"<figure><img src='{uri}'><figcaption>{label}</figcaption></figure>")
        parts.append("</div>")
    parts.append("</main></body></html>")
    (OUT / "index.html").write_text("".join(parts))
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
