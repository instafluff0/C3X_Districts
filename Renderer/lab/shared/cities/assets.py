"""Current normalized city input; no campaign, ownership or preview machinery."""
from functools import lru_cache
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[4]
PACK = Path("Renderer/packs/CityComponentsNormalized")
_inputs = ContextVar("city_asset_inputs", default=None)


@contextmanager
def track_inputs(*, generated=()):
    """Record bytes actually consumed by one build, excluding its own output."""
    records = {}
    token = _inputs.set((records, tuple(Path(p).resolve() for p in generated)))
    try:
        yield records
    finally:
        _inputs.reset(token)


def input_bytes(path):
    path = (ROOT / path).resolve()
    name = path.relative_to(ROOT).as_posix()
    data = path.read_bytes()
    tracking = _inputs.get()
    if tracking:
        records, generated = tracking
        if not any(path == directory or directory in path.parents for directory in generated):
            value = hashlib.sha256(data).hexdigest()
            if name in records and records[name] != value:
                raise ValueError("City input changed during build: " + name)
            records[name] = value
    return data


def read(path):
    return json.loads(input_bytes(path))


def rotate(point, angle):
    c, s = math.cos(angle), math.sin(angle)
    return [point[0]*c-point[1]*s, point[0]*s+point[1]*c, point[2]]


@lru_cache(None)
def component(asset, pack=PACK):
    manifest = read(pack / "manifest.json")
    landmark = read(pack / manifest["assets"][asset]["landmark"])
    parts = []
    for binding in landmark["draw_bindings"]:
        if "worked" not in binding["states"]:
            continue
        mesh = read(pack / landmark["components"]["geometry"][binding["geometry"]])
        material = read(pack / landmark["components"]["materials"][binding["material"]])
        for channel in material["channels"].values():
            if isinstance(channel, dict) and "texture" in channel:
                channel["texture"] = str(pack / channel["texture"])
        parts.append((mesh, material))
    points = [vertex["position"] for mesh, _ in parts for vertex in mesh["vertices"]]
    low = [min(point[i] for point in points) for i in range(3)]
    high = [max(point[i] for point in points) for i in range(3)]
    return dict(id=asset, parts=parts, lo=low, hi=high, sockets=landmark.get("attachment_points", []))


def source_scale(assets, factor=1.5):
    # Exactly the current production pool scale; placement is a separate solver.
    span = max(max(a["hi"][0]-a["lo"][0], a["hi"][1]-a["lo"][1]) for a in assets)
    height = statistics.median(a["hi"][2]-a["lo"][2] for a in assets)
    return max(.205/span, min(12/(80.9543*height), .32/span))*factor
