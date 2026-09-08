"""Production grassland detail on Metal; incomplete category migration, no approval."""
import json
import shutil
import time
from .cache import Cache, file_hash
from .compiler import compile_cpp, executable, run, shaders


def request(category, case, affected=False):
    if category != "grassland" or case != "detail" or affected:
        raise ValueError("Metal scene migration currently supports only lab grassland --case detail; "
                         "complete category/affected renders still use D3D11")


def reference(value):
    matches = [row for row in value["references"].get("d3d11", [])
               if (row["case"], row["hour"], row["zoom"]) == ("detail", 12, 128)]
    if len(matches) != 1:
        raise ValueError("Expected one approved grassland detail reference")
    return matches[0]


def recipe(value):
    expected = {"terrain": 2, "feature": 2, "objects": False, "hours": [12],
                "zooms": [128], "cases": ["detail", "gameplay"]}
    if value["recipe"] != expected:
        raise ValueError("Metal pilot requires the current flat grassland recipe at noon and zoom 128")
    reference(value)


def render(category, case):
    from Renderer import renderer as r
    request(category, case)
    r.require_prepared()
    started = time.perf_counter()
    identity = r.implementation_identity()
    value = r.standard(category)
    recipe(value)
    output = r.LAB / "out" / category / "metal"
    output.mkdir(parents=True, exist_ok=True)
    scene = output / "scene.csv"
    r.scene(category, case, scene)
    cache = Cache(r.LAB / ".cache/backend")
    objects = [compile_cpp(cache, source) for source in
               (r.LAB / "scene/grassland.cpp", r.ROOT / "Renderer/native/environment_runtime.cpp")]
    builder = cache.artifact("executable", {"objects": [file_hash(obj) for obj in objects]},
                            lambda path: run(["clang++", *objects, "-o", path]))
    builder.chmod(0o755)
    pack = r.ROOT / "Renderer/packs/NaturalFidelityRuntime"
    packet = cache.artifact("natural-scene", {
        "builder": file_hash(builder), "scene": file_hash(scene),
        "natural": {p.name: file_hash(p) for p in sorted(pack.iterdir()) if p.is_file()},
    }, lambda path: run([builder, r.ROOT, scene, path]))
    shader_dir = output / "shaders"
    shader_dir.mkdir(exist_ok=True)
    entries = {"VSMain": "VSNative", "VSFeature": "VSNative", "PSMain": "PSFeature", "PSFeature": "PSFeature"}
    for entry, path in shaders(cache, r.ROOT / "Renderer/native/city_fidelity/terrain.hlsl", -1,
                               entrypoints=entries).items():
        shutil.copyfile(path, shader_dir / (entry + ".msl"))
    metal = executable(cache, r.LAB / "backends/metal.mm", objc=True,
                       libraries=("-framework", "Metal", "-framework", "Foundation"))
    image = output / "detail-h12-z128.bmp"
    run([metal, packet, shader_dir, image, output / "cost.json", 4, 16])
    if r.implementation_identity() != identity:
        raise ValueError("Inputs changed during the Metal render; rerun")
    # Keep the pilot separate from complete D3D candidates and approval records.
    r.write(output / "render.json", {"backend": "metal", "scope": "grassland-detail-pilot",
        "implementation_identity": identity, "recipe": value["recipe"],
        "image": r.relative(image), "sha256": r.checksum(image),
        "wall_seconds": round(time.perf_counter() - started, 3)})
    return compare(category)


def compare(category):
    from Renderer import renderer as r
    from PIL import Image, ImageChops, ImageDraw, ImageStat
    request(category, "detail")
    r.require_prepared()
    value = r.standard(category)
    recipe(value)
    output = r.LAB / "out" / category / "metal"
    record = r.read(output / "render.json")
    if record.get("scope") != "grassland-detail-pilot" or record.get("backend") != "metal":
        raise ValueError("Not a Metal detail pilot")
    if record.get("implementation_identity") != r.implementation_identity() or record.get("recipe") != value["recipe"]:
        raise ValueError("Metal preview is stale; rerender grassland --backend metal --case detail")
    ref = reference(value)
    if r.checksum(r.local(ref["image"])) != ref["sha256"]:
        raise ValueError("Approved image was modified")
    if r.checksum(r.local(record["image"])) != record["sha256"]:
        raise ValueError("Metal candidate image changed after rendering")
    a = Image.open(r.local(ref["image"])).convert("RGB")
    b = Image.open(r.local(record["image"])).convert("RGB")
    if a.size != b.size:
        raise ValueError("Metal and approved image dimensions differ")
    delta = ImageChops.difference(a, b)
    red, green, blue = delta.split()
    maximum = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    metrics = {"scope": "grassland-detail-pilot", "approved_revision": value["approved_revision"],
        "maximum_channel_difference": max(high for low, high in delta.getextrema()),
        "mean_channel_difference": ImageStat.Stat(delta).mean,
        "identical_pixel_fraction": maximum.histogram()[0] / (a.width * a.height),
        "render_wall_seconds": record["wall_seconds"],
        "complete_category_parity": False, "approval_changed": False}
    sheet = Image.new("RGB", (a.width * 2, a.height + 24), "#222222")
    sheet.paste(a, (0, 24)); sheet.paste(b, (a.width, 24))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 4), "Approved D3D11 baseline", fill="white")
    draw.text((a.width + 8, 4), "Metal detail pilot (not approved)", fill="white")
    sheet.save(output / "compare.png")
    r.write(output / "compare.json", metrics)
    print(json.dumps(metrics, indent=2))
    print(r.relative(output / "compare.png"))
    return metrics
