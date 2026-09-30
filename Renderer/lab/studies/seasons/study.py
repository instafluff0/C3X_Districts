#!/usr/bin/env python3
"""Mac-only seasonal material feasibility study; never builds/stages game code.

Reads existing local packs and installed source metadata. Produces three small
PNG sheets and one receipt, with no copied packs or persistent DDS extraction.
These CPU material/asset views are deliberately not production-renderer parity.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "Renderer/lab/out/seasons"
SEASONS = ("Summer", "Fall", "Winter", "Spring")
TERRAINS = ("grassland", "plains", "desert")
SOURCE_CANDIDATES = (
    "Base/Platforms/Windows/BLPs/SHARED_DATA/"
    "TEXTURE_DiffuseTint_Foliage_Bld_Flowered_Color_B_null",
    "Base/Platforms/Windows/BLPs/SHARED_DATA/"
    "TEXTURE_DiffuseTint_Foliage_Bld_Flowered_White_B_null",
    "DLC/Expansion2/Platforms/Windows/BLPs/SHARED_DATA/TEXTURE_FX_Blossoms",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def inventory(assets_root):
    """Record direct evidence, using only repository/Assets-relative paths."""
    from Renderer.tools.asset_compiler.c3x_asset_compiler import parse_civbig_header
    pairs = []
    pins = {}
    for pack_name in ("VegetationNormalized", "Civ5EnvironmentVegetation"):
        pack = ROOT / "Renderer/packs" / pack_name
        manifest = load(pack / "manifest.json")
        for suffix in ("pine_01", "pine_02", "pine_03", "pine_clump_01", "pine_clump_02"):
            entries = [manifest["assets"][f"feature/{group}/{suffix}"]
                       for group in ("forest", "forest_snow")]
            meshes = [load(pack / entry["mesh"]) for entry in entries]
            materials = [load(pack / entry["material"]) for entry in entries]
            paths = set()
            snow_paths = set()
            for index, (entry, material) in enumerate(zip(entries, materials)):
                paths.update((pack / entry["mesh"], pack / entry["material"]))
                textures = {pack / material[c]["texture"] for c in ("base_color", "gloss", "opacity") if c in material}
                textures.update(pack / material["lean_normal"][k] for k in ("texture_0", "texture_1"))
                paths.update(textures)
                if index == 1:
                    snow_paths = textures
            pins.update({p.relative_to(ROOT).as_posix(): digest(p) for p in paths})
            pairs.append({"pack": pack_name, "body": suffix,
                          **{k+"_equal": len(meshes[0]["vertices"]) == len(meshes[1]["vertices"])
                             and all(a[k] == b[k] for a, b in zip(meshes[0]["vertices"], meshes[1]["vertices"]))
                             for k in ("position", "normal", "uv0")},
                          "topology_equal": meshes[0]["topology"] == meshes[1]["topology"],
                          "snow_material_bytes": sum(p.stat().st_size for p in snow_paths)})
    # The selected production adapter consumes this mixed forest, regardless of
    # which terrain pack is selected. Trace the real source instead of inferring
    # the current forest from the Base Civ VI ArtDef.
    selected_path = ROOT / "Renderer/packs/BeautyStudies/beauty_objects.bin"
    data = selected_path.read_bytes(); pos = 8
    pins[selected_path.relative_to(ROOT).as_posix()] = digest(selected_path)
    def unpack(fmt):
        nonlocal pos
        value = struct.unpack_from("<"+fmt, data, pos); pos += struct.calcsize("<"+fmt)
        return value
    def string():
        nonlocal pos
        n, = unpack("I"); value = data[pos:pos+n].decode(); pos += n
        return value
    version, nm, no, nr = unpack("4I")
    if version != 3:
        raise ValueError("Unsupported selected forest source layout")
    materials = [([string() for _ in range(7)], unpack("2I")) for _ in range(nm)]
    bodies = []
    for _ in range(no):
        name = string(); kind, mat, n = unpack("3I"); pos += n*32
        bodies.append((name, kind, mat))
    recipes = [unpack("IffIIIIff") for _ in range(nr)]
    if pos != len(data):
        raise ValueError("Unexpected selected forest source trailing bytes")
    selected = [{"body": name, "weight": sum(r[3] for r in recipes if r[0] == i),
                 "base_color": materials[mat][0][0]}
                for i, (name, kind, mat) in enumerate(bodies) if kind == 1]
    artdefs = []
    relevant_names = re.compile(r"flower|blossom|leafy|autumn|winter|spring|snow", re.I)
    for subtree in ("Base", "DLC"):
        for path in sorted((assets_root / subtree).rglob("*.artdef")):
            if path.name not in ("Clutter.artdef", "Features.artdef", "TerrainStyle.artdef", "TerrainMaterials.artdef"):
                continue
            tree = ET.parse(path)
            names = sorted({e.get("text") for e in tree.iter()
                            if e.tag in ("m_Name", "m_ElementName", "m_EntryName")
                            and e.get("text") and relevant_names.search(e.get("text"))})
            if names:
                artdefs.append({"path": path.relative_to(assets_root).as_posix(), "entries": names})
    candidates = []
    for name in SOURCE_CANDIDATES:
        path = assets_root / name
        if path.exists():
            candidates.append({"path": name, "sha256": digest(path),
                               **parse_civbig_header(path.read_bytes()),
                               "binding": "texture payload confirmed; meadow use not established"})
    return {"schema": "c3x.lab.seasons.evidence.v1", "snow_pairs": pairs, "selected_forest": selected,
            "source_artdefs": artdefs, "spring_candidates": candidates,
            "read_only_pack_inputs": pins}


def dds_image(data):
    from PIL import Image
    # Pillow decodes these UNORM blocks but rejects their sRGB enum aliases.
    # Only the in-memory format tag changes; payload and transfer stay intact.
    raw = bytearray(data)
    fmt = struct.unpack_from("<I", raw, 128)[0]
    struct.pack_into("<I", raw, 128, {72: 71, 78: 77, 99: 98}.get(fmt, fmt))
    with Image.open(io.BytesIO(raw)) as image:
        return image.convert("RGBA")


def to_linear(rgb):
    import numpy as np
    return np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)


def to_srgb(rgb):
    import numpy as np
    rgb = np.clip(rgb, 0, 1)
    return np.where(rgb <= .0031308, rgb * 12.92, 1.055 * rgb ** (1 / 2.4) - .055)


def smooth(a, b, x):
    import numpy as np
    t = np.clip((x-a)/(b-a), 0, 1)
    return t*t*(3-2*t)


def weather_field(x, y):
    """Continuous world-space study field; 32-tile periodicity is diagnostic."""
    import numpy as np
    x, y = np.remainder(x, 32), np.remainder(y, 32)
    p = 2*np.pi/32
    return (.50 + .19*np.sin(p*(x*5+y*3)) + .16*np.cos(p*(x*7-y*4))
            + .10*np.sin(p*(x*17+y*11)))


def seasonal_ground(base, snow, x, y, terrain, season):
    """Source-independent albedo recipe before lighting; source UVs unchanged."""
    import numpy as np
    if season == "Summer":
        return base.copy(), np.zeros(x.shape)
    luma = base @ np.array([.2126, .7152, .0722])
    if season == "Fall":
        strength = {"grassland": .32, "plains": .16, "desert": .035}[terrain]
        dry = luma[..., None]*np.array([1.23, .94, .65])
        return base*(1-strength)+dry*strength, np.zeros(x.shape)
    if season == "Winter":
        # Distinct exposed coverage retains biome identity even with white snow.
        threshold = {"grassland": .18, "plains": .29, "desert": .44}[terrain]
        mask = smooth(threshold-.11, threshold+.11, weather_field(x, y))*.95
        return base*(1-mask[..., None])+snow*mask[..., None], mask
    if season == "Spring":
        fresh = {"grassland": [1.01, 1.13, 1.04], "plains": [.98, 1.045, 1.01],
                 "desert": [1, 1, 1]}[terrain]
        return np.clip(base*np.array(fresh), 0, 1), np.zeros(x.shape)
    raise ValueError("Unknown season")


def flower_mask(x, y, terrain):
    """Static clumped wildflower marks in world space, no particles or clock."""
    import numpy as np
    p = 2*np.pi/32
    x, y = np.remainder(x, 32), np.remainder(y, 32)
    patch = smooth(.60, .78, weather_field(x+3, y-2))
    blooms = np.maximum(0, np.sin(x*p*515+np.sin(y*p*117))) * np.maximum(0, np.cos(y*p*494+x*p*36))
    density = {"grassland": 1., "plains": .4, "desert": 0.}[terrain]
    return smooth(.70, .91, blooms)*patch*density


def material_views(out, pins):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    pack = ROOT / "Renderer/packs/TerrainNormalized"
    size = 224
    yy, xx = np.mgrid[:size, :size]
    x, y = xx/size*4, yy/size*4
    textures = {}
    for name in (*TERRAINS, "snow"):
        path = pack / f"textures/{name}_base_color.dds"
        pins[path.relative_to(ROOT).as_posix()] = digest(path)
        image = dds_image(path.read_bytes()).resize((size, size), Image.Resampling.LANCZOS)
        textures[name] = to_linear(np.asarray(image)[..., :3]/255.)
    sheet = Image.new("RGB", (808, 1136), "#20272c")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=19)
    small = ImageFont.load_default(size=14)
    draw.text((18, 12), "SEASONAL GROUND / CURRENT SOURCE TEXTURES", font=font, fill="white")
    draw.text((18, 42), "CPU material study - fixed neutral light - not a production game render", font=small, fill="#b6c4ce")
    for col, terrain in enumerate(TERRAINS):
        draw.text((104+col*232, 74), terrain.upper(), font=small, fill="#e1d1ab")
    stats = {}
    for row, season in enumerate(SEASONS):
        draw.text((10, 148+row*248), season, font=small, fill="white")
        means = []
        for col, terrain in enumerate(TERRAINS):
            albedo, mask = seasonal_ground(textures[terrain], textures["snow"], x, y, terrain, season)
            if season == "Spring":
                flowers = flower_mask(x, y, terrain)
                white = to_linear(np.array([.99, .95, .80]))
                lavender = to_linear(np.array([.72, .49, .85]))
                color = np.where((np.sin((x*35+y*56)*2*np.pi/32) > 0)[..., None], white, lavender)
                albedo = albedo*(1-flowers[..., None]) + color*flowers[..., None]
            displayed = (np.clip(to_srgb(albedo)*255, 0, 255)+.5).astype(np.uint8)
            means.append(displayed.mean(axis=(0, 1)).tolist())
            sheet.paste(Image.fromarray(displayed), (96+col*232, 100+row*248))
            if season == "Winter":
                stats[terrain] = {"mean_snow_blend": round(float(mask.mean()), 3),
                                  "exposed_ground_fraction": round(float((mask < .5).mean()), 3)}
        if season == "Winter":
            stats["mean_rgb_distances"] = {f"{TERRAINS[a]}:{TERRAINS[b]}": round(float(np.linalg.norm(np.array(means[a])-means[b])), 2)
                                           for a, b in ((0, 1), (1, 2), (0, 2))}
    draw.text((18, 1105), "Spring marks are authored procedural wildflowers; source flower candidates are shown separately.", font=small, fill="#b6c4ce")
    sheet.save(out / "ground.png", optimize=True)
    # No seasonal color transform is applied to Summer, including source alpha.
    summer, _ = seasonal_ground(textures["grassland"], textures["snow"], x, y, "grassland", "Summer")
    assert np.array_equal(summer, textures["grassland"])
    assert np.allclose(weather_field(x, y), weather_field(x+32, y), atol=1e-12)
    assert np.allclose(flower_mask(x, y, "grassland"), flower_mask(x+32, y, "grassland"), atol=1e-11)
    # A scrolled/cropped view uses the same world inputs and must reproduce the
    # corresponding pixels; screen coordinates never seed snow or flowers.
    crop = np.s_[43:109, 51:170]
    winter, _ = seasonal_ground(textures["grassland"], textures["snow"], x, y, "grassland", "Winter")
    cropped, _ = seasonal_ground(textures["grassland"][crop], textures["snow"][crop], x[crop], y[crop], "grassland", "Winter")
    assert np.array_equal(winter[crop], cropped)
    return stats


def tree_view(pack_name, asset, season, stylized_pine, pins, material_asset=None):
    """Small vectorized source-mesh rasterizer; respects separate opacity maps."""
    import numpy as np
    from PIL import Image
    pack = ROOT / "Renderer/packs" / pack_name
    entry = load(pack / "manifest.json")["assets"][asset]
    material_entry = load(pack / "manifest.json")["assets"][material_asset] if material_asset else entry
    mesh = load(pack / entry["mesh"])
    material = load(pack / material_entry["material"])
    paths = [pack / entry["mesh"], pack / material_entry["material"]]
    paths.append(pack / material["base_color"]["texture"])
    if "opacity" in material:
        paths.append(pack / material["opacity"]["texture"])
    pins.update({p.relative_to(ROOT).as_posix(): digest(p) for p in paths})
    tex = np.asarray(dds_image(paths[2].read_bytes()))/255.
    opacity = np.asarray(dds_image(paths[3].read_bytes()))[..., 0]/255. if len(paths) > 3 else None
    points = np.array([v["position"] for v in mesh["vertices"]], dtype=float)
    normals = np.array([v["normal"] for v in mesh["vertices"]], dtype=float)
    uv = np.array([v["uv0"] for v in mesh["vertices"]], dtype=float)
    # Match the lab's half-height body experiment, not a calibrated game view.
    points[:, 2] *= .5
    normals[:, 2] *= 2
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    width, height = 224, 200
    rgb = np.full((height, width, 3), [.13, .16, .17])
    depth_buffer = np.full((height, width), -np.inf)
    scale = min(182/(np.ptp(points[:, 0])+np.ptp(points[:, 1])), 154/max(points[:, 2].max(), .01))
    light = np.array([-.45, -.60, 1.]); light /= np.linalg.norm(light)
    for tree, center in enumerate((112,)):
        p = points.copy(); p[:, 0] += tree*.25
        screen = np.column_stack((center+(p[:, 0]-p[:, 1])*scale, 165+(p[:, 0]+p[:, 1])*scale*.5-p[:, 2]*scale))
        depth = p[:, 0]+p[:, 1]+p[:, 2]*.65
        for tri in np.array(mesh["topology"]["indices"]).reshape(-1, 3):
            s = screen[tri]
            xmin, ymin = np.maximum(np.floor(s.min(axis=0)).astype(int), 0)
            xmax, ymax = np.minimum(np.ceil(s.max(axis=0)).astype(int), [width-1, height-1])
            if xmax < xmin or ymax < ymin:
                continue
            gy, gx = np.mgrid[ymin:ymax+1, xmin:xmax+1]
            a, b, c = s
            denominator = (b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
            if abs(denominator) < 1e-9:
                continue
            wa = ((b[1]-c[1])*(gx+.5-c[0])+(c[0]-b[0])*(gy+.5-c[1]))/denominator
            wb = ((c[1]-a[1])*(gx+.5-c[0])+(a[0]-c[0])*(gy+.5-c[1]))/denominator
            weights = np.stack((wa, wb, 1-wa-wb), axis=-1)
            z = weights @ depth[tri]
            texuv = np.clip(weights @ uv[tri], 0, .999999)
            ix = (texuv[..., 0]*tex.shape[1]).astype(int); iy = (texuv[..., 1]*tex.shape[0]).astype(int)
            sampled = tex[iy, ix]
            alpha = sampled[..., 3]
            if opacity is not None:
                ox = (texuv[..., 0]*opacity.shape[1]).astype(int); oy = (texuv[..., 1]*opacity.shape[0]).astype(int)
                alpha = np.minimum(alpha, opacity[oy, ox])
            admitted = (weights.min(axis=-1) >= -1e-9) & (z > depth_buffer[gy, gx]) & (alpha >= .5)
            normal = weights @ normals[tri]; normal /= np.maximum(1e-8, np.linalg.norm(normal, axis=-1)[..., None])
            albedo = to_linear(sampled[..., :3])
            green = smooth(.015, .11, sampled[..., 1]-sampled[..., 0])
            if season == "Fall" and ("leafy" in asset or stylized_pine):
                luma = albedo @ np.array([.2126, .7152, .0722])
                palette = np.array([2.30, .74, .08])
                tint = luma[..., None]*palette
                albedo = albedo*(1-green[..., None]*.88)+tint*green[..., None]*.88
            if season == "Spring":
                albedo *= 1+green[..., None]*np.array([.01, .10, .02])
            if season == "Winter" and "leafy" in asset:
                snow = green*smooth(.10, .72, normal[..., 2])*.94
                albedo = albedo*(1-snow[..., None])+to_linear(np.array([.91, .94, .98]))*snow[..., None]
            shaded = to_srgb(albedo*(.48+.62*np.abs(normal @ light))[..., None])
            rgb[gy[admitted], gx[admitted]] = shaded[admitted]
            depth_buffer[gy[admitted], gx[admitted]] = z[admitted]
    return Image.fromarray((np.clip(rgb, 0, 1)*255+.5).astype(np.uint8))


def tree_views(out, pins):
    from PIL import Image, ImageDraw, ImageFont
    sheet = Image.new("RGB", (1136, 832), "#20272c")
    draw = ImageDraw.Draw(sheet); font = ImageFont.load_default(size=18); small = ImageFont.load_default(size=14)
    draw.text((18, 14), "SEASONAL TREES / EXISTING MESHES", font=font, fill="white")
    draw.text((18, 42), "CPU asset study - original UVs and opacity - approximate fixed lighting", font=small, fill="#b6c4ce")
    for col, season in enumerate(SEASONS):
        draw.text((205+col*232, 78), season, font=font, fill="#e1d1ab")
    rows = (("Evergreen", "pine", False), ("Stylized pine", "pine", True), ("Existing leafy art", "leafy", False))
    for row, (label, kind, stylized) in enumerate(rows):
        draw.text((14, 163+row*228), label, font=small, fill="white")
        for col, season in enumerate(SEASONS):
            pack = "Civ5EnvironmentVegetation"
            asset = "feature/forest/leafy_v1_01" if kind == "leafy" else "feature/forest/pine_01"
            winter_material = "feature/forest_snow/pine_01" if kind == "pine" and season == "Winter" else None
            sheet.paste(tree_view(pack, asset, season, stylized, pins, winter_material), (192+col*232, 110+row*228))
    draw.text((18, 804), "Winter pines use upstream snow textures. Leafy winter is a slope-mask experiment; leaves remain attached.", font=small, fill="#b6c4ce")
    sheet.save(out / "trees.png", optimize=True)


def candidate_views(assets_root, out):
    from PIL import Image, ImageDraw, ImageFont
    from Renderer.tools.asset_compiler.c3x_asset_compiler import parse_civbig_header, make_dds_dx10_header, CIVBIG_HEADER_SIZE
    sheet = Image.new("RGB", (808, 348), "#20272c")
    draw = ImageDraw.Draw(sheet); font = ImageFont.load_default(size=18); small = ImageFont.load_default(size=14)
    draw.text((18, 12), "UPSTREAM SPRING CANDIDATES / TEXTURE EVIDENCE", font=font, fill="white")
    for col, (name, label) in enumerate(zip(SOURCE_CANDIDATES, ("Flowered foliage / color", "Flowered foliage / white", "Blossom sprite"))):
        path = assets_root / name
        if not path.exists():
            continue
        raw = path.read_bytes(); info = parse_civbig_header(raw)
        image = dds_image(make_dds_dx10_header(info)+raw[CIVBIG_HEADER_SIZE:CIVBIG_HEADER_SIZE+info["payload_bytes"]])
        image.thumbnail((224, 224))
        panel = Image.new("RGBA", (224, 224), "#59636a")
        panel.alpha_composite(image, ((224-image.width)//2, (224-image.height)//2))
        sheet.paste(panel.convert("RGB"), (32+col*256, 52))
        draw.text((32+col*256, 284), label, font=small, fill="white")
    draw.text((18, 322), "Payloads confirmed; model/particle binding and suitability as meadow flowers remain unproven.", font=small, fill="#b6c4ce")
    sheet.save(out / "spring-candidates.png", optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default = os.environ.get("C3X_CIV6_ASSETS")
    parser.add_argument("--assets-root", type=Path, default=Path(default) if default else Path.home()/
                        "Library/Application Support/Steam/steamapps/common/Sid Meier's Civilization VI/Civ6.app/Contents/Assets")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    if not args.inventory_only and any(importlib.util.find_spec(name) is None for name in ("PIL", "numpy")):
        configured = os.environ.get("C3X_RENDERER_PYTHON")
        candidates = ([Path(configured).expanduser()] if configured else []) + [
            Path.home()/".cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"]
        for python in candidates:
            if python.is_file() and python.resolve() != Path(sys.executable).resolve():
                probe = subprocess.run([str(python), "-c", "import PIL,numpy"], capture_output=True)
                if probe.returncode == 0:
                    os.execv(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]])
        parser.error("Use a Python with Pillow and NumPy, or set C3X_RENDERER_PYTHON; inventory-only needs neither")
    # Bound study output to the disposable Lab tree; never touch runtime packs.
    out = args.out.resolve()
    if not out.is_relative_to((ROOT / "Renderer/lab/out").resolve()):
        parser.error("Output must be beneath Renderer/lab/out")
    out.mkdir(parents=True, exist_ok=True)
    evidence = inventory(args.assets_root)
    assert all(p["position_equal"] and p["uv0_equal"] and p["topology_equal"] for p in evidence["snow_pairs"])
    if not args.inventory_only:
        pins = evidence["read_only_pack_inputs"]
        evidence["winter_material_metrics"] = material_views(out, pins)
        tree_views(out, pins)
        candidate_views(args.assets_root, out)
    assert all(digest(ROOT / name) == sha for name, sha in evidence["read_only_pack_inputs"].items())
    evidence["checks"] = ["five pine/snow pairs in each pack preserve positions, UVs and topology; selected-pack normals differ",
                          "all consumed local pack inputs unchanged"]
    if not args.inventory_only:
        evidence["checks"] += ["Summer material is exactly the decoded source albedo",
                               "snow and flower fields wrap at their diagnostic 32-tile period",
                               "cropped/scrolled winter material reproduces the same world pixels"]
    evidence["scope"] = "Mac CPU feasibility only; no D3D, VM, staging, reference changes or gameplay test"
    (out / "evidence.json").write_text(json.dumps(evidence, indent=2)+"\n")
    print(json.dumps({"checks": evidence["checks"], "winter": evidence.get("winter_material_metrics"),
                      "output_bytes": sum(p.stat().st_size for p in out.iterdir() if p.is_file())}, indent=2))


if __name__ == "__main__":
    main()
