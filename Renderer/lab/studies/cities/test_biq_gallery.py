#!/usr/bin/env python3
"""Render a broad, isolated city gallery against captured test.biq terrain."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import time
from pathlib import Path

from PIL import Image

from Renderer.lab.platform import native_command_result
from Renderer.lab.studies.cities.build_layouts import ERAS, ROOT, STYLES


OUT = ROOT / "Renderer/lab/out/cities/test-biq/gallery"
PACK = ROOT / "Renderer/lab/out/cities/test-biq/root/Renderer/packs/CityCompositionRuntime/city.bin"
DLL = ROOT / "Renderer/native/build/city-preview/C3XRenderer.dll"
TERRAIN = ROOT / "Renderer/lab/out/cities/test-biq/terrain.csv"
WINDOWS_ROOT = r"..\lab\out\cities\test-biq\root"


def render(culture: int, era: int, size: int, site: tuple[int, int], name: str,
           glow_off: bool = False) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    trace = OUT / (name + ".trace.log")
    bitmap = OUT / (name + ".bmp")
    controls = {
        "C3X_RENDERER_VISUAL_PROFILE": "city-fidelity",
        "C3X_RENDERER_PREVIEW_OBJECTS": "1",
        "C3X_RENDERER_PREVIEW_CITY_ONLY": "1",
        "C3X_RENDERER_PREVIEW_CITY": f"{culture},{era},{size},1,1",
        "C3X_RENDERER_PREVIEW_CITY_SITE": f"{site[0]},{site[1]}",
        "C3X_RENDERER_CITY_GLOW_CONTROL": "1" if glow_off else "0",
        "C3X_RENDERER_PREVIEW_CUSTOM_DEFINITIONS":
            WINDOWS_ROOT + r"\Renderer\custom.custom_rendering.txt",
        "C3X_RENDERER_TRACE": "2",
        "C3X_RENDERER_TRACE_FILE":
            "..\\lab\\out\\cities\\test-biq\\gallery\\" + trace.name,
    }
    command = " && ".join(f'set "{key}={value}"' for key, value in controls.items())
    command += (r' && build\city-preview\biq_preview.exe '
                r'build\city-preview\C3XRenderer.dll '
                f'"{WINDOWS_ROOT}" '
                r'..\default.custom_rendering.txt '
                r'..\lab\out\cities\test-biq\terrain.csv '
                '"' + "..\\lab\\out\\cities\\test-biq\\gallery\\" + bitmap.name + '" '
                f'1280 800 {site[0]} {site[1]} 256 12')
    result = None
    for attempt in range(3):
        result = native_command_result("Renderer/native", command, timeout_seconds=180)
        if result["status"] == "pass" and "0 fallback" in result["output_tail"]:
            break
        time.sleep(3)
    authority = "lab-fixed-" + STYLES[culture].lower().replace(" ", "_") + "-" + ERAS[era].lower()
    trace_text = trace.read_text(errors="replace") if trace.exists() else ""
    marker = f"stage=city-composition city=1 authority={authority} "
    if (result is None or result["status"] != "pass" or
            "0 fallback" not in result["output_tail"] or marker not in trace_text):
        raise ValueError(f"city composition was not selected for {name}: " +
                         (result["output_tail"][-500:] if result else "no replay"))
    image = OUT / (name + ".png")
    with Image.open(bitmap) as rendered:
        rendered.save(image)
    print("PASS", name, authority, flush=True)
    return {"name": name, "culture": STYLES[culture], "era": ERAS[era],
            "size": size, "site": list(site), "image": image.name,
            "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            "authority": authority}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("flat", "hill"), default="flat")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if not PACK.is_file() or not DLL.is_file() or not TERRAIN.is_file():
        raise ValueError("compile the isolated Lab pack and preview DLL first")
    if args.kind == "flat":
        cases = [(culture, era, (culture+era) % 3,
                  f"flat-{STYLES[culture].lower().replace(' ', '_')}-{ERAS[era].lower()}")
                 for culture in range(5) for era in range(4)]
    else:
        cases = [(culture, (culture+1) % 4, 0,
                  f"hill-{STYLES[culture].lower().replace(' ', '_')}-{ERAS[(culture+1) % 4].lower()}")
                 for culture in range(5)]
    if args.limit:
        cases = cases[:args.limit]
    records = []
    for culture, era, size, name in cases:
        sites = ([(23, 79), (20, 64), (13, 87), (18, 64)] if era == 3 else
                 [(20, 64), (23, 79), (13, 87), (18, 64)]) if args.kind == "flat" else \
                [(18, 60), (18, 68), (15, 63)]
        failures = []
        for tier in dict.fromkeys((size, 0)):
            for site in sites:
                try:
                    records.append(render(culture, era, tier, site, name))
                    break
                except ValueError as error:
                    failures.append(f"{tier}@{site}: {str(error)[-120:]}")
            else:
                continue
            break
        else:
            raise ValueError(f"no legal site for {name}: " + "; ".join(failures))
        time.sleep(2)
    manifest = {"schema": "c3x.lab.test_biq_city_gallery.v1",
                "source": "test.biq terrain capture with synthetic city flags",
                "pack_sha256": hashlib.sha256(PACK.read_bytes()).hexdigest(),
                "dll_sha256": hashlib.sha256(DLL.read_bytes()).hexdigest(),
                "cases": records}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / (args.kind + ".json")).write_text(json.dumps(manifest, indent=2) + "\n")
    cards = "\n".join(
        '<article><img src="' + html.escape(case["image"]) + '" alt="' +
        html.escape(case["name"]) + '"><h2>' + html.escape(case["culture"] + " " +
        case["era"]) + '</h2><p>' + ("Town" if case["size"] == 0 else
        "City" if case["size"] == 1 else "Metropolis") +
        ' · capital · walled · ' + args.kind + ' terrain</p></article>'
        for case in records)
    document = ('<!doctype html><meta charset="utf-8"><title>City Lab test.biq gallery</title>'
                '<style>body{background:#211b29;color:#f4eaf5;font:16px system-ui;margin:24px}'
                'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:18px}'
                'article{background:#372b3e;padding:10px;border-radius:8px}'
                'img{width:100%;height:auto;display:block}h2{font-size:18px;margin:10px 0 2px}'
                'p{margin:0 0 5px;color:#d7bfd9}</style><h1>City Lab · test.biq terrain</h1>'
                '<p>Synthetic cities on captured terrain; headless D3D preview, pending in-game review.</p>'
                '<main>' + cards + '</main>')
    (OUT / (args.kind + ".html")).write_text(document)
    print(f"Wrote {len(records)} examples to {OUT / (args.kind + '.html')}")


if __name__ == "__main__":
    main()
