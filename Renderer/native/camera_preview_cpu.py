"""Render the actual experimental CPU preview; not a full-renderer baseline.

Explicit generic pack plus BIQ terrain CSV input. Does not configure the game,
stage a DLL, replace references, or invoke the VM. Timings are local CPU only.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
TERRAINS = ("desert", "plains", "grassland", "tundra", "flood_plain", "hills",
            "mountains", "forest", "jungle", "marsh", "volcano", "coast", "sea", "ocean")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=2240)
    parser.add_argument("--height", type=int, default=1192)
    parser.add_argument("--tile-width", type=int, default=128)
    parser.add_argument("--x", type=int, default=35)
    parser.add_argument("--y", type=int, default=39)
    parser.add_argument("--hour", type=int, default=12)
    args = parser.parse_args()
    if not (0 < args.width <= 8192 and 0 < args.height <= 8192 and 1 < args.tile_width <= 1024):
        parser.error("invalid viewport or tile size")
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        parser.error("a C++17 compiler is required")
    manifest = json.loads((args.pack / "manifest.json").read_text())
    def pack_file(relative):
        path = (args.pack / relative).resolve()
        if not path.is_relative_to(args.pack.resolve()):
            raise ValueError("material or texture path escapes the selected pack")
        return path
    paths = []
    for name in TERRAINS:
        material = json.loads(pack_file(manifest["assets"][f"terrain/{name}/base"]["material"]).read_text())
        paths.append(pack_file(material["base_color"]["texture"]))
    rows = list(csv.reader(args.world.read_text().splitlines()))
    if rows[0][0] != "C3X_BIQ_TERRAIN_V3":
        parser.error("expected the authoritative V3 BIQ terrain fixture")
    map_width, map_height, expected = map(int, rows[0][1:])
    if expected != len(rows) - 1:
        parser.error("incomplete terrain fixture")
    tw, th = args.tile_width, args.tile_width // 2
    shift_x, shift_y = args.width // 2 - tw // 2 - args.x * tw // 2, args.height // 2 - th // 2 - args.y * th // 2
    tiles = []
    for row in rows[1:]:
        x, y, base, real = map(int, row[:4])
        for wrap in (-1, 0, 1):
            rx = x + wrap * map_width
            ax, ay = rx * tw // 2 + shift_x, y * th // 2 + shift_y
            if ax + tw < 0 or ax > args.width or ay + th < 0 or ay > args.height:
                continue
            tiles.append((rx, y, ax, ay, base, real))
    if not tiles or len(tiles) > 8192:
        parser.error("empty or oversized visible capture")
    source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
    body = "struct PublishedMapFrame {" + source.split("struct PublishedMapFrame {", 1)[1].split("// Civ III remains", 1)[0]
    program = r'''
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>
#include "Renderer/native/c3x_renderer_api.h"
#include "Renderer/native/environment_runtime.h"
''' + body + r'''
struct Texture {bool configured=true;std::vector<std::uint8_t> dds;};
int main(int argc,char** argv){
    if(argc!=16)return 2;
    std::array<Texture,14> textures;
    for(int i=0;i<14;++i){std::ifstream file(argv[i+1],std::ios::binary);
        textures[i].dds.assign(std::istreambuf_iterator<char>(file),{});}
    std::vector<c3x_renderer_tile_v1> tiles;
'''
    for tile in tiles:
        x, y, ax, ay, base, real = tile
        program += f"{{c3x_renderer_tile_v1 t={{}};t.tile_x={x};t.tile_y={y};t.anchor_x={ax};t.anchor_y={ay};t.terrain_type={base};t.real_terrain_type={real};t.tile_flags=C3X_RENDERER_TILE_RENDER;tiles.push_back(t);}}\n"
    program += f"""
    c3x_renderer_frame_v1 frame={{}};frame.target_width={args.width};frame.target_height={args.height};
    frame.tile_width={tw};frame.tile_height={th};frame.hour={args.hour};frame.season=0;
    frame.clip_right=frame.target_width;frame.clip_bottom=frame.target_height;
    frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());
""" + r'''
    CameraTerrainPreview preview;PublishedMapFrame result;std::atomic<bool> cancelled{false};
    auto start=std::chrono::steady_clock::now();
    if(!preview.render(frame,textures,result,cancelled))return 1;
    auto end=std::chrono::steady_clock::now();
    std::ofstream file(argv[15],std::ios::binary);
    file.write(reinterpret_cast<char const*>(result.pixels.data()),std::streamsize(result.pixels.size()*4));
    if(!file)return 3;
    std::printf("%.3f\n",std::chrono::duration<double,std::milli>(end-start).count());
}
'''
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="c3x-camera-preview-") as folder:
        folder = Path(folder)
        cpp, exe, raw = folder / "preview.cpp", folder / "preview", folder / "pixels.bgra"
        cpp.write_text(program)
        subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(ROOT), str(cpp),
                        str(ROOT / "Renderer/native/environment_runtime.cpp"), "-o", str(exe)], check=True)
        result = subprocess.run([str(exe), *map(str, paths), str(raw)], capture_output=True, text=True, check=True, timeout=30)
        from PIL import Image
        Image.frombytes("RGBA", (args.width, args.height), raw.read_bytes(), "raw", "BGRA").save(args.out)
    receipt = {"kind": "experimental CPU terrain preview, not production baseline",
               "viewport": [args.width, args.height], "tile_width": tw,
               "center": [args.x, args.y], "tiles": len(tiles), "local_cpu_ms": float(result.stdout),
               "source_sha256": hashlib.sha256(body.encode()).hexdigest(),
               "image_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest()}
    args.out.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
