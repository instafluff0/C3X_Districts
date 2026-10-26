#!/usr/bin/env python3
"""Day/night look study on the production renderer, kept in the Lab.

Matches C3X's clock-driven lighting to the user's four-panel Civ VI reference
(noon, 18:00, midnight, 06:00). Renders the autumn study's views (test.biq
concept, Lab village) at those hours through the autumn harness.

`build` snapshots the working tree into ignored Renderer/native/build/daynight-src,
overlays a candidate tree (the proposed production files) and gives that private
DLL one Lab-only switch: C3X_LAB_DAY_NIGHT_KEYS names a key file replacing the
curve's constants, so looks are tuned without rebuilding. Production files,
packs and staged binaries are never touched.

    python3 Renderer/lab/studies/daynight/study.py build --overlay DIR
    python3 Renderer/lab/studies/daynight/study.py build      # baseline: working tree + game transfer
    python3 Renderer/lab/studies/daynight/study.py render current \
        --dll Renderer/native/build/daynight-baseline/Renderer/native/build/candidate/C3XRenderer.dll
    python3 Renderer/lab/studies/daynight/study.py render v1 --keys v1.txt

Key file lines (any subset): `night r g b`, `winter light saturation`, `direct INDEX hour r g b moon_share`,
`sky INDEX hour r g b saturation exposure`.
The private DLL's Lab output also uses the game's display transfer (resident_scene:
Reinhard mixed half with the filmic fit) instead of the Lab's plain Reinhard, so
brightness is judged as the game shows it.
Outputs are disposable and stay under Renderer/lab/out/daynight/.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from Renderer.lab.platform import ROOT
from Renderer.lab.studies.autumn import study as harness

OUT = ROOT / "Renderer/lab/out/daynight"
PRIVATE = ROOT / "Renderer/native/build/daynight-src"
PRIVATE_DLL = PRIVATE / "Renderer/native/build/candidate/C3XRenderer.dll"
# The working tree as it is, with only the game's display transfer: a fair "current".
BASELINE = ROOT / "Renderer/native/build/daynight-baseline"
HOURS = (12, 18, 0, 6)
for _hour in range(24):
    harness.VIEWS[f"concept-{_hour:02d}"] = ((20, 75), 128, _hour, (1600, 900))
    harness.VIEWS[f"village-{_hour:02d}"] = ((18, 74), 160, _hour, (1600, 900), "village", "no-units")

# Lab-only: the curve's constants become a file-loaded table in the private DLL.
RUNTIME = "Renderer/native/environment_runtime.cpp"
LOADER = '''
void load_lab_keys() {
    static bool loaded = false;
    if (loaded)
        return;
    loaded = true;
    char const * path = std::getenv("C3X_LAB_DAY_NIGHT_KEYS");
    std::FILE * file = path && *path ? std::fopen(path, "r") : nullptr;
    if (!file)
        return;
    char name[32] = {};
    int index = 0;
    while (std::fscanf(file, "%31s", name) == 1) {
        std::string kind = name;
        if (kind == "night")
            std::fscanf(file, "%f %f %f", &night_direct[0], &night_direct[1], &night_direct[2]);
        else if (kind == "winter")
            std::fscanf(file, "%f %f", &winter_night_light, &winter_night_saturation);
        else if (kind == "direct" && std::fscanf(file, "%d", &index) == 1 && index >= 0 && index < 5) {
            float * key = direct_keys[index];
            std::fscanf(file, "%f %f %f %f %f", &key[0], &key[1], &key[2], &key[3], &key[4]);
        } else if (kind == "sky" && std::fscanf(file, "%d", &index) == 1 && index >= 0 && index < 6) {
            float * key = sky_keys[index];
            std::fscanf(file, "%f %f %f %f %f %f", &key[0], &key[1], &key[2], &key[3], &key[4], &key[5]);
        }
    }
    std::fclose(file);
}
'''
TRANSFER = ("Renderer/native/render_core/linear_target.h",
            " rgb/=1+max(rgb.r,max(rgb.g,rgb.b));\n",
            " float3 filmic=min(rgb*.65,64);filmic=saturate((filmic*(2.51*filmic+.03))/(filmic*(2.43*filmic+.59)+.14));\n"
            " rgb=lerp(rgb/(1+max(rgb.r,max(rgb.g,rgb.b))),filmic,.5);\n")
# Lab-only: the preview tool sends a frame minute from C3X_LAB_PREVIEW_MINUTE, so
# films can step in quarter hours (the renderer already lights hour + minute/60).
PREVIEW_MINUTE = ("Renderer/native/biq_preview.cpp",
                  "    frame.hour = argc == 12 ? std::atoi(argv[11]) : 12;\n",
                  "    frame.hour = argc == 12 ? std::atoi(argv[11]) : 12;\n"
                  "    char lab_minute[8] = {};\n"
                  "    if (GetEnvironmentVariableA(\"C3X_LAB_PREVIEW_MINUTE\", lab_minute, sizeof(lab_minute)))\n"
                  "        frame.minute = std::atoi(lab_minute);\n")
# Lab-only: Renderer64 draws the Winter mountain flurry as per-view quads in its
# x64 pass, which Lab previews never run. With C3X_LAB_MOUNTAIN_FLURRY set, both
# Lab object paths attach the same quad (kind 6, 1.05 by 3.6 tiles on the tile's
# front corner, same seed) as a city effect; the season root's shader draws kind 6
# only in Winter. After studies/winter/flurry_tree.py.
FLURRY_SWITCH = 'GetEnvironmentVariableA("C3X_LAB_MOUNTAIN_FLURRY",nullptr,0)!=0'
FLURRY = {
    "Renderer/native/city_fidelity/compiler.h": (
        ("            e.kind==float(effect_geyser) || e.kind==float(effect_spray);\n",
         "            e.kind==float(effect_geyser) || e.kind==float(effect_spray) || e.kind==6.f;\n"),
        ("    append_effects(library,{plume},std::make_shared<Lighting>(),project_natural,indexed,output);\n"
         "    return output.chunks.size()>before;\n}\n",
         "    append_effects(library,{plume},std::make_shared<Lighting>(),project_natural,indexed,output);\n"
         "    return output.chunks.size()>before;\n}\n"
         "template<class Project>\n"
         "bool mountain_flurry(Library const& library,int nc,int nr,float ground,\n"
         "        Project project_natural,bool indexed,Surfaces& output){\n"
         "    EffectAnchor flurry{{float(nc)+1.f,float(nr),ground/112},\n"
         "        Effect{{0,0,0},6.f,1.05f,3.6f,float((unsigned(nc)*73u^unsigned(nr)*151u)%64u),1.f}};\n"
         "    auto before=output.chunks.size();\n"
         "    append_effects(library,{flurry},std::make_shared<Lighting>(),project_natural,indexed,output);\n"
         "    return output.chunks.size()>before;\n}\n"),
    ),
    "Renderer/native/object_preparation.h": (
        ("    if(!capture.road_mask && !capture.railroad_mask && capture.city_id<0 && !volcano && !wonder && !wonder_tile &&\n",
         "    bool const flurry=capture.real_terrain_type==6 && input.city_ready && " + FLURRY_SWITCH + ";\n"
         "    if(!capture.road_mask && !capture.railroad_mask && capture.city_id<0 && !volcano && !wonder && !wonder_tile && !flurry &&\n"),
        ("    if(stop())return {};\n    result->rivers.assign(",
         "    if(flurry && library.effect_material<library.materials.size()){\n"
         "        city_fidelity::Surfaces snow;\n"
         "        GroundProjection projection{nc,nr,input.projection.half_w,input.projection.half_h,\n"
         "            input.projection.relief_projection_scale,float(input.projection.content_view_height)};\n"
         "        if(city_fidelity::mountain_flurry(library,nc,nr,height_natural(float(nc)+1.f,float(nr)),projection,true,snow) &&\n"
         "           !adopt_city(snow))return {};\n"
         "    }\n"
         "    if(stop())return {};\n    result->rivers.assign("),
    ),
    "Renderer/native/source_fidelity/geometry.h": (
        ("            for(auto& chunk:plume.chunks)city_chunks.push_back(std::move(chunk));\n    }\n",
         "            for(auto& chunk:plume.chunks)city_chunks.push_back(std::move(chunk));\n    }\n"
         "    if(!prepared_objects && city_profile && cities.ready && owner.real==6 && " + FLURRY_SWITCH + "){\n"
         "        c3x_renderer::city_fidelity::Surfaces snow;\n"
         "        if(c3x_renderer::city_fidelity::mountain_flurry(cities.library,nc,nr,height_natural(float(nc)+1.f,float(nr)),\n"
         "                project_natural,false,snow))\n"
         "            for(auto& chunk:snow.chunks)city_chunks.push_back(std::move(chunk));\n"
         "    }\n"),
    ),
}
PATCHES = (
    ('#include "environment_runtime.h"\n', '#define _CRT_SECURE_NO_WARNINGS\n#include "environment_runtime.h"\n'),
    ("#include <limits>\n", "#include <limits>\n#include <cstdio>\n#include <cstdlib>\n#include <string>\n"),
    ("constexpr float night_direct[3]", "float night_direct[3]"),
    ("constexpr float winter_night_light", "float winter_night_light"),
    ("constexpr float winter_night_saturation", "float winter_night_saturation"),
    ("constexpr float direct_keys[][5]", "float direct_keys[][5]"),
    ("constexpr float sky_keys[][6]", "float sky_keys[][6]"),
    ("// The values at an hour, eased", LOADER + "\n// The values at an hour, eased"),
    ("    float hour = wrap_hour(requested_hour);\n    float sunrise",
     "    load_lab_keys();\n    float hour = wrap_hour(requested_hour);\n    float sunrise"),
)


def build_private(overlay: Path | None, head: bool = False) -> Path:
    from Renderer.lab.platform import native_command_result
    private = PRIVATE if overlay else BASELINE
    sources = ["Renderer/native", "Renderer/sandbox", "Renderer/lab/shared", "Renderer/lab/studies/borders",
               "Renderer/lab/native_preview.cpp", "Renderer/lab/build_native_preview.bat",
               "Renderer/lab/volcano_witness.h", "Renderer/default.custom_rendering.txt"]
    files = subprocess.run(["git", "ls-files", "-c", "-o", "--exclude-standard", *sources],
                           cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
    if head:
        # Tracked sources as committed: other sessions' uncommitted edits stay out.
        private.mkdir(parents=True, exist_ok=True)
        archive = subprocess.run(["git", "archive", "HEAD", *sources], cwd=ROOT, capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", str(private)], input=archive, check=True)
        tracked = set(subprocess.run(["git", "ls-files", "-c", *sources], cwd=ROOT, capture_output=True,
                                     text=True, check=True).stdout.splitlines())
        files = [name for name in files if name not in tracked]
    for name in files:
        if name.startswith("Renderer/native/build/"):
            continue
        target = private / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    for source in overlay.rglob("*") if overlay else ():
        if source.is_file():
            target = private / source.relative_to(overlay)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    if overlay:
        text = (private / RUNTIME).read_text()
        for old, new in PATCHES:
            if text.count(old) != 1:
                raise RuntimeError("Lab key-loader anchor changed: " + old)
            text = text.replace(old, new)
        (private / RUNTIME).write_text(text)
    path, old, new = TRANSFER
    text = (private / path).read_text()
    if text.count(old) != 1:
        raise RuntimeError("Lab transfer anchor changed in " + path)
    (private / path).write_text(text.replace(old, new))
    patch_preview(private)
    patch_flurry(private)
    harness.wait_for_vm()
    result = native_command_result(private.relative_to(ROOT).as_posix() + "/Renderer/native",
                                   "call BUILD.bat candidate-compile", timeout_seconds=1500)
    dll = private / "Renderer/native/build/candidate/C3XRenderer.dll"
    if result["status"] != "pass" or not dll.is_file():
        raise RuntimeError("Private day/night build failed:\n" + result.get("output_tail", ""))
    build_preview(private, refresh=False)
    return dll


def patch_preview(private: Path) -> None:
    path, old, new = PREVIEW_MINUTE
    text = (private / path).read_text()
    if text.count(old) != 1:
        raise RuntimeError("Lab preview-minute anchor changed in " + path)
    (private / path).write_text(text.replace(old, new))


def patch_flurry(private: Path) -> None:
    for path, edits in FLURRY.items():
        text = (private / path).read_text()
        for old, new in edits:
            if text.count(old) != 1:
                raise RuntimeError("Lab flurry anchor changed in " + path)
            text = text.replace(old, new)
        (private / path).write_text(text)


def build_preview(private: Path, refresh: bool = True) -> Path:
    """The preview must speak the tree's frame ABI, which can be newer than the shared one."""
    from Renderer.lab.platform import native_command_result
    if refresh:
        for name in ("Renderer/native/biq_preview.cpp", "Renderer/lab/native_preview.cpp"):
            shutil.copy2(ROOT / name, private / name)
        patch_preview(private)
        harness.wait_for_vm()
    result = native_command_result(private.relative_to(ROOT).as_posix() + "/Renderer/lab",
                                   "call build_native_preview.bat", timeout_seconds=600)
    preview = private / "Renderer/lab/.cache/native_preview.exe"
    if result["status"] != "pass" or not preview.is_file():
        raise RuntimeError("Private native preview build failed:\n" + result.get("output_tail", ""))
    return preview


def render(label: str, dll: Path, keys: Path | None, views, season: int = 0, shader_root: Path | None = None,
           minute: int = 0, seconds: float | None = None, flurry: bool = False) -> None:
    harness.OUT = OUT
    # Each tree's DLL renders through the preview built from the same tree.
    harness.PREVIEW = dll.resolve().parents[4] / "Renderer/lab/.cache/native_preview.exe"
    extra = {"C3X_LAB_DAY_NIGHT_KEYS": harness.windows(keys.resolve())} if keys else {}
    if shader_root:
        extra["C3X_RENDERER_SHADER_SOURCE_ROOT"] = harness.windows(shader_root.resolve())
    if minute:
        extra["C3X_LAB_PREVIEW_MINUTE"] = str(minute)
    if seconds is not None:
        # The animation clock: water, waves and effects move with it (still
        # renders otherwise all sit at second 1, so a film's sea froze).
        extra["C3X_LAB_EFFECT_TIME"] = f"{seconds:.6f}"
    if flurry:
        extra["C3X_LAB_MOUNTAIN_FLURRY"] = "1"
    harness.render(label, None, season, views, dll.resolve(), extra)


# Civ VI targets measured on the user's four-panel reference (OKLab lightness
# relative to noon, hue in degrees, chroma), and the concept view's patches.
TARGETS = {12: {"grass": (1.00, 107, .091), "tundra": (1.00, 303, .022), "panel": (1.00, 109, .064)},
           18: {"grass": (.94, 74, .079), "tundra": (.88, 326, .048), "panel": (.91, 72, .057)},
           0: {"grass": (.74, 200, .027), "tundra": (.85, 274, .114), "panel": (.78, 266, .056)},
           6: {"grass": (.63, 110, .033), "tundra": (.73, 285, .044), "panel": (.77, 357, .038)}}
PATCHES_CONCEPT = {"grass": (300, 400, 570, 650), "tundra": (300, 450, 790, 870), "panel": (0, 1600, 0, 900)}


def oklab(rgb):
    import numpy as np
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    lms = np.cbrt(linear @ np.array([[.4122214708, .5363325363, .0514459929], [.2119034982, .6806995451, .1073969566],
                                     [.0883024619, .2817188376, .6299787005]]).T)
    return lms @ np.array([[.2104542553, .7936177850, -.0040720468], [1.9779984951, -2.4285922050, .4505937099],
                           [.0259040371, .7827717662, -.8086757660]]).T


def load_rgb(path: Path):
    import numpy as np
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(900, 1600, 3)


def measure(labels) -> None:
    """Concept-view patches per hour against the Civ VI targets."""
    import numpy as np
    for label in labels:
        noon = {}
        for hour in HOURS:
            image = load_rgb(OUT / label / f"concept-{hour:02d}/preview.png") / 255.0
            cells = []
            for name, (x0, x1, y0, y1) in PATCHES_CONCEPT.items():
                with np.errstate(all="ignore"):  # spurious matmul flags from macOS Accelerate
                    lab = oklab(image[y0:y1, x0:x1].reshape(-1, 3))
                lightness, chroma = lab[:, 0].mean(), np.hypot(lab[:, 1], lab[:, 2]).mean()
                hue = np.degrees(np.arctan2(lab[:, 2].mean(), lab[:, 1].mean())) % 360
                noon.setdefault(name, lightness)
                target = TARGETS[hour][name]
                cells.append(f"{name} L x{lightness / noon[name]:.2f} h{hue:3.0f} C{chroma:.3f} "
                             f"(Civ VI x{target[0]:.2f} h{target[1]} C{target[2]:.3f})")
            print(f"{label} {hour:02d}  " + " | ".join(cells))


# Clock-bar colours: Summer gold, Autumn amber, Winter ice, Spring green.
SEASON_COLOURS = ((235, 200, 120), (225, 135, 60), (175, 205, 240), (140, 205, 110))


def timelapse(labels, view: str, seasons=None, seconds_per_hour: float = .75, fps: int = 30,
              target: Path | None = None, minutes=(0,)) -> Path:
    """Crossfade the 24 hourly renders of each label into one looping film.

    Several labels play one day each, in order, switching at midnight (one
    season per day); the clock bar takes the season's colour and four marks
    show which season is playing."""
    import numpy as np
    seasons = list(seasons) if seasons else [0] * len(labels)
    # Minute m of a label's hours lives in label-mMM (render --minute m).
    paths = [(day, hour + minute / 60, OUT / (label if not minute else f"{label}-m{minute:02d}") /
              f"{view}-{hour:02d}/preview.png")
             for day, label in enumerate(labels) for hour in range(24) for minute in minutes]
    target = target or OUT / labels[0] / f"timelapse-{view}.mp4"
    encoder = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                "-s", "1600x900", "-r", str(fps), "-i", "-", "-vf", "scale=1280:-2",
                                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(target)],
                               stdin=subprocess.PIPE)
    steps = max(1, round(seconds_per_hour * fps / len(minutes)))
    current = load_rgb(paths[0][2]).astype(np.float32)
    for index, (day, hour, _) in enumerate(paths):
        following = load_rgb(paths[(index + 1) % len(paths)][2]).astype(np.float32)
        colour = np.array(SEASON_COLOURS[seasons[day] % 4], np.float32)
        for step in range(steps):
            t = step / steps
            # Hourly films linger a little on each render; finer steps glide.
            image = current + (following - current) * (t * t * (3 - 2 * t) if len(minutes) == 1 else t)
            image[-10:, :] = image[-10:, :] * .35
            marker = int((hour + t / len(minutes)) / 24 * 1600)
            image[-10:, :marker] = image[-10:, :marker] * .4 + colour * .6
            image[-14:, max(0, marker - 2):marker + 2] = 255
            if len(labels) > 1:
                for mark in range(4):
                    lit = np.array(SEASON_COLOURS[mark], np.float32) if mark == seasons[day] % 4 else 60
                    image[-34:-22, 12 + mark * 22:28 + mark * 22] = lit
            encoder.stdin.write(image.astype(np.uint8).tobytes())
        current = following
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("timelapse encoding failed")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--overlay", type=Path, help="tree of proposed production files; omit for the baseline")
    build.add_argument("--preview-only", action="store_true", help="rebuild only the private preview tool")
    build.add_argument("--head", action="store_true", help="tracked sources from HEAD, not the shared working tree")
    run = sub.add_parser("render")
    run.add_argument("label")
    run.add_argument("--dll", type=Path, default=PRIVATE_DLL)
    run.add_argument("--keys", type=Path, help="key file for the private DLL")
    run.add_argument("--season", type=int, default=0, choices=range(4), help="0 Summer, 1 Autumn, 2 Winter, 3 Spring")
    run.add_argument("--minute", type=int, default=0, help="minutes past each view's hour (private preview)")
    run.add_argument("--time", type=float, help="animation clock second (water, waves, effects)")
    run.add_argument("--flurry", action="store_true", help="Winter mountain flurry (private build, season root)")
    run.add_argument("--shader-root", type=Path,
                     help="private shader root; seasons need `tools/season_shading.py --lab-root DIR`")
    run.add_argument("--views", nargs="*", default=[f"concept-{h:02d}" for h in HOURS],
                     choices=[name for name in harness.VIEWS if name.startswith(("concept-", "village-"))])
    check = sub.add_parser("measure", help="concept patches against the Civ VI targets")
    check.add_argument("labels", nargs="+")
    lapse = sub.add_parser("timelapse", help="MP4 from the 24 hourly renders of one view per label")
    lapse.add_argument("labels", nargs="+", help="one day each, in order (e.g. one per season)")
    lapse.add_argument("--view", default="concept", choices=("concept", "village"))
    lapse.add_argument("--seasons", nargs="*", type=int, help="season of each label, for the clock colours")
    lapse.add_argument("--out", type=Path, help="output MP4 (default beside the first label)")
    lapse.add_argument("--minutes", nargs="*", type=int, default=[0], help="minutes rendered per hour, e.g. 0 30")
    args = parser.parse_args()
    if args.command == "measure":
        measure(args.labels)
    elif args.command == "timelapse":
        print(timelapse(args.labels, args.view, args.seasons, target=args.out, minutes=args.minutes))
    elif args.command == "build":
        print(build_preview(PRIVATE) if args.preview_only else
              build_private(args.overlay.resolve() if args.overlay else None, args.head))
    else:
        render(args.label, args.dll, args.keys, args.views, args.season, args.shader_root, args.minute, args.time, args.flurry)


if __name__ == "__main__":
    main()
