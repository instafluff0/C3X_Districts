"""Build an isolated witness and record source closure, flags and binary identity.

Uses the existing BENCHMARK_ZOOM build; never stages production or runs a game.
"""
import argparse
from pathlib import Path
import json
import platform
import os
import re
import shutil
import time

from Renderer.lab.platform import ROOT, native_command_result, windows_root
from Renderer.native.record_navigation_evidence import digest


def source_inputs():
    paths = (p for folder in ("Renderer/native", "Renderer/lab/shared")
             for p in (ROOT / folder).rglob("*") if p.is_file()
             and p.suffix in (".h", ".cpp", ".def", ".bat")
             and not any(part in ("build", "out", ".cache") for part in p.parts))
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}


DLL_UNITS=("c3x_renderer", "terrain_scene_runtime", "environment_runtime",
           "terrain_definition_runtime", "scene_export", "frame_scheduler")


def unit_inputs(unit, root=ROOT):
    """Conservative transitive quoted-include closure, including inactive branches.

    System headers are tied to the verified compiler/SDK stamp in the native batch.
    Missing/dynamic local includes fail closed instead of authorizing object reuse.
    """
    pending=[root / "Renderer/native" / (unit+".cpp")]
    found={}
    while pending:
        path=pending.pop().resolve()
        relative=path.relative_to(root).as_posix()
        if relative in found:continue
        found[relative]=digest(path)
        source=path.read_text()
        for token in re.findall(r'^\s*#\s*include\s+([^\n]+)',source,re.M):
            token=token.strip()
            if token.startswith("<"):continue
            match=re.match(r'"([^"]+)"',token)
            if not match:raise ValueError("Dynamic include requires explicit build dependency")
            included=path.parent / match[1]
            if not included.is_file():included=root/"Renderer/native"/match[1]
            if not included.is_file():raise ValueError("Unresolved local include: "+match[1])
            pending.append(included)
    return found


def reusable_units(previous, closures, tier, recipe):
    try:record=json.loads((previous/"build-evidence.json").read_text())
    except (OSError, ValueError):return []
    if record.get("returncode")!=0 or not record.get("sources_unchanged") or record.get("tier")!=tier or record.get("build_recipe")!=recipe:
        return []
    return [unit for unit,closure in closures.items() if record.get("unit_inputs",{}).get(unit)==closure
            and (previous/(unit+".obj")).is_file()
            and record.get("objects",{}).get(unit)==digest(previous/(unit+".obj"))]


def compiler_timing(output):
    marks=re.findall(r"BUILD_TIMING phase=(\w+) clock=\s*(\d+):(\d+):(\d+)\.(\d+)", output)
    if [m[0] for m in marks] != ["begin","compiler_begin","dll_done","preview_done"]:
        return {"status":"unmeasured"}
    times=[(int(h)*3600+int(m)*60+int(s)+float("0."+fraction))*1000 for _,h,m,s,fraction in marks]
    spans=[(b-a)%(24*3600*1000) for a,b in zip(times,times[1:])]
    return {"status":"measured", "resolution_ms":10, "compiler_environment_ms":spans[0],
            "dll_compile_link_ms":spans[1],"preview_compile_link_ms":spans[2]}


def main():
    started=time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="A new directory under Renderer/native/build")
    parser.add_argument("--tier", choices=("normal", "384", "768"), default="normal")
    parser.add_argument("--reuse-from", type=Path, help="Reuse verified unchanged translation-unit objects from an isolated prior build")
    parser.add_argument("--preview-only", action="store_true", help="Build the preview and copy the verified current production DLL without rebuilding or staging it")
    args = parser.parse_args()
    out = (ROOT / args.out).resolve()
    out.relative_to(ROOT / "Renderer/native/build")
    relative = str(out.relative_to(ROOT / "Renderer/native")).replace("/", "\\")
    if any(c in relative for c in '\"%\r\n'):
        raise ValueError("Unsupported build directory")
    out.mkdir(parents=True, exist_ok=False)
    before = source_inputs()
    verified=time.perf_counter()
    closures={unit:unit_inputs(unit) for unit in (*DLL_UNITS,"biq_preview")}
    recipe=digest(ROOT/"Renderer/native/BENCHMARK_ZOOM.bat")
    reused=[]
    if args.reuse_from:
        previous=args.reuse_from.resolve()
        previous.relative_to(ROOT/"Renderer/native/build")
        reused=reusable_units(previous,closures,args.tier,recipe)
        if not (previous/"toolchain.txt").is_file():reused=[]
        if reused:
            shutil.copy2(previous/"toolchain.txt",out/"previous-toolchain.txt")
            for unit in reused:shutil.copy2(previous/(unit+".obj"),out/(unit+".obj"))
    staged = ROOT / "Renderer/bin/C3XRenderer.dll"
    staged_hash = None
    if args.preview_only:
        from Renderer.renderer import require_current_candidate
        require_current_candidate()
        if args.tier != "normal" or digest(staged) != digest(ROOT / "Renderer/native/build/candidate/C3XRenderer.dll"):
            raise ValueError("Preview-only evidence requires the verified current normal-tier production DLL")
        staged_hash = digest(staged)
        shutil.copy2(staged, out / "C3XRenderer.dll")
    flags = {"C3X_ZOOM_OUT": relative, "C3X_ZOOM_LARGE_CACHE": "" if args.tier=="normal" else "1",
             "C3X_ZOOM_GPU_CACHE_MIB": "384" if args.tier=="normal" else args.tier,
             "C3X_ZOOM_BUILD_UNITS":" ".join(unit for unit in (*DLL_UNITS,"biq_preview") if unit not in reused) or "none"}
    command = " && ".join(f'set "{key}={value}"' for key,value in flags.items())
    mode = "preview-only" if args.preview_only else "build-only"
    command += " && call BENCHMARK_ZOOM.bat candidate " + mode
    dispatched=time.perf_counter()
    # Keep the VM transport short; long environment/command tails can fail
    # before dispatch. Preserve the exact build command in a local batch.
    batch_path=out/"build.bat"
    batch_path.write_bytes(("@echo off\r\n"+command+"\r\nexit /b %errorlevel%\r\n").encode())
    win_out=(ROOT if os.name=="nt" else windows_root()) / out.relative_to(ROOT).as_posix()
    result = native_command_result("Renderer/native",f'call "{win_out / "build.bat"}"')
    compiled=time.perf_counter()
    after = source_inputs()
    finished=time.perf_counter()
    record = {"timing_ms":{"source_verification_and_setup":(verified-started)*1000,
                            "copy_and_dispatch_preparation":(dispatched-verified)*1000,
                            "compile_link_and_vm_dispatch":(compiled-dispatched)*1000,
                            "source_verification_after":(finished-compiled)*1000,
                            "total_before_evidence":(finished-started)*1000},
              "native_compile_timing_ms":compiler_timing(result.get("output_tail", "")),
              "timing_coverage":{"vm_transport_only":"unmeasured; outer dispatch includes process startup/exit"},
              "unit_inputs":closures, "build_recipe":recipe,
              "requested_reused_units":reused,
              "compiled_units":re.findall(r"BUILD_UNIT unit=(\w+)",result.get("output_tail", "")),
              "objects":{unit:digest(out/(unit+".obj")) for unit in closures if (out/(unit+".obj")).is_file()},
              "sources": before, "sources_unchanged": before==after,
              "tier": args.tier, "environment": flags, "command": "BENCHMARK_ZOOM.bat candidate " + mode,
              "preview_only": args.preview_only, "copied_staged_dll_sha256": staged_hash,
              "flags": "MSVC x86 /std:c++17 /EHsc /O2 /W4 /WX /DC3X_RENDERER_BENCHMARK_ORACLE; preview /LARGEADDRESSAWARE",
              "host_os": platform.platform(), "returncode": result["returncode"],
              "output_tail": result.get("output_tail", ""),
              "binaries": {name:digest(out/name) for name in ("C3XRenderer.dll","biq_preview.exe") if (out/name).is_file()}}
    (out/"build-evidence.json").write_text(json.dumps(record,indent=2))
    if args.preview_only and (digest(staged) != staged_hash or digest(out / "C3XRenderer.dll") != staged_hash):
        raise ValueError("Production DLL changed during preview verification")
    print(json.dumps({"out":str(out),"returncode":record["returncode"],"sources_unchanged":record["sources_unchanged"]}))
    raise SystemExit(0 if result["returncode"]==0 and before==after else 1)


if __name__ == "__main__":
    main()
