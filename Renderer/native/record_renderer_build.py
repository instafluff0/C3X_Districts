"""Build an isolated witness and record source closure, flags and binary identity.

Uses the existing BENCHMARK_ZOOM build; never stages production or runs a game.
"""
import argparse
import json
import platform
import shutil

from Renderer.lab.platform import ROOT, native_command_result
from Renderer.native.record_navigation_evidence import digest


def source_inputs():
    paths = (p for folder in ("Renderer/native", "Renderer/lab/shared")
             for p in (ROOT / folder).rglob("*") if p.is_file()
             and p.suffix in (".h", ".cpp", ".def", ".bat")
             and not any(part in ("build", "out", ".cache") for part in p.parts))
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="A new directory under Renderer/native/build")
    parser.add_argument("--tier", choices=("normal", "384", "768"), default="normal")
    parser.add_argument("--preview-only", action="store_true", help="Build the preview and copy the verified current production DLL without rebuilding or staging it")
    args = parser.parse_args()
    out = (ROOT / args.out).resolve()
    out.relative_to(ROOT / "Renderer/native/build")
    relative = str(out.relative_to(ROOT / "Renderer/native")).replace("/", "\\")
    if any(c in relative for c in '\"%\r\n'):
        raise ValueError("Unsupported build directory")
    out.mkdir(parents=True, exist_ok=False)
    before = source_inputs()
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
             "C3X_ZOOM_GPU_CACHE_MIB": "384" if args.tier=="normal" else args.tier}
    command = " && ".join(f'set "{key}={value}"' for key,value in flags.items())
    mode = "preview-only" if args.preview_only else "build-only"
    command += " && call BENCHMARK_ZOOM.bat candidate " + mode
    result = native_command_result("Renderer/native",command)
    after = source_inputs()
    record = {"sources": before, "sources_unchanged": before==after,
              "tier": args.tier, "environment": flags, "command": "BENCHMARK_ZOOM.bat candidate " + mode,
              "preview_only": args.preview_only, "copied_staged_dll_sha256": staged_hash,
              "flags": "MSVC x86 /std:c++17 /EHsc /O2 /W4 /WX; preview /LARGEADDRESSAWARE",
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
