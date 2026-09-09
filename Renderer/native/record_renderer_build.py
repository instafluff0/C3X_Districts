"""Build an isolated witness and record source closure, flags and binary identity.

Uses the existing BENCHMARK_ZOOM build; never stages production or runs a game.
"""
import argparse
import json
import platform

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
    args = parser.parse_args()
    out = (ROOT / args.out).resolve()
    out.relative_to(ROOT / "Renderer/native/build")
    relative = str(out.relative_to(ROOT / "Renderer/native")).replace("/", "\\")
    if any(c in relative for c in '\"%\r\n'):
        raise ValueError("Unsupported build directory")
    out.mkdir(parents=True, exist_ok=False)
    before = source_inputs()
    flags = {"C3X_ZOOM_OUT": relative, "C3X_ZOOM_LARGE_CACHE": "" if args.tier=="normal" else "1",
             "C3X_ZOOM_GPU_CACHE_MIB": "384" if args.tier=="normal" else args.tier}
    command = " && ".join(f'set "{key}={value}"' for key,value in flags.items())
    command += " && call BENCHMARK_ZOOM.bat candidate build-only"
    result = native_command_result("Renderer/native",command)
    after = source_inputs()
    record = {"sources": before, "sources_unchanged": before==after,
              "tier": args.tier, "environment": flags, "command": "BENCHMARK_ZOOM.bat candidate build-only",
              "flags": "MSVC x86 /std:c++17 /EHsc /O2 /W4 /WX; preview /LARGEADDRESSAWARE",
              "host_os": platform.platform(), "returncode": result["returncode"],
              "binaries": {name:digest(out/name) for name in ("C3XRenderer.dll","biq_preview.exe") if (out/name).is_file()}}
    (out/"build-evidence.json").write_text(json.dumps(record,indent=2))
    print(json.dumps({"out":str(out),"returncode":record["returncode"],"sources_unchanged":record["sources_unchanged"]}))
    raise SystemExit(0 if result["returncode"]==0 and before==after else 1)


if __name__ == "__main__":
    main()
