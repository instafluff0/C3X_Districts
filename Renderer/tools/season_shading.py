#!/usr/bin/env python3
"""Apply the seasonal shading (Renderer/lab/shared/seasons) to a shader tree.

Renderer64 compiles its shaders from the runtime pack, so the season overlay
must be written into it. Every season insertion is marked, so this tool strips
any earlier seasons from a file and re-applies the current ones; other
features' overlays and every other byte are kept. Re-run it after any other
`overlay_*_shading.py` tool, which may copy a region without the seasons back
over the pack.

    python3 Renderer/tools/season_shading.py --dry-run            # runtime pack, report only
    python3 Renderer/tools/season_shading.py                      # runtime pack, backup + receipt
    python3 Renderer/tools/season_shading.py --strip              # remove seasons from the pack
    python3 Renderer/tools/season_shading.py --lab-root DIR       # Lab shader root from the checkout

`--lab-root` writes a private shader root for `C3X_RENDERER_SHADER_SOURCE_ROOT`:
the checkout's flattened shaders with seasons applied (only changed files are
rewritten, so the runtime's per-entry shader cache survives).
"""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Renderer.lab.shared.seasons import seasons
from Renderer.tools.overlay_resource_shading import digest


def overlay(pack, backup=None, dry_run=False, strip=False):
    pack = pack.resolve()
    changed = {}
    for relative in seasons.TARGETS:
        path = pack / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        text = path.read_text()
        result = seasons.strip(text) if strip else seasons.apply(relative, text)
        if result != text:
            changed[relative] = result
    for relative in changed:
        print(("strip " if strip else "seasons ") + relative)
    if not changed or dry_run:
        return None
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    backup = (backup or pack.with_name(pack.name + "-before-seasons-" + stamp)).resolve()
    if backup.exists():
        raise ValueError("Backup already exists; choose another: " + str(backup))
    shutil.copytree(pack, backup)
    receipt = pack / "season-overlay.json"
    record = {
        "previous": json.loads(receipt.read_text()) if receipt.exists() else None,
        "source_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                        capture_output=True, text=True).stdout.strip(),
        "purpose": ("seasonal shading removed" if strip else
                    "seasonal shading (Autumn v18, Winter v24-laden, Spring v20) as marked, season-guarded "
                    "insertions; other shader code preserved"),
        "base": backup.relative_to(ROOT).as_posix() if backup.is_relative_to(ROOT) else str(backup),
        "library": {name: digest((seasons.HERE / name).read_bytes())
                    for name in ("autumn.hlsl", "winter_natural.hlsl", "winter_integrated.hlsl", "spring.hlsl",
                                 "seasons.py")},
        "inputs": {relative: digest((pack / relative).read_bytes()) for relative in changed},
    }
    for relative, text in changed.items():
        (pack / relative).write_text(text)
    record["outputs"] = {relative: digest((pack / relative).read_bytes()) for relative in changed}
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    return record


def lab_root(target):
    """A private Lab shader root: the checkout's shaders with seasons applied."""
    target = Path(target).resolve()
    sources = [path for path in (ROOT / "Renderer/native").rglob("*.hlsl") if "build" not in
               path.relative_to(ROOT / "Renderer/native").parts]
    sources.append(ROOT / seasons.WATER_SURFACE)
    written = 0
    for source in sources:
        relative = source.relative_to(ROOT).as_posix()
        text = source.read_text()
        if relative in seasons.TARGETS:
            text = seasons.apply(relative, text)
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file() or path.read_text() != text:
            path.write_text(text)
            written += 1
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pack", type=Path, default=ROOT / "Renderer/packs/Renderer64ResidentRuntime")
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strip", action="store_true", help="remove every season insertion")
    parser.add_argument("--lab-root", type=Path, help="write a private Lab shader root instead of the pack")
    args = parser.parse_args()
    if args.lab_root:
        print(f"{lab_root(args.lab_root)} files written to {args.lab_root}")
        return
    record = overlay(args.pack, args.backup, args.dry_run, args.strip)
    if record:
        print("receipt:", (args.pack / "season-overlay.json").as_posix())


if __name__ == "__main__":
    main()
