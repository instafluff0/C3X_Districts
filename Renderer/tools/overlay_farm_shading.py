#!/usr/bin/env python3
"""Overlay the current Lab farm-kit shading onto a Renderer64 runtime shader pack.

Renderer64 compiles its shaders from a pinned pack, so Lab edits do not reach
the game by themselves. For every pack shader with a generated counterpart in
the checkout, this applies only the differences that concern the farm kit (its
natural height-depth basis for fields and props) and keeps every other byte,
including other features' overlays. Review with --dry-run first. The previous
pack is kept beside it, and `farm-overlay.json` records the hashes.
"""
import argparse
import datetime
import difflib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Renderer.tools.overlay_resource_shading import digest


def farm_hunks(pack_text, source_text, shown=None):
    """The pack text with the source's farm-kit differences applied, and their count."""
    old, new = pack_text.splitlines(keepends=True), source_text.splitlines(keepends=True)
    result, applied = [], 0
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        text = "".join(old[i1:i2] + new[j1:j2]).lower()
        if op != "equal" and ("farm_kit" in text or "farm kit" in text):
            result.extend(new[j1:j2])
            applied += 1
            if shown is not None:
                shown.append("".join("- " + line for line in old[i1:i2]) + "".join("+ " + line for line in new[j1:j2]))
        else:
            result.extend(old[i1:i2])
    return "".join(result), applied


def overlay(pack, backup=None, dry_run=False):
    from Renderer.lab.preparation import require_current
    require_current(ROOT)
    pack = pack.resolve()
    changed, hunks = {}, {}
    for path in sorted(pack.rglob("*.hlsl")):
        source = ROOT / path.relative_to(pack)
        if not source.is_file():
            continue
        text, shown = path.read_text(), []
        result, applied = farm_hunks(text, source.read_text(), shown)
        if result != text:
            changed[path], hunks[path] = result, applied
            if dry_run:
                print(f"##### {path.relative_to(pack).as_posix()}\n" + "".join(hunk + "---\n" for hunk in shown))
    if not changed or dry_run:
        return None
    stamp = datetime.date.today().strftime("%Y%m%d")
    backup = (backup or pack.with_name(pack.name + "-before-farms-" + stamp)).resolve()
    if backup.exists():
        raise ValueError("Backup already exists; choose another: " + str(backup))
    shutil.copytree(pack, backup)
    name = lambda path: path.relative_to(pack).as_posix()
    receipt = pack / "farm-overlay.json"
    record = {
        "previous": json.loads(receipt.read_text()) if receipt.exists() else None,
        "source_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                        capture_output=True, text=True).stdout.strip(),
        "purpose": "Lab farm-kit shading (natural depth basis of fields and props) overlaid on the "
                   "active runtime pack; other shader code preserved",
        "base": backup.relative_to(ROOT).as_posix(),
        "hunks": {name(path): count for path, count in hunks.items()},
        "inputs": {name(path): digest(path.read_bytes()) for path in changed},
    }
    for path, text in changed.items():
        path.write_text(text)
    record["outputs"] = {name(path): digest(path.read_bytes()) for path in changed}
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=ROOT / "Renderer/packs/Renderer64ResidentRuntime")
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="print the hunks that would be applied")
    args = parser.parse_args()
    record = overlay(args.pack, args.backup, args.dry_run)
    if not args.dry_run:
        print(json.dumps(record, indent=2) if record else "Pack farm shading is already current")


if __name__ == "__main__":
    main()
