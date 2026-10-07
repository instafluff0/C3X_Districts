#!/usr/bin/env python3
"""Overlay the current Lab resource shading onto a Renderer64 runtime shader pack.

Renderer64 compiles its shaders from a pinned pack, so Lab edits do not reach
the game by themselves. For every pack shader with a generated counterpart in
the checkout, this applies only the differences that concern resources (baked
composition ground decals, cut-out resource bodies, the natural depth basis of
static and animated resource bodies) and keeps every other byte, including
other features' overlays. The previous pack is kept beside it, and
`resource-overlay.json` records the hashes.
"""
import argparse
import datetime
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def digest(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def resource_hunks(pack_text, source_text, shown=None, match=None):
    """The pack text with the source's resource differences applied, and their count.

    A differing hunk is taken whole when it mentions resources (and `match`,
    when given), so a resource change directly beside another feature's change
    carries that change too; `shown` collects the applied hunks for review
    (--dry-run). Other features' unaccepted shader work that mentions resources
    is excluded with `match`."""
    old, new = pack_text.splitlines(keepends=True), source_text.splitlines(keepends=True)
    result, applied = [], 0
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        text = "".join(old[i1:i2] + new[j1:j2])
        if op != "equal" and "resource" in text.lower() and (match is None or match in text):
            result.extend(new[j1:j2])
            applied += 1
            if shown is not None:
                shown.append("".join("- " + line for line in old[i1:i2]) + "".join("+ " + line for line in new[j1:j2]))
        else:
            result.extend(old[i1:i2])
    return "".join(result), applied


def overlay(pack, backup=None, dry_run=False, match=None):
    from Renderer.lab.preparation import require_current
    require_current(ROOT)
    pack = pack.resolve()
    changed, hunks = {}, {}
    for path in sorted(pack.rglob("*.hlsl")):
        source = ROOT / path.relative_to(pack)
        if not source.is_file():
            continue
        text, shown = path.read_text(), []
        result, applied = resource_hunks(text, source.read_text(), shown, match)
        if result != text:
            changed[path], hunks[path] = result, applied
            if dry_run:
                print(f"##### {path.relative_to(pack).as_posix()}\n" + "".join(hunk + "---\n" for hunk in shown))
    if not changed or dry_run:
        return None
    stamp = datetime.date.today().strftime("%Y%m%d")
    backup = (backup or pack.with_name(pack.name + "-before-resources-" + stamp)).resolve()
    if backup.exists():
        raise ValueError("Backup already exists; choose another: " + str(backup))
    shutil.copytree(pack, backup)
    name = lambda path: path.relative_to(pack).as_posix()
    receipt = pack / "resource-overlay.json"
    record = {
        "previous": json.loads(receipt.read_text()) if receipt.exists() else None,
        "match": match,
        "source_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                        capture_output=True, text=True).stdout.strip(),
        "purpose": "Lab resource shading (composition decals, cut-out bodies, natural depth) overlaid on "
                   "the active runtime pack; other shader code preserved",
        "base": backup.relative_to(ROOT).as_posix(),
        "hunks": {name(path): count for path, count in hunks.items()},
        "inputs": {name(path): digest(path.read_bytes()) for path in changed},
    }
    for path, text in changed.items():
        path.write_text(text)
    record["outputs"] = {name(path): digest(path.read_bytes()) for path in changed}
    (pack / "resource-overlay.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=ROOT / "Renderer/packs/Renderer64ResidentRuntime")
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="print the hunks that would be applied")
    parser.add_argument("--match", help="apply only resource hunks that contain this text")
    args = parser.parse_args()
    record = overlay(args.pack, args.backup, args.dry_run, args.match)
    if not args.dry_run:
        print(json.dumps(record, indent=2) if record else "Pack resource shading is already current")


if __name__ == "__main__":
    main()
