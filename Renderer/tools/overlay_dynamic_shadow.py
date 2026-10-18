#!/usr/bin/env python3
"""Overlay the current dynamic-shadow policy onto a Renderer64 runtime shader pack.

Renderer64 compiles its shaders from a pinned pack, so Lab edits do not reach
the game by themselves. This replaces only the shared dynamic-shadow policy
block and the animated-resource ground-shadow alpha (surface kind 15) with the
current generated Lab text, in every pack shader that carries them. The rest of
each kind-15 branch must already match the Lab exactly; everything else keeps
its bytes. The previous pack is kept beside it, and
`dynamic-shadow-overlay.json` records the hashes.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
POLICY = "Renderer/lab/shared/shaders/lighting/shadow_policy.hlsl"
OLD_POLICY = """// Shared scalar policy; also included inside the CPU lighting namespace.
// Dynamic ground footprints approximate lost direct illumination on a cached
// background. Self/world receivers preserve their material's ambient term.
static const float c3x_dynamic_shadow_opacity = 0.48f;
"""
START = "    if (input.panel > 0.5 && input.surface_kind > 14.5 && input.surface_kind < 15.5)\n    {"
AFTER = "clip(coverage - 0.08);\n"
UNTIL = "        clip(alpha - 0.004);"


def digest(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def alpha_span(text):
    """The kind-15 alpha statements, plus the branch with them removed."""
    if text.count(START) != 1:
        return None
    start = text.index(START)
    depth, end = 1, text.index("{", start) + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    body = text[start:end]
    if body.count(AFTER) != 1 or body.count(UNTIL) != 1:
        return None
    a = start + body.index(AFTER) + len(AFTER)
    b = start + body.index(UNTIL)
    return a, b, text[start:a] + text[b:end]


def updated(pack_text, lab_text, policy):
    text = pack_text
    if OLD_POLICY in text:
        text = text.replace(OLD_POLICY, policy)
    pack_span, lab_span = alpha_span(text), alpha_span(lab_text) if lab_text else None
    if pack_span and lab_span and text[pack_span[0]:pack_span[1]] != lab_text[lab_span[0]:lab_span[1]]:
        if pack_span[2] != lab_span[2]:
            raise ValueError("kind-15 branch differs from the Lab beyond its alpha")
        text = text[:pack_span[0]] + lab_text[lab_span[0]:lab_span[1]] + text[pack_span[1]:]
    if "c3x_dynamic_shadow_opacity *" in text or "static const float c3x_dynamic_shadow_opacity" in text:
        raise ValueError("old dynamic-shadow text remains")
    return text


def overlay(pack, backup=None, check=False):
    from Renderer.lab.preparation import require_current
    require_current(ROOT)
    pack = pack.resolve()
    policy = (ROOT / POLICY).read_text()
    changed = {}
    for path in sorted(pack.rglob("*.hlsl")):
        name = path.relative_to(pack).as_posix()
        text = path.read_text()
        if name.endswith("lighting/shadow_policy.hlsl") and text == OLD_POLICY:
            changed[path] = policy
            continue
        if OLD_POLICY not in text and "c3x_dynamic_shadow_opacity *" not in text:
            continue
        lab = ROOT / name
        try:
            result = updated(text, lab.read_text() if lab.exists() else "", policy)
        except ValueError as error:
            raise ValueError(f"{name}: {error}") from None
        if result != text:
            changed[path] = result
    if check or not changed:
        return {"changed": sorted(p.relative_to(pack).as_posix() for p in changed)}
    stamp = datetime.date.today().strftime("%Y%m%d")
    backup = (backup or pack.with_name(pack.name + "-before-dynamic-shadow-" + stamp)).resolve()
    if backup.exists():
        raise ValueError("Backup already exists; choose another: " + str(backup))
    shutil.copytree(pack, backup)
    name = lambda path: path.relative_to(pack).as_posix()
    record = {
        "source_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                        capture_output=True, text=True).stdout.strip(),
        "purpose": "Dynamic ground-shadow opacity from frame_illumination, uniform resource "
                   "footprint alpha; other shader code preserved",
        "policy": POLICY,
        "base": backup.relative_to(ROOT).as_posix(),
        "inputs": {name(path): digest(path.read_bytes()) for path in changed},
    }
    for path, text in changed.items():
        path.write_text(text)
    record["outputs"] = {name(path): digest(path.read_bytes()) for path in changed}
    (pack / "dynamic-shadow-overlay.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=ROOT / "Renderer/packs/Renderer64ResidentRuntime")
    parser.add_argument("--backup", type=Path, help="Where to keep the previous pack (default: dated sibling)")
    parser.add_argument("--check", action="store_true", help="List the shaders that would change")
    args = parser.parse_args()
    record = overlay(args.pack, args.backup, args.check)
    print(json.dumps({key: record[key] for key in ("changed", "outputs", "base") if key in record}, indent=2))


if __name__ == "__main__":
    main()
