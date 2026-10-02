#!/usr/bin/env python3
"""Append generic resident vertex adapters to an exact inherited shader pack.

The baseline pack is never edited. A complete candidate is built in a temporary
directory, checked against its input hashes, then published to a new output path.
All inherited shader bodies and unrelated files retain their exact original bytes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "input": "Renderer/lab/shared/shaders/objects/resident_instance.hlsl",
    "natural": "Renderer/lab/shared/shaders/objects/instance_geometry.hlsl",
    "reflection": "Renderer/native/environment_refresh/prepare.py",
    "rigid": "Renderer/native/render_core/rigid_feature.hlsl",
    "caster": "Renderer/native/render_core/rigid_caster.hlsl",
}


def digest(content):
    return hashlib.sha256(content).hexdigest()


def function(text, name, occurrence=0):
    pattern = (r"\b(?:RigidInput|InstanceInput|FeaturePixelInput|RigidPixel|InstancePixel|P)\s+"
               + re.escape(name) + r"\([^)]*\)\s*\{")
    matches = list(re.finditer(pattern, text))
    if occurrence >= len(matches):
        raise ValueError("Missing resident adapter: " + name)
    start, end = matches[occurrence].start(), matches[occurrence].end()
    depth = 1
    while depth and end < len(text):
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    if depth:
        raise ValueError("Unclosed resident adapter: " + name)
    return text[start:end] + "\n"


def adapters(source_bytes):
    source = {name: content.decode() for name, content in source_bytes.items()}
    shared = source["input"] + "\n"
    natural = function(source["natural"], "resident_natural_input")
    body = function(source["natural"], "VSResidentInstance", 1)
    caster = function(source["natural"], "VSResidentInstance", 0)
    placed = function(source["natural"], "VSResidentPlacedInstance")
    reflection = function(source["reflection"], "VSResidentReflectionInstance")
    rigid = function(source["rigid"], "resident_rigid_input")
    rigid_body = function(source["rigid"], "VSResidentSharedFeature")
    rigid_reflection = function(source["rigid"], "VSResidentSharedFeatureReflection")
    rigid_caster = function(source["caster"], "VSResidentSharedCaster")
    rigid_placed = function(source["caster"], "VSResidentPlacedCaster")
    specifications = {
        "source_fidelity/objects.hlsl":
            (natural + body, ["VSInstance"], ["VSResidentInstance"]),
        "source_fidelity/instance_caster.hlsl":
            (natural + caster + placed, ["VSInstance"], ["VSResidentInstance", "VSResidentPlacedInstance"]),
        "environment_refresh/objects.hlsl":
            (natural + body + reflection, ["VSInstance", "VSReflectionInstance"],
             ["VSResidentInstance", "VSResidentReflectionInstance"]),
        "city_fidelity/objects.hlsl":
            (natural + body + reflection, ["VSInstance", "VSReflectionInstance"],
             ["VSResidentInstance", "VSResidentReflectionInstance"]),
        "city_fidelity/rigid_feature.hlsl":
            (rigid + rigid_body + rigid_reflection, ["VSSharedFeature", "VSSharedFeatureReflection"],
             ["VSResidentSharedFeature", "VSResidentSharedFeatureReflection"]),
        "city_fidelity/rigid_caster.hlsl":
            (rigid_caster + rigid_placed, ["VSSharedCaster"],
             ["VSResidentSharedCaster", "VSResidentPlacedCaster"]),
    }
    return {"Renderer/native/" + name: (shared + adapter, required, added)
            for name, (adapter, required, added) in specifications.items()}


def inventory(directory):
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Shader pack contains a symlink: " + path.relative_to(directory).as_posix())
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = digest(path.read_bytes())
    return result


def portable_name(path, source_root, kind):
    try:
        return path.relative_to(source_root).as_posix()
    except ValueError:
        return "<configured " + kind + ">"


def prepare(baseline, output, source_root=ROOT):
    baseline, output, source_root = Path(baseline).resolve(), Path(output).resolve(), Path(source_root).resolve()
    if not baseline.is_dir():
        raise ValueError("Baseline shader pack is missing")
    if output.exists():
        raise ValueError("Output already exists; select a new candidate path")
    if output.is_relative_to(baseline) or baseline.is_relative_to(output):
        raise ValueError("Baseline and output must be separate directories")
    initial = inventory(baseline)
    tool_source = Path(__file__).read_bytes()
    source_bytes = {name: (source_root / path).read_bytes() for name, path in SOURCES.items()}
    prepared, records = {}, {}
    for name, (adapter, required, added) in adapters(source_bytes).items():
        path = baseline / name
        if name not in initial:
            raise ValueError("Missing inherited shader: " + name)
        original = path.read_bytes()
        text = original.decode()
        for entry in required:
            if not re.search(r"\b" + entry + r"\s*\([^)]*\)\s*\{", text):
                raise ValueError(name + " lacks inherited entry " + entry)
        for entry in added:
            if re.search(r"\b" + entry + r"\s*\(", text):
                raise ValueError(name + " already contains " + entry)
        if re.search(r"\b(?:ResidentInstanceInput|ResidentPlacement|C3XResidentPlacements|C3X_RESIDENT_INSTANCE_INPUT)\b", text):
            raise ValueError(name + " already contains the resident input namespace")
        addition = ("\n// Persistent generic instance submission: original shader prefix is preserved.\n" + adapter).encode()
        prepared[name] = original + addition
        records[name] = {"baseline_sha256": digest(original), "candidate_sha256": digest(prepared[name]),
                         "baseline_bytes": len(original), "added_bytes": len(addition),
                         "entries": added, "original_prefix_exact": True}
    receipt = {"version": 1, "baseline": portable_name(baseline, source_root, "baseline"),
               "output": portable_name(output, source_root, "output"),
               "baseline_sha256": initial,
               "source_sha256": {SOURCES[name]: digest(content) for name, content in source_bytes.items()},
               "tool": "Renderer/tools/prepare_resident_submission_shaders.py", "tool_sha256": digest(tool_source),
               "changed": records, "unchanged_inherited_files": True,
               "contract": "Append generic resident input, decode and vertex entries; preserve all inherited shader bodies."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="resident-shaders-", dir=output.parent) as directory:
        candidate = Path(directory) / "candidate"
        shutil.copytree(baseline, candidate)
        if inventory(candidate) != initial or inventory(baseline) != initial:
            raise ValueError("Baseline shader inputs changed during candidate preparation")
        for name, updated in prepared.items():
            original = (candidate / name).read_bytes()
            if updated[:len(original)] != original:
                raise ValueError("Inherited shader prefix changed: " + name)
            (candidate / name).write_bytes(updated)
        expected = dict(initial)
        expected.update({name: digest(content) for name, content in prepared.items()})
        if inventory(candidate) != expected or inventory(baseline) != initial:
            raise ValueError("Candidate shader closure differs from the validated append-only result")
        if any((source_root / SOURCES[name]).read_bytes() != content for name, content in source_bytes.items()):
            raise ValueError("Resident adapter sources changed during candidate preparation")
        if Path(__file__).read_bytes() != tool_source:
            raise ValueError("Resident preparation tool changed during candidate preparation")
        (candidate / "resident-submission-overlay.json").write_text(json.dumps(receipt, indent=2) + "\n")
        candidate.rename(output)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Frozen runtime shader root to inherit")
    parser.add_argument("--out", type=Path, required=True, help="New candidate shader root")
    parser.add_argument("--source-root", type=Path, default=ROOT, help="Repository containing the resident adapter sources")
    args = parser.parse_args(argv)
    receipt = prepare(args.baseline, args.out, args.source_root)
    print(json.dumps({"output": receipt["output"], "changed": list(receipt["changed"]),
                      "receipt": "resident-submission-overlay.json"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
