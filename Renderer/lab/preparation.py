"""Prepare current production shader bindings, without touching art or approvals.

Adapters run in a disposable mirror first. Only a complete, validated result
replaces generated files. A cache miss must reproduce existing files exactly;
an edited generated file is never silently overwritten. The receipt is a small
disposable build cache, not a visual revision or a historical handoff.
"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = "Renderer/lab/.cache/shader-preparation.json"
STEPS = (
    ("Renderer/native/source_fidelity/prepare.py", "--shaders-only"),
    ("Renderer/native/source_fidelity/prepare_hydrology.py",),
    ("Renderer/native/environment_refresh/prepare.py",),
    ("Renderer/native/environment_refresh/prepare_unit_shader.py",),
    ("Renderer/native/city_fidelity/prepare_shader.py",),
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_paths(root):
    paths = {step[0] for step in STEPS}
    paths.update(("Renderer/lab/preparation.py",
        "Renderer/native/render_core/generate_shaders.py",
        "Renderer/native/render_core/shadow_receiver.hlsl",
        "Renderer/native/render_core/terrain_scene.hlsl",
        "Renderer/native/render_core/source_caster.hlsl",
        "Renderer/native/source_fidelity/shadow_adapter.hlsl",
        "Renderer/native/terrain_rendering.hlsl",
        "Renderer/native/integrated_terrain.hlsl",
        "Renderer/lab/shared/hydrology/field.h",
        "Renderer/lab/shared/hydrology/river_corridor.h"))
    paths.update(p.relative_to(root).as_posix()
                 for p in (root / "Renderer/lab/shared/shaders").rglob("*.hlsl"))
    return sorted(paths)


def local(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Preparation path escapes the repository: " + name)
    return path


def inputs(root):
    return {name: digest(local(root, name)) for name in input_paths(root)}


def receipt(root):
    path = root / RECEIPT
    return json.loads(path.read_text()) if path.exists() else None


def require_current(root=ROOT):
    record = receipt(root)
    if not record or record.get("inputs") != inputs(root):
        raise ValueError("Production shader bindings are stale; run Renderer/renderer.py prepare, then rerender")
    for name, expected in record["outputs"].items():
        path = local(root, name)
        if not path.is_file() or digest(path) != expected:
            raise ValueError("Generated shader is missing or edited: " + name + "; rerun preparation before reviewing")


def generate(mirror):
    for step in STEPS:
        result = subprocess.run([sys.executable, str(mirror / step[0]), *step[1:]],
                                cwd=mirror, capture_output=True, text=True)
        if result.returncode:
            raise ValueError("Shader preparation failed: " + step[0] + "\n" + result.stderr[-4000:])


def prepare(root=ROOT):
    before = inputs(root)
    previous = receipt(root)
    # Missing generated files are recoverable. Existing edits must be kept,
    # even when source inputs also changed or a prior receipt was lost.
    for name, expected in (previous or {}).get("outputs", {}).items():
        path = local(root, name)
        if path.exists() and digest(path) != expected:
            raise ValueError("Preserving edited generated shader: " + name +
                             "; move the intended edit into its shared source/adapter first")
    if previous and previous.get("inputs") == before and all(
            local(root, name).is_file() for name in previous["outputs"]):
        return []
    cache = root / "Renderer/lab/.cache"
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prepare-", dir=cache) as directory:
        mirror = Path(directory).resolve()
        for name in before:
            target = local(mirror, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local(root, name), target)
        generate(mirror)
        produced = {p.relative_to(mirror).as_posix(): digest(p)
                    for p in mirror.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                    and p.relative_to(mirror).as_posix() not in before}
        if not produced:
            raise ValueError("Shader preparation produced no output")
        # All adapter inputs must remain immutable, including copied inputs.
        if inputs(root) != before or any(digest(local(mirror, name)) != value for name, value in before.items()):
            raise ValueError("Shader inputs changed during preparation; rerun")
        changed = []
        for name, value in produced.items():
            target = local(root, name)
            if target.exists() and digest(target) != value:
                old = (previous or {}).get("outputs", {}).get(name)
                if old is None or digest(target) != old:
                    raise ValueError("Preserving unverified generated shader: " + name +
                                     "; establish preparation from the matching source before editing")
            if not target.exists() or digest(target) != value:
                changed.append(name)
        # Validate the whole output set before publishing any file. Do not
        # delete retired outputs here; their consumers need a dependency audit.
        for name in changed:
            target = local(root, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local(mirror, name), target)
        record = {"inputs": before, "outputs": produced}
        destination = root / RECEIPT
        with tempfile.NamedTemporaryFile(mode="w", prefix="shader-", suffix=".tmp", dir=cache, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(record, indent=2) + "\n")
        try:
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return sorted(changed)
