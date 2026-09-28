"""Build a private Renderer64 shader bundle with selected current Lab materials.

Unselected runtime shaders remain byte-identical. Adapters run in a disposable
mirror, preserving generated files owned by concurrent Lab work. This does not
stage binaries, change reference images, or accept a visual result.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Renderer.lab import preparation

MATERIALS = {
    "city-lighting": (),
    "terrain": ("Renderer/native/city_fidelity/terrain.hlsl",),
    "mountain": ("Renderer/native/city_fidelity/mountain.hlsl",),
    "coast": ("Renderer/native/city_fidelity/hydrology.hlsl",
              "Renderer/sandbox/water_surface.hlsl"),
}


def prepare(base, output, materials):
    base, output = base.resolve(), output.resolve()
    if not base.is_relative_to(ROOT) or not output.is_relative_to(ROOT):
        raise ValueError("Bundle paths must be inside the repository")
    if output.exists():
        raise ValueError("Use a new output directory; existing evidence is preserved")
    if not materials or any(name not in MATERIALS for name in materials):
        raise ValueError("Select a supported material category")
    if materials == ['city-lighting'] or materials == ('city-lighting',):
        from Renderer.native.city_fidelity.scene_lights import upgrade
        paths=list(base.rglob('*.hlsl'))
        if not paths:raise ValueError('Runtime shader bundle is missing')
        inputs={p.relative_to(base).as_posix():preparation.digest(p) for p in paths}
        adapter=ROOT/'Renderer/native/city_fidelity/scene_lights.py'
        adapter_hash=preparation.digest(adapter)
        output.mkdir(parents=True)
        outputs={};changed=[]
        for path in paths:
            name=path.relative_to(base).as_posix();source=path.read_text()
            updated=upgrade(source) if 'city_fidelity' in path.parts else source
            target=output/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(updated)
            outputs[name]=preparation.digest(target)
            if source!=updated:changed.append(name)
        if adapter_hash!=preparation.digest(adapter) or any(preparation.digest(base/name)!=value for name,value in inputs.items()):
            raise ValueError('City shader inputs changed during preparation')
        record=dict(base=base.relative_to(ROOT).as_posix(),inputs=inputs,outputs=outputs,
                    changed=changed,adapter_sha256=adapter_hash)
        (output/'city-light-preparation.json').write_text(json.dumps(record,indent=2)+'\n')
        return record
    if 'city-lighting' in materials:raise ValueError('Prepare city lighting separately to preserve selected terrain materials')
    inputs = preparation.inputs(ROOT)
    if "coast" in materials:
        name = "Renderer/sandbox/water_surface.hlsl"
        inputs[name] = preparation.digest(ROOT / name)
    base_files = {p.relative_to(base).as_posix(): preparation.digest(p)
                  for p in base.rglob("*.hlsl")}
    if not base_files:
        raise ValueError("Runtime shader bundle is missing")
    selected = {path for name in materials for path in MATERIALS[name]}
    if not selected.issubset(base_files):
        raise ValueError("Runtime bundle lacks a selected material")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="materials-", dir=output.parent) as directory:
        mirror = Path(directory) / "source"
        bundle = Path(directory) / "bundle"
        for name in inputs:
            target = mirror / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        preparation.generate(mirror)
        for name in base_files:
            source = (mirror if name in selected else base) / name
            if name in selected and name in (MATERIALS["terrain"][0], MATERIALS["mountain"][0]) and "#ifdef SANDBOX_TERRAIN_MATERIAL" not in source.read_text():
                raise ValueError("Selected material lacks retained GPU outputs: " + name)
            target = bundle / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        if any(preparation.digest(ROOT / name) != value for name, value in inputs.items()) or any(
                preparation.digest(base / name) != value for name, value in base_files.items()):
            raise ValueError("Shader sources changed during preparation")
        outputs = {name: preparation.digest(bundle / name) for name in base_files}
        record = {"materials": sorted(materials), "inputs": inputs,
                  "base": base.relative_to(ROOT).as_posix(), "base_shaders": base_files,
                  "outputs": outputs, "selected": sorted(selected),
                  "tool_sha256": preparation.digest(Path(__file__)),
                  "changed": sorted(name for name in outputs if outputs[name] != base_files[name])}
        (bundle / "material-preparation.json").write_text(json.dumps(record, indent=2) + "\n")
        # Compiled shader caches are derived, source-hash checked, and rebuilt
        # by D3D. They are deliberately excluded from the candidate's inputs.
        bundle.rename(output)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "Renderer/packs/Renderer64CutoverControl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("materials", choices=tuple(MATERIALS), nargs="+")
    args = parser.parse_args()
    record = prepare(args.base, args.out, args.materials)
    print(json.dumps({"changed": record["changed"], "shader_count": len(record["outputs"])}, indent=2))


if __name__ == "__main__":
    main()
