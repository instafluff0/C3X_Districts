"""Compile the selected Cities Lab recipes into a source-agnostic runtime pack.

Layouts and variation are resolved offline. The game selects an immutable
composition by culture, era, size, capital, walls and its existing tile seed.
No source asset discovery or layout search occurs during drawing.
"""
from pathlib import Path
import argparse
import json
import math
import sys
import hashlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Renderer.lab.shared.cities.assets import component
from Renderer.lab.shared.cities.fingerprint import geometry_digest
from Renderer.lab.shared.cities.auxiliary import restore


def normalized_frame(mesh):
    """Retain the Lab derivative's normals; derive tangent frames from its UVs.

    Foundation clipping changes vertices, so an unmodified source-frame index
    cannot be applied to these meshes. This is a normalized-mesh derivation,
    not a claim that these tangent vectors were recovered from the source game.
    """
    vertices = mesh['vertices']
    ts = [[0., 0., 0.] for _ in vertices]
    bs = [[0., 0., 0.] for _ in vertices]
    indices = mesh['topology']['indices']
    for start in range(0, len(indices), 3):
        tri = indices[start:start+3]
        a, b, c = (vertices[i] for i in tri)
        e = [b['position'][j]-a['position'][j] for j in range(3)]
        f = [c['position'][j]-a['position'][j] for j in range(3)]
        u, v = ([p['uv0'][j]-a['uv0'][j] for j in range(2)] for p in (b,c))
        det = u[0]*v[1]-u[1]*v[0]
        if abs(det) < 1e-12:
            continue
        for i in tri:
            for j in range(3):
                ts[i][j] += (e[j]*v[1]-f[j]*u[1])/det
                bs[i][j] += (f[j]*u[0]-e[j]*v[0])/det
    def unit(v):
        length = math.sqrt(sum(x*x for x in v))
        return [x/length for x in v] if length > 1e-10 else None
    normals, tangents, bitangents = [], [], []
    for vertex, t, b in zip(vertices, ts, bs):
        n = unit(vertex['normal'])
        if n is None:
            raise ValueError('City derivative has a zero normal')
        dot = sum(x*y for x,y in zip(t,n))
        tangent = unit([t[j]-dot*n[j] for j in range(3)])
        if tangent is None:
            axis = min(range(3), key=lambda j: abs(n[j]))
            tangent = unit([float(j==axis)-n[axis]*n[j] for j in range(3)])
        cross = [n[1]*tangent[2]-n[2]*tangent[1], n[2]*tangent[0]-n[0]*tangent[2],
                 n[0]*tangent[1]-n[1]*tangent[0]]
        sign = -1 if sum(x*y for x,y in zip(cross,b)) < 0 else 1
        normals.append(n); tangents.append(tangent); bitangents.append([sign*x for x in cross])
    return dict(geometry_digest=geometry_digest(mesh), normals=normals,
                tangents=tangents, bitangents=bitangents)


def recipe_sources():
    # Include the layout inputs read directly by the Lab recipe composer, plus
    # its code. Mesh/material bytes are added by the pack compiler's read log.
    source_layouts=ROOT/"Renderer/lab/out/cities/all-era-source-auditions"
    paths=set(source_layouts.glob('review/*/*/layouts.json'))
    paths.update((ROOT/'Renderer/lab/studies/cities').glob('*.py'))
    paths.update((ROOT/'Renderer/lab/studies/cities').glob('*.json'))
    paths.update((ROOT/'Renderer/lab/shared/cities').glob('*.py'))
    paths.update([Path(__file__),ROOT/'Renderer/native/city_fidelity/prepare_pack.py'])
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def build(output, variants=3, *, return_inputs=False):
    from Renderer.lab.studies.cities.civ3_culture_recipe_review import compose, RECIPE, ERAS
    from Renderer.lab.studies.cities.build_layouts import wall_instances
    from Renderer.native.city_fidelity.prepare_pack import build_pack
    if not 1 <= variants <= 8:
        raise ValueError('Use one to eight precompiled variations')
    intermediate = ROOT/'Renderer/lab/out/cities/integration'
    intermediate.mkdir(parents=True, exist_ok=True)
    designs = []
    for seed in range(variants):
        for profile in json.loads(RECIPE.read_text())['profiles']:
            for era in ERAS:
                design, _ = compose(profile, era, seed)
                design['variant'] = seed
                walls = wall_instances(design['wall_kit'])
                for item in walls:
                    body = component(item['asset'], Path(item['pack']))
                    cx, cy = [(body['lo'][j]+body['hi'][j])/2 for j in (0,1)]
                    c, s = math.cos(item['rotation']), math.sin(item['rotation'])
                    item['offset'] = [item['offset'][0]+item['scale']*(cx*c-cy*s),
                                      item['offset'][1]+item['scale']*(cx*s+cy*c)]
                    item['source_z_factor'] = 1.0
                design['wall_instances'] = walls
                designs.append(design)
                print('RECIPE', profile['id'], era, seed, flush=True)
    assets = set()
    def portable(node):
        if isinstance(node, dict):
            if 'pack' in node:
                node['pack'] = (ROOT/Path(node['pack'])).resolve().relative_to(ROOT).as_posix()
                if 'asset' in node:
                    assets.add((node['pack'], node['asset']))
            for value in node.values(): portable(value)
        elif isinstance(node, list):
            for value in node: portable(value)
    portable(designs)
    auxiliary=json.loads((ROOT/"Renderer/packs/CityRecipeAuxiliaryUV/uv.json").read_text())["meshes"]
    frames = {}
    for pack, asset in sorted(assets):
        for mesh, _ in component(asset, Path(pack))['parts']:
            mesh=restore(mesh,auxiliary)
            key = mesh['asset_id']; frame = normalized_frame(mesh)
            if key in frames and frames[key] != frame:
                raise ValueError('Ambiguous derivative geometry ID: '+key)
            frames[key] = frame
    layout_path, frame_path = intermediate/'layouts.json', intermediate/'frames.json'
    layout_path.write_text(json.dumps(dict(schema='c3x.lab.city_design.v1', designs=designs))+'\n')
    frame_path.write_text(json.dumps(dict(meshes=frames))+'\n')
    meta, consumed = build_pack(output, layout_path, lab_frames=frame_path, layouts_only=True)
    consumed={p:h for p,h in consumed.items() if not p.startswith("Renderer/lab/out/cities/integration/")}
    consumed.update(recipe_sources())
    (output/'recipes.json').write_bytes(layout_path.read_bytes())
    (output/'recipe-inputs.json').write_text(json.dumps(consumed, indent=2)+'\n')
    return (meta, consumed) if return_inputs else meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variants', type=int, default=3)
    args = parser.parse_args()
    build((ROOT/args.output).resolve(), args.variants)
