#!/usr/bin/env python3
"""Make an offline city study pack with buried mesh geometry clipped at grade.

The source pack remains untouched. This derivative removes negative-Z source
foundations from chosen building pools while retaining above-ground triangles,
normals, UVs, and material bindings. It is not a runtime terrain adjustment.
"""

import argparse
import json
import math
import os
import shutil
from pathlib import Path


def intersection(a, b, grade):
    za, zb = a["position"][2], b["position"][2]
    t = (grade - za) / (zb - za)
    vertex = {}
    for key in a:
        if isinstance(a[key], list):
            values = [u + t * (v - u) for u, v in zip(a[key], b[key])]
            if key == "normal":
                length = math.sqrt(sum(value * value for value in values))
                values = [value / length for value in values]
            if key == "position":
                values[2] = grade
            vertex[key] = values
        else:
            vertex[key] = a[key]
    return vertex


def clip_triangle(vertices, grade):
    result = []
    for a, b in zip(vertices, vertices[1:] + vertices[:1]):
        inside_a, inside_b = a["position"][2] >= grade, b["position"][2] >= grade
        if inside_a:
            result.append(a)
        if inside_a != inside_b:
            result.append(intersection(a, b, grade))
    return [result[0], result[1], result[2], *(
        [result[0], result[2], result[3]] if len(result) == 4 else [])] if len(result) >= 3 else []


def trim_mesh(mesh, grade):
    original = mesh["vertices"]
    indices = mesh["topology"]["indices"]
    if min(vertex["position"][2] for vertex in original) >= grade and grade == 0:
        return mesh, False
    vertices, triangles = [], []
    # Imported meshes use indexed triangles. Preserve each face's winding;
    # only faces crossing grade need interpolated vertices.
    for start in range(0, len(indices), 3):
        triangle = [original[index] for index in indices[start:start + 3]]
        clipped = clip_triangle(triangle, grade)
        for vertex in clipped:
            triangles.append(len(vertices))
            vertices.append({**vertex, "position": [vertex["position"][0],
                                                   vertex["position"][1],
                                                   vertex["position"][2] - grade]})
    if vertices:
        mesh = {**mesh, "vertices": vertices,
                "topology": {**mesh["topology"], "indices": triangles}}
    return mesh if vertices else None, True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source-pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--culture", action="append", default=[],
                        help="Exact source culture tag to trim")
    parser.add_argument("--all-cultures", action="store_true",
                        help="Trim individual buildings from every pool in the report")
    parser.add_argument("--grade-map", type=Path,
                        help="Offline source-entry foundation cut heights")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve existing Lab derivative; choose a new output path")
    report = json.loads(args.report.read_text())
    cultures = ([pool["source_culture"] for pool in report["pools"]]
                if args.all_cultures else args.culture)
    if not cultures:
        raise ValueError("select --all-cultures or at least one --culture")
    grades = json.loads(args.grade_map.read_text()) if args.grade_map else {}
    manifest = json.loads((args.source_pack / "manifest.json").read_text())
    selected = {record["asset_id"]: {**record, "source_culture": pool["source_culture"]}
                for pool in report["pools"] if pool["source_culture"] in cultures
                for record in pool["selected"]}
    if not selected:
        raise ValueError("no source city components in requested pools")
    shutil.copytree(args.source_pack, args.output, copy_function=os.link)
    changed_assets, removed_parts = 0, 0
    for asset, record in selected.items():
        grade = grades.get("culture_defaults", {}).get(record["source_culture"], 0.0)
        for prefix, value in grades.get("entry_prefixes", {}).items():
            if record["entry"].startswith(prefix):
                grade = value
        grade = grades.get("entries", {}).get(record["entry"], grade)
        landmark_path = args.output / manifest["assets"][asset]["landmark"]
        landmark = json.loads(landmark_path.read_text())
        omitted = set()
        modified = False
        for index, name in enumerate(landmark["components"]["geometry"]):
            geometry_path = args.output / name
            mesh = json.loads(geometry_path.read_text())
            trimmed, changed = trim_mesh(mesh, grade)
            if not changed:
                continue
            modified = True
            if trimmed is None:
                omitted.add(index)
            else:
                temporary = geometry_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(trimmed, separators=(",", ":")) + "\n")
                os.replace(temporary, geometry_path)
        if omitted:
            landmark["draw_bindings"] = [binding for binding in landmark["draw_bindings"]
                                          if binding["geometry"] not in omitted]
            if not any("worked" in binding["states"] for binding in landmark["draw_bindings"]):
                raise ValueError(f"grade clip removed every worked part of {record['entry']}")
            temporary = landmark_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(landmark, indent=2) + "\n")
            os.replace(temporary, landmark_path)
            removed_parts += len(omitted)
        changed_assets += modified
    print(json.dumps({"selected_components": len(selected), "trimmed_components": changed_assets,
                      "removed_buried_parts": removed_parts, "cultures": cultures}))


if __name__ == "__main__":
    main()
