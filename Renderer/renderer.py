#!/usr/bin/env python3
"""Category-based C3X visual workbench. Production appearance is the baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
LAB = ROOT / "Renderer/lab"
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def checksum(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relative(path):
    return path.resolve().relative_to(ROOT).as_posix()


def local(path):
    target = (ROOT / path).resolve()
    if not target.is_relative_to(ROOT):
        raise ValueError("Path escapes the repository")
    return target


def catalog():
    return read(LAB / "catalog.json")["categories"]


def standard_path(category):
    entries = catalog()
    if category not in entries:
        raise ValueError("Unknown category: " + category)
    return LAB / "categories" / entries[category] / "standard.json"


def standard(category):
    return read(standard_path(category))


def affected(category):
    """Transitive consumers select a bounded, deduplicated preview set."""
    selected = {category}
    changed = True
    entries = {key: standard(key) for key in catalog()}
    while changed:
        changed = False
        for key, value in entries.items():
            if key not in selected and selected.intersection(value["depends_on"]):
                selected.add(key)
                changed = True
    return sorted(selected)


def implementation_identity():
    """Internal stale-preview guard; category pages remain free of hash ledgers."""
    paths = set()
    for category in catalog():
        paths.update(local(p) for p in standard(category)["implementation"])
    paths.update((ROOT / "Renderer/native").rglob("*.hlsl"))
    paths.update((ROOT / "Renderer/native").glob("*.cpp"))
    paths.update((ROOT / "Renderer/native").glob("*.h"))
    paths.update([ROOT / "Renderer/default.custom_rendering.txt", ROOT / "Renderer/custom.custom_rendering.txt"])
    # Binary pack indices include material/texture identities; source asset
    # validation remains part of the importer and production verification.
    paths.update((ROOT / "Renderer/packs").glob("*/*.bin"))
    records = {relative(path): checksum(path) for path in sorted(paths) if path.is_file()}
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def scene(category, case, destination):
    """Small complete synthetic world with no implied live-game provenance."""
    recipe = standard(category)["recipe"]
    rows = []
    for y in range(32):
        for x in range(y % 2, 32, 2):
            base, real = recipe["terrain"], recipe["feature"]
            river = 0
            if recipe["feature"] in (5, 6, 7, 8):
                base, real = 2, 2
                if (x, y) in ((16, 16), (18, 16), (16, 18)):
                    real = recipe["feature"]
            if case == "gameplay":
                if x < 12:
                    base = real = 1
                if (x, y) == (18, 14):
                    base, real = 2, 5
                if (x, y) == (14, 12):
                    base, real = 2, 7
                if (x, y) == (20, 10):
                    base, real = 2, 6
            if category == "transitions":
                base = real = (2 if x < 16 else 1) if y < 16 else (0 if x < 16 else 3)
            if category in ("shorelines", "seas-oceans"):
                base = real = 2 if x < 13 else 11 if x < 17 else 12 if x < 23 else 13
                if category == "seas-oceans" and case == "detail":
                    base = real = 12 if x < 16 else 13
            if category in ("rivers", "floodplains"):
                # Opposite edge bits agree on the diagonal raw-coordinate chain.
                river = 2 if x == y else 32 if x == y + 2 else 0
            rows.append(f"{x},{y},{base},{real},0,0,{river}")
    destination.write_text("C3X_BIQ_TERRAIN_V3,32,32,512\n" + "\n".join(rows) + "\n")


def native_render(category, case, hour, zoom, output, *, baseline=False):
    from Renderer.tools.renderer_dev import native_command_result
    if category in ("units", "animation"):
        raise ValueError("Unit/animation roster replay migration is still required; a terrain-only image cannot establish this category's baseline")
    output.mkdir(parents=True, exist_ok=True)
    csv = output / "scene.csv"
    scene(category, case, csv)
    name = f"{case}-h{hour:02}-z{zoom}"
    image = output / (name + ".bmp")
    dll = ROOT / "Renderer/bin/C3XRenderer.dll" if baseline else ROOT / "Renderer/native/build/candidate/C3XRenderer.dll"
    if baseline and checksum(dll) != read(LAB / "baseline.json")["dll_sha256"]:
        raise ValueError("Staged DLL differs from the user-approved baseline")
    if not dll.is_file():
        raise ValueError("Build the candidate DLL before native rendering")
    # Clear all diagnostic overrides that could otherwise change the scene.
    env = {
        "C3X_RENDERER_VISUAL_PROFILE": "", "C3X_RENDERER_TRACE": "0",
        "C3X_RENDERER_TRACE_FILE": "", "C3X_RENDERER_PREVIEW_COLOR": "",
        "C3X_RENDERER_PREVIEW_CUSTOM_DEFINITIONS": r"..\..\Renderer\custom.custom_rendering.txt",
        "C3X_RENDERER_PREVIEW_OBJECTS": "1" if standard(category)["recipe"]["objects"] else "",
        "C3X_RENDERER_PREVIEW_CITY": "0,3,1,1",
        "C3X_RENDERER_PREVIEW_REPLAY": "", "C3X_RENDERER_PREVIEW_MINIMAP": "",
        "C3X_RENDERER_PREVIEW_EDITS": "", "C3X_RENDERER_PREVIEW_ANIMATION": "",
        "C3X_RENDERER_PREVIEW_UNITS": "", "C3X_RENDERER_PREVIEW_SEASON": "0",
        "C3X_RENDERER_PREVIEW_ACTIVE_VOLCANO": "",
        "C3X_RENDERER_FIDELITY_SHADOW_CONTROL": "", "C3X_RENDERER_REFLECTION_CONTROL": "",
        "C3X_RENDERER_CITY_LIGHT_CONTROL": "", "C3X_RENDERER_CITY_GLOW_CONTROL": "",
    }
    def windows(path):
        return "..\\..\\" + relative(path).replace("/", "\\")
    command = " && ".join(f'set "{key}={value}"' for key, value in env.items())
    command += (f' && build\\biq_preview.exe "{windows(dll)}" ..\\.. '
                f'..\\default.custom_rendering.txt "{windows(csv)}" "{windows(image)}" '
                f'640 480 16 16 {zoom} {hour}')
    print(f"Rendering {category}: {name} through the production renderer", flush=True)
    # Keep the VM transport argument short; Parallels rejects some long command
    # strings before launching cmd. This generated script is disposable output.
    batch = output / "render.bat"
    batch.write_text("@echo off\nsetlocal\n" + command.replace(" && ", "\n") + "\nexit /b %errorlevel%\n")
    result = native_command_result("Renderer/native", f'call "{windows(batch)}"')
    if result["status"] != "pass" or "0 fallback, output=" not in result.get("output_tail", "") or not image.is_file():
        raise ValueError("Production renderer failed: " + name)
    return {"image": relative(image), "sha256": checksum(image), "dll_sha256": checksum(dll),
            "case": case, "hour": hour, "zoom": zoom, "backend": "d3d11", "fixture": "synthetic"}


def render(category, *, baseline=False, selected_case=None):
    value = standard(category)
    out = LAB / ("references" if baseline else "out") / category
    if baseline:
        out /= "r1"
        if value["approved_revision"] != 1 or value["references"]:
            raise ValueError("Baseline references already captured or superseded")
    outputs = []
    identity = implementation_identity()
    recipe = value["recipe"]
    for case in ([selected_case] if selected_case else recipe["cases"]):
        for hour in recipe["hours"]:
            for zoom in recipe["zooms"]:
                outputs.append(native_render(category, case, hour, zoom, out / case, baseline=baseline))
    if implementation_identity() != identity:
        raise ValueError("Renderer inputs changed during rendering; rerun the preview")
    result = {"category": category, "revision": value["revision"], "outputs": outputs,
              "implementation_identity": identity, "recipe": recipe,
              "dependency_revisions": {key: standard(key)["approved_revision"] for key in value["depends_on"]}}
    write(out / "render.json", result)
    if baseline:
        value["references"] = {"d3d11": outputs}
        write(standard_path(category), value)
    print(relative(out / "render.json"))
    return result


def compare(category):
    from PIL import Image, ImageChops, ImageDraw
    value = standard(category)
    result = read(LAB / "out" / category / "render.json")
    refs = value["references"].get("d3d11", [])
    if not refs:
        raise ValueError("No approved native references for " + category)
    panels = []
    summaries = []
    for entry in result["outputs"]:
        if checksum(local(entry["image"])) != entry["sha256"]:
            raise ValueError("Candidate image changed after rendering")
        matches = [r for r in refs if all(r[k] == entry[k] for k in ("case", "hour", "zoom", "backend"))]
        if len(matches) != 1:
            raise ValueError("Candidate has no corresponding approved reference")
        ref = matches[0]
        if checksum(local(ref["image"])) != ref["sha256"]:
            raise ValueError("Approved image was modified")
        a = Image.open(local(ref["image"])).convert("RGB")
        b = Image.open(local(entry["image"])).convert("RGB")
        if a.size != b.size:
            raise ValueError("Candidate dimensions differ from reference")
        difference = ImageChops.difference(a, b)
        panel = Image.new("RGB", (a.width * 2, a.height + 28), "#222222")
        panel.paste(a, (0, 28)); panel.paste(b, (a.width, 28))
        draw = ImageDraw.Draw(panel)
        draw.text((8, 8), "Approved baseline", fill="white")
        draw.text((a.width + 8, 8), "Candidate", fill="white")
        panels.append(panel)
        summaries.append({"case": entry["case"], "hour": entry["hour"], "zoom": entry["zoom"],
                          "identical_pixels": difference.getbbox() is None})
    sheet = Image.new("RGB", (max(p.width for p in panels), sum(p.height for p in panels)))
    offset = 0
    for panel in panels:
        sheet.paste(panel, (0, offset)); offset += panel.height
    target = LAB / "out" / category / "compare.png"
    sheet.save(target)
    write(target.with_suffix(".json"), summaries)
    print(json.dumps(summaries, indent=2))
    print(relative(target))
    return summaries


def gallery():
    from PIL import Image, ImageDraw
    cards = []
    for key in catalog():
        value = standard(key)
        refs = value["references"].get("d3d11", [])
        if not refs:
            continue
        selected = [next((r for r in refs if r["case"] == case and r["hour"] == 12), None)
                    for case in ("detail", "gameplay")]
        card = Image.new("RGB", (640, 264), "#222222")
        ImageDraw.Draw(card).text((8, 7), value["title"] + " / current production", fill="white")
        for index, ref in enumerate(selected):
            if ref:
                if checksum(local(ref["image"])) != ref["sha256"]:
                    raise ValueError("Modified reference: " + key)
                im = Image.open(local(ref["image"])).convert("RGB")
                im.thumbnail((320, 240))
                card.paste(im, (index * 320, 24))
        cards.append(card)
    if not cards:
        raise ValueError("No captured category references")
    sheet = Image.new("RGB", (1280, ((len(cards) + 1) // 2) * 264), "#222222")
    for index, card in enumerate(cards):
        sheet.paste(card, ((index % 2) * 640, (index // 2) * 264))
    destination = LAB / "out/gallery.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)
    print(relative(destination))


def validate(*, complete=False):
    entries = catalog()
    errors = []
    for category in entries:
        value = standard(category)
        if value["id"] != category:
            errors.append(category + ": mismatched identity")
        for dependency in value["depends_on"]:
            if dependency not in entries:
                errors.append(category + ": missing dependency " + dependency)
        for path in value["implementation"]:
            if not local(path).is_file():
                errors.append(category + ": missing implementation " + path)
        for refs in value["references"].values():
            for ref in refs:
                if not local(ref["image"]).is_file() or checksum(local(ref["image"])) != ref["sha256"]:
                    errors.append(category + ": missing or modified reference " + ref["image"])
        if complete and not value["references"]:
            errors.append(category + ": reference capture is incomplete")
    visiting, visited = set(), set()
    def visit(category):
        if category in visiting:
            raise ValueError("Category dependency cycle: " + category)
        if category in visited:
            return
        visiting.add(category)
        for dependency in standard(category)["depends_on"]:
            if dependency in entries:
                visit(dependency)
        visiting.remove(category)
        visited.add(category)
    for category in entries:
        visit(category)
    if errors:
        raise ValueError("\n".join(errors))
    return len(entries)


def pending():
    integrated = read(ROOT / "Renderer/integration/status.json")["categories"]
    changes = []
    for category in catalog():
        value = standard(category)
        before = integrated.get(category, {}).get("integrated_revision")
        if value["approved_revision"] != before:
            changes.append({"category": category, "approved": value["approved_revision"], "integrated": before})
    return changes


def approve(category, note):
    """Record an actual user decision; a technical pass is never approval."""
    selected = affected(category)
    identity = implementation_identity()
    updates = []
    for key in selected:
        value = standard(key)
        path = LAB / "out" / key / "render.json"
        if not path.is_file():
            raise ValueError("Render the affected category before approval: " + key)
        result = read(path)
        if result.get("implementation_identity") != identity or result.get("recipe") != value["recipe"]:
            raise ValueError("Preview is stale: " + key)
        expected = {(case, hour, zoom) for case in value["recipe"]["cases"]
                    for hour in value["recipe"]["hours"] for zoom in value["recipe"]["zooms"]}
        actual = {(r["case"], r["hour"], r["zoom"]) for r in result["outputs"]}
        if actual != expected or len(actual) != len(result["outputs"]):
            raise ValueError("Approval requires the complete category preview: " + key)
        revision = value["approved_revision"] + 1
        folder = LAB / "references" / key / f"r{revision}"
        if folder.exists():
            raise ValueError("Reference destination already exists: " + relative(folder))
        for entry in result["outputs"]:
            if checksum(local(entry["image"])) != entry["sha256"]:
                raise ValueError("Preview image changed: " + key)
        updates.append((key, value, result, revision, folder))
    # Validate the entire affected set before writing any approved reference.
    # Prior references remain intact if an I/O error interrupts the new set.
    for key, value, result, revision, folder in updates:
        folder.mkdir(parents=True)
        references = []
        for entry in result["outputs"]:
            target = folder / Path(entry["image"]).name
            shutil.copyfile(local(entry["image"]), target)
            references.append(dict(entry, image=relative(target)))
        value.update(revision=revision, approved_revision=revision,
                     approval={"user_statement": note, "affected_categories": selected},
                     references={"d3d11": references}, candidate=None)
        write(standard_path(key), value)
    print("Approved " + ", ".join(f"{key} r{revision}" for key, _, _, revision, _ in updates))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    commands.add_parser("gallery")
    check = commands.add_parser("check")
    check.add_argument("--complete", action="store_true")
    integration = commands.add_parser("integration")
    integration.add_argument("action", choices=["pending"])
    approval = commands.add_parser("approve")
    approval.add_argument("category", choices=list(catalog()))
    approval.add_argument("--user-approval", required=True, help="Quote the user's actual approval of the complete affected preview set")
    for name in ("show", "affected", "lab", "compare", "baseline"):
        parser = commands.add_parser(name)
        parser.add_argument("category", choices=list(catalog()))
        if name == "lab":
            parser.add_argument("--case", choices=["detail", "gameplay"])
            parser.add_argument("--affected", action="store_true", help="Render dependent category previews too")
    args = p.parse_args()
    try:
        if args.command == "list":
            for key in catalog():
                value = standard(key)
                print(f'{key:18} approved r{value["approved_revision"]} | {value["title"]}')
        elif args.command == "show":
            print(json.dumps(standard(args.category), indent=2))
        elif args.command == "gallery":
            gallery()
        elif args.command == "affected":
            print("\n".join(affected(args.category)))
        elif args.command in ("lab", "baseline"):
            for key in affected(args.category) if getattr(args, "affected", False) else [args.category]:
                render(key, baseline=args.command == "baseline", selected_case=getattr(args, "case", None))
        elif args.command == "compare":
            compare(args.category)
        elif args.command == "check":
            print(f"PASS {validate(complete=args.complete)} category definitions and current references")
        elif args.command == "integration":
            changes = pending()
            print(json.dumps(changes, indent=2) if changes else "No approved revisions await integration.")
        elif args.command == "approve":
            if not args.user_approval.strip():
                raise ValueError("An actual user approval statement is required")
            approve(args.category, args.user_approval)
        return 0
    except (ValueError, OSError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
