"""Prepare Q8's bounded matched coastal comparison from published owner inputs.

No art or system implementation is authored here. Terrain is Q0's verified
export; augmentation is a small, explicit Lab placement recipe.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
V2 = ROOT / "Renderer/terrain_lab/v2"
sys.path.insert(0, str(V2 / "app"))
import real_map
from cache import file_hash


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def rel(path):
    return path.relative_to(ROOT).as_posix()


def prepare(candidate):
    if not candidate.replace("-", "").isalnum():
        raise ValueError("Use an alphanumeric version name")
    out = V2 / "fixtures/beauty" / candidate
    if out.exists():
        raise ValueError("Use a new version: existing checkpoint inputs are immutable")
    real_map.export("mixed", out, "Q8-beauty", True, halo=2)
    f = json.loads((out / "fixture.json").read_text())
    overlay = json.loads((out / "augmentation.json").read_text())
    overlay.update(owner="Q8-beauty", profile="q8_coastal_context_v1", seed=8)
    # Keep the single Q0-selected legal city anchor; thin the all-edges road
    # inventory to one connected local route with a short branch.
    route_pairs = {((0, 3), (1, 3)), ((1, 3), (2, 3)),
                   ((2, 3), (3, 3)), ((3, 2), (3, 3))}
    overlay["routes"] = [r for r in overlay["routes"]
                         if (tuple(r["from"]), tuple(r["to"])) in route_pairs]
    overlay["assumptions"] = [
        "One ancient town and a connected land road to its surroundings; no captured city state.",
        "Small coastal context, not a complete developed-gameplay or Q8 acceptance fixture.",
        "Frozen source city and road presentation retained until Q7/Q5 compatible modules arrive.",
    ]
    save(out / "augmentation.json", overlay)
    road_path = ROOT / f["scenarios"]["roads"]
    header = road_path.read_text().splitlines()[0].split(",")
    header[3] = str(len(overlay["routes"]))
    lines = [",".join(map(str, [*r["from"], *r["to"], 0, 1, 0, 0]))
             for r in overlay["routes"]]
    road_path.write_text(",".join(header) + "\n" + "\n".join(lines) + "\n")
    provenance = f["real_map"]
    provenance["overlay_sha256"] = file_hash(out / "augmentation.json")
    provenance["scenario_hashes"] = {
        k: file_hash(ROOT / p) for k, p in f["scenarios"].items()}
    f["viewport"] = [592, 376]
    f["references"] = ["civ6.sea_and_shore", "civ3.real_gameplay_layout"]
    f["settings"].update(samples=4, anisotropy=8, mip_bias=0,
        postprocess={"shader": "Renderer/terrain_lab/v2/shaders/sampling/linear_reconstruct.hlsl",
                     "owner": "Q1-sampling", "contract": 2})
    inputs = {
        "before_module": V2 / "fixtures/lighting/real-mixed/linear.module.json",
        "after_module": V2 / "systems/hydrology/source-linear.module.json",
        "shadow_module": V2 / "fixtures/lighting/real-mixed/world.module.json",
    }
    modules = {k: json.loads(p.read_text()) for k, p in inputs.items()}
    for label in ("before", "after"):
        module = copy.deepcopy(modules[label + "_module"])
        module.update(owner="Q8-beauty", id="q8-coastal-" + label)
        if label == "after":
            # Composition wrapper: no changed material or lighting functions.
            shader = out / "combined.hlsl"
            source = "#define Q6_WORLD_SHADOWS 1\n#include \"../../../shaders/hydrology/scene_linear.hlsl\"\n"
            shader.write_text(source)
            module["shader"] = rel(shader)
            module["packet_postprocessor"] = modules["shadow_module"]["packet_postprocessor"]
            header = out / "owner_hooks.h"
            header.write_text('#include "../../../systems/hydrology/terrain_consumer.h"\n'
                              '#include "../../../systems/hydrology/scene_adapter.h"\n')
            for hook in ("terrain_hooks", "hydrology_hooks"):
                module[hook]["header"] = rel(header)
        else:
            shader = out / "before.hlsl"
            shader.write_text('#include "../../../shaders/lighting/generated/scene_linear_v1.hlsl"\n')
            module["shader"] = rel(shader)
        save(out / (label + ".module.json"), module)
        view = copy.deepcopy(f)
        view.update(id="q8-coastal-" + label, modules=[rel(out / (label + ".module.json"))])
        save(out / (label + ".fixture.json"), view)
        real_map.validate_provenance(view)
    save(out / "RECIPE.json", {
        "schema": "c3x.q8.gameplay_checkpoint.v1", "accepted": False,
        "owner": "Q8-beauty", "region": provenance["region"],
        "source_sha256": provenance["source_sha256"],
        "placement": overlay, "fixed_settings": f["settings"],
        "comparison": "Matched input A/B: prior source linear scene vs Q2/Q3 materials/shore plus Q6 mesh shadows. Not an isolated-variable causal test.",
        "pending": ["Q7 new city module", "Q5 route candidate", "Q4 source cliff/vegetation clearance",
                    "filled larger gameplay viewport and label envelopes", "phase/zoom/scroll and heldout acceptance"],
        "consumed_module_hashes": {rel(p): file_hash(p) for p in inputs.values()},
        "reference_hashes": {rel(ROOT / ("Renderer/canonical/" + name)):
            file_hash(ROOT / ("Renderer/canonical/" + name))
            for name in ("sea_and_shore.png", "civ3_real_example.jpg")},
    })
    print(rel(out))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("candidate")
    prepare(p.parse_args().candidate)
