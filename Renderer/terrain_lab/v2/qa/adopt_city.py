"""Compose Q7's published source city with the Q8 coastal checkpoint."""
import json
from pathlib import Path
from prepare_checkpoint import ROOT, V2, rel, save
from cache import file_hash
import real_map

out = V2 / "fixtures/beauty/coastal-r02"
if out.exists():
    raise SystemExit("Existing r02 is immutable; select a new version for further changes")
out.mkdir(parents=True)
old = V2 / "fixtures/beauty/coastal-r01"
f = json.loads((old / "after.fixture.json").read_text())
q7_path = V2 / "fixtures/objects/generated/registered-mixed-linear-v1/fixture.json"
q7 = json.loads(q7_path.read_text())
assert q7["real_map"]["region"] == f["real_map"]["region"]
assert q7["viewport"] == f["viewport"]
f.update(id="q8-coastal-city-r02")
module = json.loads((old / "after.module.json").read_text())
module.pop("packet_postprocessor")
module["id"] = "q8-coastal-source-r02"
shader = out / "combined.hlsl"
shader.write_text('#define Q6_WORLD_SHADOWS 1\n'
                  '#define Q3_MATERIAL_ORIGIN_X 27.5\n'
                  '#define Q3_MATERIAL_ORIGIN_Y -14.5\n'
                  '#define Q3_MATERIAL_WRAP_WIDTH 100\n'
                  '#include "../../../shaders/hydrology/scene_linear.hlsl"\n')
module["shader"] = rel(shader)
save(out / "terrain.module.json", module)
f["modules"] = [rel(out / "terrain.module.json"), q7["modules"][1]]
f["packet_postprocessor"] = q7["packet_postprocessor"]
f["packs"]["presentation_geometry"] = q7["packs"]["presentation_geometry"]
city = out / "no-legacy-cities.csv"
city.write_bytes((ROOT / q7["scenarios"]["cities"]).read_bytes())
f["scenarios"]["cities"] = rel(city)
overlay = json.loads((ROOT / f["real_map"]["overlay"]).read_text())
overlay["assumptions"].append("r02 replaces the same-anchor legacy city with Q7's full source-parts town via a separate linear module; Q5/Q4 final candidates still pending.")
overlay["city_recipe"] = q7["packs"]["presentation_geometry"]
save(out / "augmentation.json", overlay)
f["real_map"].update(overlay=rel(out / "augmentation.json"),
    overlay_sha256=file_hash(out / "augmentation.json"))
f["real_map"]["scenario_hashes"]["cities"] = file_hash(city)
save(out / "after.fixture.json", f)
real_map.validate_provenance(f)
geometry = ROOT / f["packs"]["presentation_geometry"]
save(out / "RECIPE.json", {
    "accepted": False, "parent": rel(old / "RECIPE.json"),
    "intentional_changes": ["Replace inherited city with Q7 same-anchor source assembly",
        "Run Q6 shadow postprocessor after both terrain and city packets compose",
        "Adopt current Q3 material with authoritative raw-origin anchoring"],
    "unchanged": ["verified terrain", "city anchor", "road edges", "camera/viewport", "noon/zoom/sampling"],
    "q7_fixture_sha256": file_hash(q7_path),
    "geometry_hashes": {rel(p): file_hash(p) for p in geometry.iterdir()
                        if p.is_file() and p.suffix in (".bin", ".json")},
    "pending": ["Q4 cliffs/continuity/clearance", "Q5 route candidate", "full visual gates"],
})
print(rel(out / "after.fixture.json"))
