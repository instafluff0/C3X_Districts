"""One-time migration of inputs used by the accepted production city compiler.

Copies licensed derived data into the local source pack before historical Lab
directories are removed. Does not rebuild or replace the production runtime pack.
"""
from pathlib import Path
import json
import shutil

ROOT = Path(__file__).resolve().parents[4]
OLD = ROOT / "Renderer/terrain_lab/v2"
OUT = ROOT / "Renderer/packs/CityFidelitySources/current"


def read(path):
    return json.loads(path.read_text())


def main():
    if OUT.exists():
        raise ValueError("Current inputs already migrated; do not overwrite local edits")
    copies = {
        "palace-normals.json": OLD / "fixtures/beauty/city-capital-materials-r1/combined-normals.json",
        "palace-materials.json": OLD / "fixtures/beauty/city-capital-materials-r1/combined-extra.json",
        "ground-parts.json": OLD / "fixtures/beauty/city-ground-binding-r1/modern-ground-parts.json",
    }
    selected = []
    for revision, name in [(60, "asian-medieval-open"), (61, "mediterranean-ancient"),
                           (64, "asian-medieval-wooded"), (111, "american-capital-open"),
                           (112, "american-capital-wooded"), (101, "american-capital-coastal")]:
        path = next((OLD / f"fixtures/beauty/city-scene-r{revision}").glob("*/augmentation.json"))
        source = read(path)
        recipe = {key: source[key] for key in ("pool", "instances", "capital", "stage_component_counts",
                                              "vegetation_clearance", "river_exclusion") if key in source}
        item = {"name": name, "runtime_authority": f"selected-r{revision}", "recipe": recipe}
        if revision in (111, 112):
            site = "inland" if revision == 111 else "holdout"
            proof = OLD / "audits/beauty/out/city-central-capital-r2" / site
            item["surface"] = read(path.parent / "surface.json")
            item["lights"] = read(proof / "lights.json")["lights"]
            item["ground"] = read(proof / "ground/settlement.json")
        selected.append(item)
    report = read(OLD / "audits/beauty/out/city-source-expanded-r1/build.json")
    record = {
        "selected": selected,
        "ground_binding": read(OLD / "fixtures/beauty/city-ground-binding-r1/modern.json"),
        "ground_reference": read(OLD / "audits/beauty/out/city-central-capital-r2/inland/ground/settlement.json"),
        "blocks_by_pool": {pool["pool"]: [entry["asset_id"] for entry in pool["selected"]
                                          if "_Block_" in entry["entry"]] for pool in report["pools"]},
    }
    # Validate every source first, then copy; the old data remains intact.
    for path in copies.values():
        read(path)
    OUT.mkdir(parents=True)
    for name, path in copies.items():
        shutil.copyfile(path, OUT / name)
    (OUT / "recipes.json").write_text(json.dumps(record, indent=2) + "\n")
    print("Preserved current city inputs in Renderer/packs/CityFidelitySources/current")


if __name__ == "__main__":
    main()
