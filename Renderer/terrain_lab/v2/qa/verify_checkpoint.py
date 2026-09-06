"""Verify the bounded Q8 comparison evidence, not artistic acceptance."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
AUDIT = ROOT / "Renderer/terrain_lab/v2/audits/beauty"


def verify():
    reports = [json.loads((AUDIT / "out" / ("coastal-r01-" + name) /
                          "report.json").read_text()) for name in ("before", "after")]
    before, after = [r["effective"] for r in reports]
    for key in ("contract", "pack_hash", "settings", "postprocess_hash", "capabilities"):
        if before.get(key) != after.get(key):
            raise ValueError("Comparison drift: " + key)
    changed = {key for key in before["fixture"]
               if before["fixture"][key] != after["fixture"].get(key)}
    if changed != {"id", "modules"}:
        raise ValueError("Unexpected fixture changes: " + str(changed))
    for report in reports:
        if len(report["outputs"]) != 1:
            raise ValueError("Expected one bounded checkpoint output")
        out = report["outputs"][0]
        image = ROOT / out["image"]
        if hashlib.sha256(image.read_bytes()).hexdigest() != out["sha256"]:
            raise ValueError("Rendered image hash mismatch")
    newest = json.loads((AUDIT / "out/coastal-r02-after/report.json").read_text())
    current = newest["effective"]
    for key in ("settings", "postprocess_hash", "capabilities"):
        if after.get(key) != current.get(key):
            raise ValueError("City checkpoint settings drift: " + key)
    for key in ("terrain", "viewport", "tile_count"):
        if after["fixture"][key] != current["fixture"][key]:
            raise ValueError("City checkpoint context drift: " + key)
    for key, path in after["fixture"]["scenarios"].items():
        if key != "cities" and path != current["fixture"]["scenarios"][key]:
            raise ValueError("Unexpected scenario change: " + key)
    frame = newest["outputs"][0]
    if hashlib.sha256((ROOT / frame["image"]).read_bytes()).hexdigest() != frame["sha256"]:
        raise ValueError("City checkpoint image hash mismatch")
    return {"matched_inputs": True, "image_hashes_verified": True,
            "render_identities": [r["render_identity"] for r in reports],
            "city_adoption_render_identity": newest["render_identity"],
            "beauty_accepted": False, "review": "Renderer/terrain_lab/v2/audits/beauty/REVIEW.md"}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
