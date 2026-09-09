"""Classify diagnostic region misses; fingerprints are evidence, never cache keys."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

COMPONENTS = ("context", "draw", "lights", "shadow", "reflected_draw",
              "reflected_lights", "reflected_shadow")


def analyze(lines, samples=100):
    seen = {}
    frames = []
    pending = []
    for line in lines:
        if "stage=render-region-dependencies " in line:
            fields = {k: int(v) for k, v in re.findall(r"(\w+)=(-?\d+)", line)}
            pending.append(fields)
        elif "stage=render-region-cache " in line:
            counters = {k: int(v) for k, v in re.findall(r"(\w+)=(-?\d+)", line)}
            if not pending:
                continue  # No region submission in an unchanged view.
            if len(pending) != counters["hits"] + counters["misses"]:
                raise ValueError("Incomplete diagnostic region coverage")
            if sum(row["hit"] for row in pending) != counters["hits"]:
                raise ValueError("Diagnostic hit count differs from cache counter")
            causes, changes = Counter(), Counter()
            for row in pending:
                if row["parts"] not in (4, 7):
                    raise ValueError("Unknown region component recipe")
                coordinate = row["x"], row["y"]
                signature = tuple(row[name] for name in COMPONENTS)
                previous = seen.get(coordinate)
                if not row["hit"]:
                    if previous is None:
                        causes["first_observed_region"] += 1
                    elif signature == previous:
                        causes["unchanged_fingerprints"] += 1
                    else:
                        causes["changed_dependencies"] += 1
                        for name, before, after in zip(COMPONENTS, previous, signature):
                            if before != after:
                                changes[name] += 1
                seen[coordinate] = signature
            frames.append({"hits": counters["hits"], "misses": counters["misses"],
                           "causes": dict(causes), "changed_components": dict(changes)})
            pending = []
    if pending or len(frames) < samples:
        raise ValueError("Incomplete diagnostic frames")
    selected = frames[-samples:]
    causes, changes = Counter(), Counter()
    for frame in selected:
        causes.update(frame["causes"])
        changes.update(frame["changed_components"])
    return {"scope": "Last completed diagnostic frames; prior trace frames seed observed regions. "
                     "Coordinates are unwrapped occurrences. Component hashes do not prove equality. "
                     "Changed-component counts overlap; no latency claim.",
            "samples": samples, "hits": sum(f["hits"] for f in selected),
            "misses": sum(f["misses"] for f in selected), "causes": dict(causes),
            "changed_components": dict(changes), "frames": selected}


def analyze_pixels(lines, folder, samples=100):
    """Compare each miss with an earlier fully covered image of that region.

    This is an offline fixed-clock observation, not permission to weaken keys.
    Warmup images are unavailable; report these comparison gaps explicitly.
    """
    from PIL import Image, ImageChops
    frames, pending = [], []
    for line in lines:
        if "stage=render-region-dependencies " in line:
            pending.append({k: int(v) for k,v in re.findall(r"(\w+)=(-?\d+)", line)})
        elif "stage=render-region-cache " in line and pending:
            frames.append(pending);pending=[]
    if pending or len(frames)<samples:
        raise ValueError("Incomplete diagnostic frames")
    previous, counts, changed = {}, Counter(), []
    for step, rows in enumerate(frames[-samples:]):
        with Image.open(folder / f"zoom.bmp.resident{step}.bmp") as source:
            image = source.convert("RGB")
        for row in rows:
            coordinate = row["x"], row["y"]
            box = tuple(row[k] for k in ("left", "top", "right", "bottom"))
            current = image.crop(box)
            if not row["hit"]:
                old = previous.get(coordinate)
                if old is None:
                    counts["no_prior_full_region_image"] += 1
                else:
                    local = (box[0]-row["screen_x"], box[1]-row["screen_y"],
                             box[2]-row["screen_x"], box[3]-row["screen_y"])
                    difference = ImageChops.difference(current, old.crop(local))
                    if difference.getbbox() is None:
                        counts["identical_visible_pixels"] += 1
                    else:
                        counts["changed_visible_pixels"] += 1
                        red, green, blue = difference.split()
                        unchanged = ImageChops.lighter(ImageChops.lighter(red, green), blue).histogram()[0]
                        changed.append({"step": step, "x": coordinate[0], "y": coordinate[1],
                                        "changed_pixels": difference.width*difference.height-unchanged,
                                        "maximum_channel_difference": max(high for low,high in difference.getextrema())})
            if box == (row["screen_x"], row["screen_y"], row["screen_x"]+128, row["screen_y"]+128):
                previous[coordinate] = current
    return {"scope": "Compare misses to earlier fully visible regions from this fixed-clock sweep. "
                     "Partial current regions compare only visible pixels; no claim about guard pixels. "
                     "Image equality here cannot establish a generally safe invalidation rule.",
            "counts": dict(counts), "changes": changed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--images", type=Path, help="Also compare previously fully visible region pixels")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("samples must be positive")
    with args.log.open(encoding="utf-8") as stream:
        report = analyze(stream, args.samples)
    if args.images:
        with args.log.open(encoding="utf-8") as stream:
            report["pixel_observations"] = analyze_pixels(stream, args.images, args.samples)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("frames", "pixel_observations")}))
    if args.images:
        print(json.dumps(report["pixel_observations"]["counts"]))


if __name__ == "__main__":
    main()
