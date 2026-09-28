"""Review sampled game-window frames around a recorded movement or combat event.

Requires Pillow. Output is visual evidence, not a live-game FPS measurement.
"""
import argparse
import json
from pathlib import Path
import re

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--kind", choices=("movement", "combat"), required=True)
    parser.add_argument("--event", type=int, default=1, help="One-based event occurrence")
    parser.add_argument("--crop", type=int, nargs=4, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"))
    parser.add_argument("--times", type=float, nargs="+")
    args = parser.parse_args()
    folder = args.capture
    log = (folder / "renderer.log").read_text(errors="replace")
    marker = "unit-motion-admitted" if args.kind == "movement" else "scripted-combat-start"
    events = list(re.finditer(r"^.*stage=" + marker + r"\b.*$", log, re.MULTILINE))
    if not 1 <= args.event <= len(events):
        parser.error(f"Requested event {args.event}; found {len(events)} {marker} events")
    event = events[args.event - 1]
    clock = re.search(r"\bqpc=(\d+)", event.group())
    basis = "event QPC"
    if clock is None:
        clock = re.search(r"\bqpc=(\d+)", log[event.end():])
        basis = "first renderer QPC following event marker"
    if clock is None:
        parser.error("No renderer QPC near the event")
    origin = int(clock[1])
    metadata = json.loads((folder / "window/started.json").read_text())
    frequency = int(metadata["qpc_frequency"])
    if frequency <= 0:
        parser.error("Invalid recorded QPC frequency")
    rows = [json.loads(line) for line in (folder / "window/timeline.jsonl").read_text().splitlines()]
    frames = [row for row in rows if "frame" in row and "arrival_qpc" in row]
    if not frames:
        parser.error("No captured window frames")
    times = args.times or ([-.1, .1, .3, .5, .7, .9, 1.1, 1.3, 1.5, 1.7, 2.1, 2.5, 3, 4, 5]
                          if args.kind == "movement" else
                          [-.1, .2, .4, .6, .8, 1.1, 1.4, 1.8, 2.1, 2.4, 2.8, 3.2,
                           3.8, 4.4, 5, 5.6, 6.2, 6.6, 7, 7.5, 8, 8.5, 9, 10])
    crop = tuple(args.crop) if args.crop else (0, 0, frames[0]["width"], frames[0]["height"])
    width, height = crop[2] - crop[0], crop[3] - crop[1]
    if width <= 0 or height <= 0:
        parser.error("Crop must have positive dimensions")
    columns = 3 if args.kind == "movement" else 4
    scale = min(1., 460. / width)
    width, height = round(width * scale), round(height * scale)
    sheet = Image.new("RGB", (columns * width, ((len(times) + columns - 1) // columns) * (height + 20)))
    painter = ImageDraw.Draw(sheet)
    selected = []
    for index, seconds in enumerate(times):
        row = min(frames, key=lambda r: abs(r["arrival_qpc"] - origin - seconds * frequency))
        source = folder / "window" / f"window-{row['frame']:06d}.jpg"
        with Image.open(source) as image:
            tile = image.crop(crop).resize((width, height), Image.Resampling.LANCZOS)
        x, y = index % columns * width, index // columns * (height + 20)
        actual = (row["arrival_qpc"] - origin) / frequency
        sheet.paste(tile, (x, y + 20))
        painter.text((x + 5, y + 3), f"{actual:+.3f}s", fill="white")
        selected.append({"frame": row["frame"], "requested_seconds": seconds, "actual_seconds": actual})
    output = folder / f"{args.kind}-contact"
    sheet.save(output.with_suffix(".jpg"))
    output.with_suffix(".json").write_text(json.dumps({"origin_qpc": origin, "origin_basis": basis,
        "qpc_frequency": frequency, "crop": crop, "frames": selected,
        "scope": "Sampled compositor frames; does not measure FPS or prove every animation frame"}, indent=2) + "\n")
    print(output.with_suffix(".jpg"))


if __name__ == "__main__":
    main()
