"""Review sampled game-window frames around a recorded movement, combat or drag.

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
    parser.add_argument("--window-source", type=Path,
                        help="Read original window evidence here without copying the full frame sequence")
    parser.add_argument("--kind", choices=("movement", "combat", "mouse"), required=True)
    parser.add_argument("--event", type=int, default=1, help="One-based event occurrence")
    parser.add_argument("--input-event", type=int,
                        help="Use this one-based mouse-events.json input instead of a renderer log event")
    parser.add_argument("--crop", type=int, nargs=4, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"))
    parser.add_argument("--times", type=float, nargs="+")
    args = parser.parse_args()
    folder = args.capture
    window = args.window_source or folder / "window"
    log = (folder / "renderer.log").read_text(errors="replace")
    marker = {"movement": "unit-motion-admitted", "combat": "scripted-combat-start",
              "mouse": r"map-click\b.*\bmode=1"}[args.kind]
    events = list(re.finditer(r"^.*stage=" + marker + r"\b.*$", log, re.MULTILINE))
    if args.input_event is None and not 1 <= args.event <= len(events):
        parser.error(f"Requested event {args.event}; found {len(events)} {marker} events")
    if args.input_event is not None:
        inputs = json.loads((folder / "mouse-events.json").read_text(encoding="utf-8-sig"))
        if not 1 <= args.input_event <= len(inputs["events"]):
            parser.error("Input event is outside the recorded sequence")
        origin = int(inputs["events"][args.input_event - 1]["qpc"])
        basis = "injected input QPC"
    else:
        event = events[args.event - 1]
        clock = re.search(r"\bqpc=(\d+)", event.group())
        basis = "event QPC"
        if clock is None:
            clock = re.search(r"\bqpc=(\d+)", log[event.end():])
            basis = "first renderer QPC following event marker"
        if clock is None:
            parser.error("No renderer QPC near the event")
        origin = int(clock[1])
    metadata = json.loads((window / "started.json").read_text())
    frequency = int(metadata["qpc_frequency"])
    if frequency <= 0:
        parser.error("Invalid recorded QPC frequency")
    if args.input_event is not None and frequency != int(inputs["qpc_frequency"]):
        parser.error("Input and window QPC frequencies differ")
    rows = [json.loads(line) for line in (window / "timeline.jsonl").read_text().splitlines()]
    frames = [row for row in rows if "frame" in row and "arrival_qpc" in row]
    if not frames:
        parser.error("No captured window frames")
    compositor_clock = all("compositor_100ns" in row for row in frames)
    def frame_seconds(row):
        # WGC SystemRelativeTime uses the QPC epoch in 100 ns units. Arrival
        # includes observer delay and can overstate visible response latency.
        return (row["compositor_100ns"] / 10_000_000 if compositor_clock else
                row["arrival_qpc"] / frequency) - origin / frequency
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
        row = min(frames, key=lambda r: abs(frame_seconds(r) - seconds))
        source = window / f"window-{row['frame']:06d}.jpg"
        with Image.open(source) as image:
            tile = image.crop(crop).resize((width, height), Image.Resampling.LANCZOS)
        x, y = index % columns * width, index // columns * (height + 20)
        actual = frame_seconds(row)
        sheet.paste(tile, (x, y + 20))
        painter.text((x + 5, y + 3), f"{actual:+.3f}s", fill="white")
        selected.append({"frame": row["frame"], "requested_seconds": seconds, "actual_seconds": actual})
    output = folder / f"{args.kind}-contact"
    sheet.save(output.with_suffix(".jpg"))
    output.with_suffix(".json").write_text(json.dumps({"origin_qpc": origin, "origin_basis": basis,
        "qpc_frequency": frequency, "crop": crop, "frames": selected,
        "frame_time_basis": "WGC compositor timestamp" if compositor_clock else "observer arrival QPC",
        "scope": "Sampled compositor frames; does not measure FPS or prove every animation frame"}, indent=2) + "\n")
    print(output.with_suffix(".jpg"))


if __name__ == "__main__":
    main()
