"""Build review sheets from verified production captures, never fixed references.

Run with a Pillow-enabled Python: python3 -m Renderer.lab.shadow_report
"""
from PIL import Image, ImageDraw
from Renderer import renderer


def main():
    root = renderer.LAB / "out/shadows"
    renderer.require_prepared(["shadows"])
    renderer.require_current_candidate()
    current = renderer.read(root / "render.json")
    if current["input_signature"] != renderer.category_signatures()["shadows"]:
        raise ValueError("Rerender shadows before making the review sheets")

    def verified(records):
        images = {}
        for entry in records["outputs"]:
            path = renderer.local(entry["image"])
            if renderer.checksum(path) != entry["sha256"]:
                raise ValueError("Changed capture: " + entry["image"])
            images[entry["case"], entry["hour"], entry["zoom"]] = Image.open(path).convert("RGB")
        return images

    images = verified(current)
    phases = [(12, "12 pm / West"), (18, "6 pm / South"),
              (0, "Midnight / East"), (6, "6 am / North")]
    outputs = []
    for case in ("detail", "gameplay"):
        for zoom in (128, 64):
            w, h = images[case, 12, zoom].size
            sheet = Image.new("RGB", (w * 2, (h + 28) * 2), "#20262a")
            draw = ImageDraw.Draw(sheet)
            for index, (hour, label) in enumerate(phases):
                x, y = index % 2 * w, index // 2 * (h + 28)
                sheet.paste(images[case, hour, zoom], (x, y + 28))
                draw.text((x + 12, y + 8), label + f" / {case} / tile {zoom}", fill="white")
            path = root / f"daynight-{case}-z{zoom}.png"
            sheet.save(path)
            outputs.append(renderer.relative(path))

    before_path = root / "before/render.json"
    if before_path.exists():
        before = verified(renderer.read(before_path))
        w, h = images["detail", 12, 128].size
        sheet = Image.new("RGB", (w * 2, (h + 28) * 2), "#20262a")
        draw = ImageDraw.Draw(sheet)
        for row, case in enumerate(("detail", "gameplay")):
            y = row * (h + 28)
            for column, (source, label) in enumerate(((before, "Before shared refactor"), (images, "Current candidate"))):
                x = column * w
                sheet.paste(source[case, 12, 128], (x, y + 28))
                draw.text((x + 12, y + 8), label + " / noon / " + case, fill="white")
        path = root / "before-after-noon.png"
        sheet.save(path)
        outputs.append(renderer.relative(path))
    renderer.write(root / "review.json", {"input_signature": current["input_signature"],
                   "dll_sha256": current["outputs"][0]["dll_sha256"], "images": outputs,
                   "visual_acceptance": "pending", "staged": False})
    print("\n".join(outputs))


if __name__ == "__main__":
    main()
