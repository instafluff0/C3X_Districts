"""Generate focused review images and record native deterministic/control checks."""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import numpy as np
from PIL import Image, ImageDraw
from Renderer.lab.studies.waves.study import OUT, ROOT, SNAPSHOT, render, write
from Renderer.tools.asset_compiler.wave_blp_extractor import digest


def pixels(path):
    return np.array(Image.open(path).convert("RGB"))


def main():
    earlier=OUT/"native/gameplay-waves-t8-h12-z128/frame.bmp"
    previous=pixels(earlier)
    print("Checking absolute-time replay",flush=True)
    repeated=render("gameplay",8)
    replay=np.array_equal(previous,pixels(repeated))
    if not replay:raise ValueError("Repeated native phase changed pixels")
    print("Rendering lowland, reduced zoom, close and day/night samples",flush=True)
    lowland=render("lowland",8)
    reduced=render("gameplay",8,zoom=64)
    close=render("lowland",8,zoom=256)
    daynight=[earlier]
    for hour in (18,0,6):daynight.append(render("gameplay",8,hour=hour))
    panel=Image.new("RGB",(1920,1328),"#14212c")
    for index,path in enumerate(daynight):
        x,y=index%2*960,index//2*664
        panel.paste(Image.open(path).convert("RGB"),(x,y+24))
        ImageDraw.Draw(panel).text((x+12,y+6),f"WAVES / hour {(12,18,0,6)[index]:02} / shared lighting",fill="white")
    panel.save(OUT/"day-night.png")
    # Verify that the study's texture repacking has no hidden wave-off effect.
    baseline=OUT/"native/gameplay-control-t0-h12-z128/frame.bmp"
    before=pixels(baseline)
    packed={}
    try:
        for name in ("TerrainNormalized","Civ5EnvironmentSkin"):
            for texture in ("large_lean1.dds","small_lean1.dds"):
                relative=Path("Renderer/packs")/name/"textures/water/surface"/texture
                target=SNAPSHOT/relative
                if target.exists():
                    packed[target]=target.read_bytes()
                    write(target,(ROOT/relative).read_bytes())
        print("Checking unmodified source-channel control",flush=True)
        control=render("gameplay",enabled=False)
        neutral=np.array_equal(before,pixels(control))
        lowland_control=render("lowland",enabled=False)
    finally:
        for target,data in packed.items():write(target,data)
    if not neutral:raise ValueError("Study channel repacking changed wave-off pixels")
    delta=np.abs(previous.astype(int)-before.astype(int))
    # Fixed diagnostic rectangles: left dry ground, right far offshore water.
    dry_unchanged=bool(np.all(delta[:,:240]==0))
    offshore_unchanged=bool(np.all(delta[:,800:]==0))
    if not dry_unchanged or not offshore_unchanged:
        raise ValueError("Surf changed dry-ground or offshore control rectangles")
    comparison=Image.new("RGB",(1920,680),"#14212c")
    comparison.paste(Image.fromarray(before),(0,40))
    comparison.paste(Image.fromarray(previous),(960,40))
    draw=ImageDraw.Draw(comparison)
    draw.text((16,12),"CURRENT WATER / waves off",fill="white")
    draw.text((976,12),"RECOVERED CRESTS + AUXILIARY FOAM / isolated native study",fill="white")
    comparison.save(OUT/"gameplay-comparison.png")
    comparison.paste(Image.open(lowland_control).convert("RGB"),(0,40))
    comparison.paste(Image.open(lowland).convert("RGB"),(960,40))
    comparison.save(OUT/"lowland-comparison.png")
    samples=[]
    for path in sorted((OUT/"native").glob("*/frame.bmp")):
        log=(path.parent/"native.log").read_text()
        samples.append(dict(case=path.parent.name,sha256=digest(Image.open(path).tobytes()),
                            zero_fallback="0 fallback" in log))
    report=dict(status="experimental; visual approval pending",native_samples=samples,
        absolute_time_replay_identical=replay,unused_slot_control_identical=neutral,
        dry_rectangle_unchanged=dry_unchanged,offshore_rectangle_unchanged=offshore_unchanged,
        changed_pixels=int(np.count_nonzero(np.max(delta,axis=2))),
        native_motion_samples=15,motion_sampling_seconds=2,motion_playback_speed=4,
        source_tests=5,shoreline_tests=144,
        limits=["synthetic native scenes, not a live game", "no live scheduling or FPS claim",
                "contour placement is diagnostic-direction only; wrapping not proved"])
    (OUT/"review.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="native_samples"},indent=2))


if __name__=="__main__":main()
