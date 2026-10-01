"""Small source-derived tissue masks and crown fields; original bodies stay intact."""
import hashlib
import json
import math
from pathlib import Path
import struct


def prepare(work, bodies):
    import numpy as np
    from PIL import Image, ImageDraw
    from Renderer.lab.studies.seasons.study import dds_image
    root = Path(__file__).resolve().parents[4]
    source = root / "Renderer/packs/BeautyStudies/beauty_objects.bin"
    raw = source.read_bytes(); at = 8

    def take(fmt):
        nonlocal at
        result = struct.unpack_from("<" + fmt, raw, at)
        at += struct.calcsize("<" + fmt)
        return result

    def string():
        nonlocal at
        n, = take("I"); result = raw[at:at+n].decode(); at += n
        return result

    version, nm, no, nr = take("4I")
    if version != 3:
        raise ValueError("Selected source body version")
    materials = [([string() for _ in range(7)], take("2I")) for _ in range(nm)]
    rows = []
    for _ in range(no):
        name = string(); kind, material, n = take("3I")
        vertices = np.array([take("8f") for _ in range(n)])
        if kind == 1:
            rows.append((name, material, vertices))
    if len(rows) != 22:
        raise ValueError("Original forest association")
    payload = bytearray(b"C3XCRN1\0" + struct.pack("<I", len(bodies)))
    descriptors = []; evidence = []
    size = 256
    for index, body in enumerate(bodies):
        if index >= len(rows) or body["role"] != 1:
            payload += struct.pack("<I", 0); descriptors.append("- 0 0 0")
            continue
        name, material, vertices = rows[index]
        if name != body["source_evidence"]:
            raise ValueError("Crown metadata ordering")
        color_path = root / materials[material][0][0]
        color = np.array(dds_image(color_path.read_bytes()).convert("RGB"), dtype=float)/255
        linear = np.where(color <= .04045, color/12.92, ((color+.055)/1.055)**2.4)
        triangles = vertices.reshape(-1, 3, 8)
        shrub = "shrub" in name
        # Every inspected leafy family has four low-normal trunk fan faces per
        # stem and upward crown fans. This is offline, confirmed family evidence;
        # the shader receives generic tissue data, never these source names.
        tissue = np.ones(len(triangles), dtype=bool) if shrub else triangles[:, :, 5].mean(1) > .55
        stem_tops = []
        for triangle in triangles[~tissue]:
            top = triangle[np.argmax(triangle[:, 2]), :3]
            if not any(np.linalg.norm(top[:2]-p[:2]) < .005 for p in stem_tops):
                stem_tops.append(top)
        if not stem_tops:
            stem_tops = [np.median(vertices[:, :3], axis=0)]
        stems = np.array(stem_tops)
        centers = triangles[:, :, :3].mean(1)
        assignment = ((centers[:, None, :2]-stems[None, :, :2])**2).sum(2).argmin(1)
        field = np.zeros((len(vertices), 4), dtype=np.float32)
        field[:, 3] = np.repeat(tissue.astype(float), 3)
        for lobe in range(len(stems)):
            selected = tissue & (assignment == lobe)
            if not selected.any():
                continue
            points = triangles[selected, :, :3].reshape(-1, 3)
            lo, hi = points.min(0), points.max(0)
            center = (lo+hi)*.5; radius = np.maximum((hi-lo)*.5, .015)
            indices = np.flatnonzero(np.repeat(selected, 3))
            normal = (vertices[indices, :3]-center)/radius
            normal[:, :2] *= .90; normal[:, 2] = normal[:, 2]*.65+.25
            normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], .001)
            field[indices, :3] = normal
        mask = Image.new("L", (size, size)); wood = Image.new("L", (size, size))
        leaf_draw, wood_draw = ImageDraw.Draw(mask), ImageDraw.Draw(wood)
        for triangle, leaf in zip(triangles, tissue):
            uv = triangle[:, 6:8]
            # UV bounds can straddle or lie outside the unit cell. Rasterize
            # every intersecting periodic copy, matching the working sampler.
            for oy in range(math.floor(uv[:, 1].min()), math.floor(uv[:, 1].max())+1):
                for ox in range(math.floor(uv[:, 0].min()), math.floor(uv[:, 0].max())+1):
                    points = [((u-ox)*size, (v-oy)*size) for u, v in uv]
                    (leaf_draw if leaf else wood_draw).polygon(points, fill=255)
        # Exact per-face tissue disambiguates shared UVs. Atlas chroma only
        # protects woody texels inside a leaf region, especially the shrubs.
        small = np.array(Image.fromarray((color*255).astype(np.uint8)).resize((size,size)), dtype=float)/255
        green = (small[:, :, 1]-np.maximum(small[:, :, 0], small[:, :, 2]))/np.maximum(.01,small.max(2))
        amount = np.clip((green+.025)/.10,0,1)
        occupied = np.array(mask)/255
        mask = Image.fromarray(np.round(occupied*amount*255).astype(np.uint8))
        full_mask = np.array(mask.resize((color.shape[1],color.shape[0]),Image.Resampling.NEAREST)) > 120
        samples = linear[full_mask] @ np.array([.2126,.7152,.0722])
        if len(samples) < 100:
            raise ValueError("Nonempty source leaf mask required")
        values = np.maximum(np.percentile(samples, [10,50,90]), .002)
        levels = []; level = mask
        for _ in range(9):
            levels.append(level.tobytes())
            level = level.resize((max(1,level.width//2),max(1,level.height//2)),Image.Resampling.BOX)
        words = [124,0x2100f,size,size,size,0,9,*([0]*11),32,4,
                 int.from_bytes(b"DX10","little"),0,0,0,0,0,0x401008,0,0,0,0]
        path = work / f"autumn-tissue-{index:02d}.dds"
        path.write_bytes(b"DDS "+struct.pack("<31I",*words)+struct.pack("<5I",61,3,0,1,0)+b"".join(levels))
        payload += struct.pack("<I", len(field))+field.astype("<f4").tobytes()
        descriptors.append(path.name+" "+" ".join(str(v) for v in values))
        evidence.append({"body":name,"source_color":color_path.relative_to(root).as_posix(),
                         "source_sha256":hashlib.sha256(color_path.read_bytes()).hexdigest(),
                         "leaf_triangles":int(tissue.sum()),"wood_triangles":int((~tissue).sum()),
                         "crown_lobes":len(stems),"linear_leaf_quantiles":values.tolist(),
                         "mask_bytes":path.stat().st_size})
    (work / "autumn-crowns.bin").write_bytes(payload)
    (work / "autumn-materials.txt").write_text("\n".join(descriptors)+"\n")
    return {"schema":"c3x.lab.autumn_inputs.v1","source_sha256":hashlib.sha256(raw).hexdigest(),
            "bodies":evidence,"crown_field_bytes":len(payload),
            "policy":"Original body positions, normals, UV, opacity, scale and placements unchanged. Small temporary tissue masks and per-vertex irradiance fields only."}
