"""Bounded offline seasonal inputs. Source names never enter the shader ABI."""
from pathlib import Path
import hashlib
import json
import os
import struct
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def assets_root():
    return Path(os.environ.get("C3X_CIV6_ASSETS", Path.home() /
        "Library/Application Support/Steam/steamapps/common/Sid Meier's Civilization VI/Civ6.app/Contents/Assets"))


def prepare(work, bodies, winter_exposure=False, winter_decals=False):
    from PIL import Image
    import numpy as np
    from Renderer.lab.studies.seasons.study import dds_image
    from Renderer.tools.asset_compiler.c3x_asset_compiler import (
        parse_civbig_header, make_dds_dx10_header, CIVBIG_HEADER_SIZE)
    evidence = {"schema": "c3x.lab.seasonal_inputs.v1", "authored_winter_bodies": [], "flower_atlas": None}
    vegetation = ROOT / "Renderer/packs/Civ5EnvironmentVegetation"
    lines = []
    for body in bodies:
        name = body["source_evidence"]
        paths = []
        if body["role"] == 2:
            suffix = name.split("/")[-1]
            m = json.loads((vegetation / f"materials/features/forest_snow_{suffix}.json").read_text())
            paths = [vegetation / m[c]["texture"] for c in ("base_color", "gloss")]
            evidence["authored_winter_bodies"].append({"body": name, "channels": [
                {"path": p.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
                "policy": "Use authored color/gloss with original geometry, normals and opacity."})
        lines.append("|".join(p.relative_to(ROOT).as_posix() for p in paths) if paths else "-")
    (work / "winter-materials.txt").write_text("\n".join(lines) + "\n")
    mask_paths = ["-"] * len(bodies)
    if winter_exposure:
        import shutil
        import subprocess
        cache = HERE / "assets/winter-exposure"
        inputs, identities = [], {}
        for index, body in enumerate(bodies):
            if body["role"] != 1:
                continue
            suffix = body["source_evidence"].split("/")[-1]
            mesh = vegetation / f"meshes/features/forest_{suffix}.json"
            material_path = vegetation / f"materials/features/forest_{suffix}.json"
            material = json.loads(material_path.read_text())
            opacity = vegetation / material["opacity"]["texture"] if material.get("opacity") else None
            for path in (mesh, material_path, opacity):
                if path:
                    identities[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            inputs.append({"body":index, "mesh":str(mesh), "opacity_source":opacity})
        generator = HERE / "winter_exposure.py"
        identities[generator.relative_to(ROOT).as_posix()] = hashlib.sha256(generator.read_bytes()).hexdigest()
        manifest_path = cache / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
        valid = manifest and manifest["inputs"] == identities and all((cache / r["path"]).is_file() and
            hashlib.sha256((cache / r["path"]).read_bytes()).hexdigest() == r["sha256"] for r in manifest["masks"])
        if not valid:
            for row in inputs:
                opacity = row.pop("opacity_source")
                row["opacity"] = None
                if opacity:
                    image = dds_image(opacity.read_bytes());image.thumbnail((512,512))
                    png = work / f"opacity-{row['body']:02d}.png";image.save(png)
                    row["opacity"] = str(png)
            specification = work / "winter-exposure-inputs.json"
            specification.write_text(json.dumps(inputs))
            generated = work / "winter-exposure"
            blender = os.environ.get("C3X_BLENDER", "/Applications/Blender.app/Contents/MacOS/Blender")
            subprocess.run([blender, "--background", "--factory-startup", "--python-exit-code", "1", "--python", str(generator),
                "--", "--inputs", str(specification), "--output", str(generated)], check=True)
            manifest = json.loads((generated / "bake.json").read_text())
            manifest.update(schema="c3x.lab.winter_exposure.v1", inputs=identities,
                policy="Reproducible opacity-aware BVH snow exposure masks for unchanged source meshes; overlapping UV exposure is an averaged C3X approximation.")
            for row in manifest["masks"]:
                row["sha256"] = hashlib.sha256((generated / row["path"]).read_bytes()).hexdigest()
            cache.mkdir(parents=True,exist_ok=True)
            for row in manifest["masks"]:
                shutil.copyfile(generated / row["path"], cache / row["path"])
            manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
        for row in manifest["masks"]:
            mask_paths[row["body"]] = (cache / row["path"]).relative_to(ROOT).as_posix()
        evidence["winter_exposure"] = {"manifest":manifest_path.relative_to(ROOT).as_posix(),
            "blender_version":manifest["blender_version"], "inputs":identities,
            "masks":[dict(row,path=(cache / row["path"]).relative_to(ROOT).as_posix()) for row in manifest["masks"]],
            "policy":manifest["policy"]}
    (work / "winter-exposure.txt").write_text("\n".join(mask_paths)+"\n")
    (work / "winter-decals.txt").write_text("0\n")
    if winter_decals:
        pack = HERE / "assets/snow-decals"
        manifest = json.loads((pack / "manifest.json").read_text())
        payload = bytearray(b"C3XSND1\0" + struct.pack("<I",len(manifest["assets"])))
        paths = [pack / "manifest.json"]
        channel_paths = None
        for asset in manifest["assets"].values():
            path = pack / asset["decal"];descriptor = json.loads(path.read_text());paths.append(path)
            channels = [pack / descriptor["channels"][c]["texture"] for c in ("base_color","height","specular")]
            if channel_paths is not None and channels != channel_paths:
                raise ValueError("Bounded snow-decal adapter requires one shared material")
            channel_paths = channels
            vertices = descriptor["mesh"]["vertices"];indices = descriptor["mesh"]["indices"]
            payload.extend(struct.pack("<I",len(indices)))
            for index in indices:
                vertex=vertices[index];payload.extend(struct.pack("<4f",*vertex["position"],*vertex["uv0"]))
        paths.extend(channel_paths)
        (work / "winter-decals.bin").write_bytes(payload)
        (work / "winter-decals.txt").write_text("1\n"+"\n".join(p.relative_to(ROOT).as_posix() for p in channel_paths)+"\n")
        evidence["winter_decals"] = {"variants":len(manifest["assets"]),
            "inputs":[{"path":p.relative_to(ROOT).as_posix(),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
            "policy":"Actual normalized authored meshes and UVs conformed to existing ground; C3X world-stable scatter and inferred packed slope response. No original terrain or tree changes."}
    # Isolate complete blossom heads from transparent connected components;
    # retain authored petal light/dark structure, normalize only their tint.
    source_name = "DLC/Expansion2/Platforms/Windows/BLPs/SHARED_DATA/TEXTURE_FX_Blossoms"
    cached = HERE / "assets/flowers/blossoms.source"
    cache_manifest = HERE / "assets/manifest.json"
    path = cached if cache_manifest.is_file() else assets_root() / source_name
    if cache_manifest.is_file():
        manifest = json.loads(cache_manifest.read_text())
        admitted = next(row for row in manifest["files"] if row["path"] == "flowers/blossoms.source")
        if hashlib.sha256(cached.read_bytes()).hexdigest() != admitted["sha256"]:
            raise ValueError("Preserved blossom source failed its integrity check")
    atlas = Image.new("RGBA", (128, 32))
    enabled = path.is_file()
    if enabled:
        raw = path.read_bytes();info = parse_civbig_header(raw)
        im = dds_image(make_dds_dx10_header(info) + raw[CIVBIG_HEADER_SIZE:CIVBIG_HEADER_SIZE+info["payload_bytes"]])
        a = np.array(im);occupied = a[:,:,3] > 40;seen = np.zeros(occupied.shape, bool);components = []
        for y,x in zip(*np.where(occupied)):
            if seen[y,x]:continue
            stack=[(int(x),int(y))];points=[];seen[y,x]=True
            while stack:
                px,py=stack.pop();points.append((px,py))
                for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                    nx,ny=px+dx,py+dy
                    if 0<=nx<im.width and 0<=ny<im.height and occupied[ny,nx] and not seen[ny,nx]:
                        seen[ny,nx]=True;stack.append((nx,ny))
            if len(points)>35:
                xs,ys=zip(*points);components.append((len(points),(max(0,min(xs)-1),max(0,min(ys)-1),min(im.width,max(xs)+2),min(im.height,max(ys)+2))))
        chosen=sorted(components,reverse=True)[:4]
        if len(chosen)!=4:raise ValueError("Blossom atlas no longer has four admitted flower heads")
        for i,(_,box) in enumerate(chosen):
            crop=im.crop(box);pixels=np.array(crop,dtype=np.float32)
            luminance=(pixels[:,:,:3]*np.array([.2126,.7152,.0722])).sum(2)
            mask=pixels[:,:,3]>40;mean=float(np.mean(luminance[mask]))
            gray=np.clip(luminance/max(mean,1)*205,0,255).astype(np.uint8)
            pixels[:,:,:3]=gray[:,:,None];crop=Image.fromarray(pixels.astype(np.uint8),"RGBA")
            crop.thumbnail((26,26),Image.Resampling.LANCZOS)
            atlas.alpha_composite(crop,(i*32+(32-crop.width)//2,(32-crop.height)//2))
        evidence["flower_atlas"]={"source": source_name, "sha256": hashlib.sha256(raw).hexdigest(),
            "input_storage": cached.relative_to(ROOT).as_posix() if path == cached else "installed source",
            "components": [list(box) for _,box in chosen], "adaptation": "Four isolated complete heads; grayscale petal shading for recipe tint; transparent padding and premultiplied-alpha filtering."}
    else:
        # Portable recipe remains usable without licensed flower assets.
        atlas=Image.new("RGBA",(128,32),(255,255,255,255))
    levels=[];level=atlas
    for i in range(5):
        levels.append(level.tobytes());level=level.resize((max(1,level.width//2),max(1,level.height//2)),Image.Resampling.LANCZOS)
    # Existing CIVBIG helpers cover the cooked block formats. This small
    # generated atlas uses ordinary RGBA8_SRGB with a standard DX10 DDS header.
    words=[124,0x2100f,32,128,128*4,0,5,*([0]*11),32,4,
           int.from_bytes(b"DX10","little"),0,0,0,0,0,0x401008,0,0,0,0]
    header=b"DDS "+struct.pack("<31I",*words)+struct.pack("<5I",29,3,0,1,0)
    (work / "flowers.dds").write_bytes(header+b"".join(levels))
    (work / "flower-enabled.txt").write_text(str(int(enabled))+"\n")
    return evidence


def census():
    """Direct ArtDef names and texture payloads; no engine behavior inference."""
    from Renderer.tools.asset_compiler.c3x_asset_compiler import parse_civbig_header
    root = assets_root()
    entries=[]
    for path in sorted(root.rglob("*.artdef")):
        if path.name not in ("Clutter.artdef","Features.artdef","TerrainStyle.artdef","TerrainMaterials.artdef","VFX.artdef"):continue
        tree=ET.parse(path)
        names=sorted({e.get("text") for tag in ("m_EntryName","m_Name","m_ElementName") for e in tree.iter(tag)
            if e.get("text") and any(k in e.get("text").lower() for k in ("snow","blossom","flower","autumn","petal"))})
        if names:entries.append({"path": path.relative_to(root).as_posix(),"names": names})
    payloads=[]
    for path in sorted(root.rglob("TEXTURE_*")):
        if "SHARED_DATA" not in path.parts:continue
        if not any(k in path.name.lower() for k in ("snow","blossom","flower","petal")):continue
        data=path.read_bytes()
        try:info=parse_civbig_header(data)
        except ValueError:continue
        payloads.append({"path":path.relative_to(root).as_posix(),"sha256":hashlib.sha256(data).hexdigest(),
            "width":info["width"],"height":info["height"],"format":info["dxgi_format"]})
    return {"schema":"c3x.lab.seasonal_census.v1","artdefs":entries,"texture_payloads":payloads,
        "scope":"Installed Base and DLC ArtDefs named above and matching shared texture payloads; source names prove availability, not suitability or engine bindings."}


def snow_decal_evidence():
    """Read all 11 Base snow decals, their actual bindings, UVs and placements.

    No extraction or pack writes. The source descriptor decoder remains owned
    by the existing generic decal compiler, including its validation rules.
    """
    from Renderer.tools.asset_compiler.generic_decal_compiler import (
        read_artdef_group, decode_decal_descriptor, decode_decal_mesh,
        TYPE_DECAL_VECTOR, TYPE_DECAL)
    from Renderer.tools.asset_compiler.clutter_blp_extractor import (
        StaticPackage, landmark_base_model, TYPE_TEXTURE, TYPE_VERTEX_BUFFER,
        TYPE_INDEX_BUFFER, decode_buffer_entry, decode_texture_entry)
    root=assets_root();package_path="Base/Platforms/Windows/BLPs/environment/clutter.blp"
    names=[f"TER_Snow_Decal{i:02d}" for i in range(1,10)]+["TER_Snow_Decal_Dark01","TER_Snow_Decal_Dark02"]
    placements,artdef=read_artdef_group(root/"Base/ArtDefs/Clutter.artdef","CLUTTER_SNOW","Plants",{n:n for n in names})
    package=StaticPackage(root/package_path,names[0]);rows=[]
    for name,placement in zip(names,placements):
        package.select_direct_string(name)
        _,landmark,_=landmark_base_model(package)
        fields=package.pointer_fields(landmark,TYPE_DECAL_VECTOR)
        if len(fields)!=1:raise ValueError("Snow source no longer has one decal vector")
        descriptors=package.pointer_fields(fields[0][1],TYPE_DECAL)
        if len(descriptors)!=1:raise ValueError("Snow source decal layout changed")
        pointer=descriptors[0][1];count=package.allocations[pointer-1]["element_count"]
        textures=package.unique_allocation(TYPE_TEXTURE)
        vertices=package.unique_allocation(TYPE_VERTEX_BUFFER)
        indices=package.unique_allocation(TYPE_INDEX_BUFFER)
        samples=[]
        for i in range(count):
            raw=package.array_element(pointer,i)
            descriptor=decode_decal_descriptor(raw,lambda ti:decode_texture_entry(package,textures,ti),12)
            bi,=struct.unpack_from("<I",raw,0x3c)
            vb=decode_buffer_entry(package,vertices,bi,True);ib=decode_buffer_entry(package,indices,bi,False)
            mesh,_=decode_decal_mesh(raw,descriptor["footprint_bounds"],package.big_data(vb["offset"],vb["bytes"]),
                package.big_data(ib["offset"],ib["bytes"]),vb["count"],ib["count"])
            uv=[v["uv0"] for v in mesh["vertices"]]
            samples.append({"footprint":descriptor["footprint_bounds"],"content":descriptor["content_bounds"],
                "uv_bounds":[min(p[0] for p in uv),min(p[1] for p in uv),max(p[0] for p in uv),max(p[1] for p in uv)],
                "channels":{role:{"name":entry["name"],"class":entry["class"]} for role,entry in descriptor["textures"].items()},
                "vertices":len(uv),"indices":len(mesh["indices"])})
        rows.append({"source_asset":name,"placement":placement,"descriptors":samples})
    return {"schema":"c3x.lab.snow_decal_evidence.v1","package":package_path,"artdef":artdef,"assets":rows,
        "interpretation":"Confirmed descriptors, UVs and ArtDef placement parameters; original engine blending is not inferred."}
