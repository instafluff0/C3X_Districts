#!/usr/bin/env python3
"""Manifest-driven Lab v2 runner. Run from any directory; ordinary work is Mac-only."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
from cache import Cache, canonical, digest, file_hash

ROOT = Path(__file__).resolve().parents[4]
V2 = ROOT / "Renderer/terrain_lab/v2"
APP = V2 / "app"
CONTRACT = V2 / "contracts/platform_v1.json"
DEFAULT = V2 / "tests/platform/micro.fixture.json"


def run(args, **kw):
    result = subprocess.run(
        [str(x) for x in args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        **kw,
    )
    if result.returncode:
        raise ValueError(f"command failed: {Path(str(args[0])).name}\n{result.stdout}")
    return result.stdout


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def local(value):
    p = (ROOT / value).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError("path escapes repository")
    return p


def read_json(path, schema):
    v = json.loads(Path(path).read_text())
    if not isinstance(v, dict) or v.get("schema") != schema:
        raise ValueError(f"contract drift: {Path(path).name}")
    return v


def package(track):
    return read_json(
        V2 / f"campaigns/Q1/work_packages/{track}.json",
        "c3x.renderer_lab_v2_work_package.v0",
    )


def owned(path, track):
    p = relative(path)
    paths = package(track)["owns_paths"]  # Still reject unknown track identities.
    campaign = json.loads((V2/'campaigns/Q1/campaign.json').read_text())
    if campaign['execution_policy']['mode'] == 'single_lead' and local(p).is_relative_to(V2):
        return
    if not any(p.startswith(x) for x in paths):
        raise ValueError(f"{track} does not own {p}")


def fixture(path):
    f = read_json(path, "c3x.lab_v2.fixture.v1")
    required = {
        "schema",
        "id",
        "track",
        "campaign",
        "tile_count",
        "viewport",
        "terrain",
        "modules",
        "packs",
        "references",
        "isolations",
        "settings",
    }
    if not required.issubset(f) or set(f) - required - {"scenarios", "real_map", "sidecars", "packet_postprocessor"}:
        raise ValueError("fixture missing/unknown fields")
    owned(path, f["track"])
    if f["campaign"] != "Q1" or not re.fullmatch(r"[a-zA-Z0-9_-]+", f["id"]):
        raise ValueError("invalid fixture identity")
    if not isinstance(f["tile_count"], int) or not 16 <= f["tile_count"] <= 192:
        raise ValueError("invalid fixture tile count")
    if len(f["viewport"]) != 2 or any(
        type(x) != int or x < 64 or x > 8192 or x % 4 for x in f["viewport"]
    ):
        raise ValueError("invalid viewport")
    terrain = local(f["terrain"])
    owned(terrain, f["track"])
    header = terrain.read_text().splitlines()[0].split(",")
    if (
        len(header) != 9
        or int(header[3]) != f["tile_count"]
        or int(header[1]) * int(header[2]) != f["tile_count"]
    ):
        raise ValueError("fixture tile count disagrees with terrain")
    if "real_map" in f:
        from real_map import validate_provenance
        validate_provenance(f)
    if f.get("sidecars"):
        sys.path.insert(0,str(V2/'shared'))
        from scene_exchange import validate as validate_exchange
        spaces=set()
        for sidecar in f['sidecars']:
            if set(sidecar)!={'path','sha256','schema','owner'}: raise ValueError('invalid sidecar reference')
            path=local(sidecar['path']);owned(path,sidecar['owner'])
            if file_hash(path)!=sidecar['sha256']: raise ValueError('stale scene sidecar')
            data=read_json(path,sidecar['schema']);validate_exchange(data)
            if data['terrain_sha256']!=file_hash(terrain): raise ValueError('sidecar terrain revision mismatch')
            spaces.add(data['coordinate_space'])
            def resource_refs(value):
                if isinstance(value,dict):
                    if 'path' in value and 'sha256' in value:
                        if file_hash(local(value['path']))!=value['sha256']: raise ValueError('stale sidecar resource')
                    for child in value.values():resource_refs(child)
                elif isinstance(value,list):
                    for child in value:resource_refs(child)
            resource_refs(data)
        if len(spaces)>1: raise ValueError('incompatible scene coordinate spaces')
    if not f["modules"]:
        raise ValueError("fixture needs at least one module")
    modules = []
    for name in f["modules"]:
        path = local(name)
        if not path.is_file():
            raise ValueError("missing module: " + name)
        m = read_json(path, "c3x.lab_v2.module.v1")
        if m.get("contract") != 1:
            raise ValueError("module contract drift: " + name)
        owned(path, m["owner"])
        if m.get("provider") not in ("frozen_l21", "cpp_packet"):
            raise ValueError("unsupported provider: " + str(m.get("provider")))
        if m["provider"] == "cpp_packet":
            source = local(m["source"])
            owned(source, m["owner"])
            if not source.is_file() or source.suffix != ".cpp":
                raise ValueError("missing C++ packet module")
        shader = local(m["shader"])
        if m.get('coastal_rocks'):
            resource=m['coastal_rocks']
            if (set(resource)-{'path','sha256','placement_version'} or
                not {'path','sha256'}.issubset(resource) or
                resource.get('placement_version',1) not in (1,2,3,4) or
                file_hash(local(resource['path']))!=resource['sha256']):
                raise ValueError('invalid pinned coastal rock bundle')
        if m.get('hill_source'):
            resource=m['hill_source']
            if (set(resource)!={'path','sha256','height_multiplier'} or
                file_hash(local(resource['path']))!=resource['sha256'] or
                not 0 < resource['height_multiplier'] <= 4):
                raise ValueError('invalid pinned hill height resource')
        if not shader.is_file():
            raise ValueError("missing shader module: " + m["shader"])
        if not relative(shader).startswith("Renderer/terrain_lab/v2/shaders/common/"):
            owned(shader, m.get("shader_owner",m["owner"]))
        modules.append(m)
    ids = [m["id"] for m in modules]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate module identity")
    if len({file_hash(local(m["shader"])) for m in modules}) != 1 and not all(m.get("color_branch")=="q6_scene_linear_premultiplied_v1" for m in modules):
        raise ValueError(
            "mixed shader contracts require a versioned per-draw shader interface"
        )
    ordered = []
    pending = list(modules)
    while pending:
        ready = [
            m
            for m in pending
            if set(m.get("after", [])).issubset({x["id"] for x in ordered})
        ]
        if not ready:
            raise ValueError("render graph cycle or missing dependency")
        for m in ready:
            ordered.append(m)
            pending.remove(m)
    modules = ordered
    if not {"terrain", "vegetation", "decals", "relief", "shore"}.issubset(f["packs"]):
        raise ValueError("missing/unknown normalized pack mount")
    for name, p in f["packs"].items():
        if not local(p).is_dir():
            raise ValueError("missing pack mount: " + name)
    validate_settings(f["settings"])
    result=modules[0] if len(modules)==1 else dict(modules[0],provider="compose",modules=modules)
    if len(modules)>1:
        for key in ("packet_postprocessor","terrain_hooks","hydrology_hooks","placement_hooks","world_positions","hydrology_data","linear_adapter"):
            result.pop(key,None)
    if f.get("packet_postprocessor"):
        result=dict(result,packet_postprocessor=f["packet_postprocessor"])
    return f,result


def validate_settings(s):
    required = {
        "anisotropy",
        "mip_bias",
        "samples",
        "render_scale",
        "postprocess",
        "camera_offsets",
    }
    if set(s) != required:
        raise ValueError("sampling settings missing/unknown fields")
    if type(s["anisotropy"]) != int or not 1 <= s["anisotropy"] <= 16:
        raise ValueError("unsupported anisotropy")
    if type(s["samples"]) != int or s["samples"] not in (1, 2, 4, 8):
        raise ValueError("unsupported sample count")
    if type(s["render_scale"]) != int or s["render_scale"] not in (1, 2, 4):
        raise ValueError("unsupported render scale")
    if (
        not isinstance(s["mip_bias"], (int, float))
        or not math.isfinite(s["mip_bias"])
        or not -4 <= s["mip_bias"] <= 4
    ):
        raise ValueError("unsupported mip bias")
    if s["postprocess"] != "box" and not isinstance(s["postprocess"], dict):
        raise ValueError("unsupported postprocess module")
    if isinstance(s["postprocess"], dict):
        post = s["postprocess"]
        if set(post) != {"shader", "owner", "contract"} or post["contract"] not in (1, 2):
            raise ValueError("postprocess contract drift")
        path = local(post["shader"])
        owned(path, post["owner"])
        if not path.is_file():
            raise ValueError("missing postprocess shader")
    if (
        not isinstance(s["camera_offsets"], list)
        or not 1 <= len(s["camera_offsets"]) <= 64
    ):
        raise ValueError("invalid camera sequence")
    for xy in s["camera_offsets"]:
        if (
            not isinstance(xy, list)
            or len(xy) != 2
            or any(
                not isinstance(v, (int, float)) or not math.isfinite(v) or abs(v) > 4096
                for v in xy
            )
        ):
            raise ValueError("invalid camera offset")


def closure(path, active=()):
    path = Path(path).resolve()
    if path in active:
        raise ValueError("cyclic shader include")
    result = {relative(path): file_hash(path)}
    for name in re.findall(r'^\s*#include\s+"([^"]+)"', path.read_text(), re.M):
        child = (path.parent / name).resolve()
        if not child.is_file():
            raise ValueError("missing shader include: " + name)
        result.update(closure(child, active + (path,)))
    return result


def shader_source(path, active=()):
    path = Path(path).resolve()
    if path in active:
        raise ValueError("cyclic shader include")

    def include(match):
        return shader_source(path.parent / match.group(1), active + (path,))

    return re.sub(
        r'^\s*#include\s+"([^"]+)"[^\n]*', include, path.read_text(), flags=re.M
    )


def tool_libraries(env):
    return {
        str(index) + ":" + p.name: file_hash(p)
        for index, directory in enumerate(env["DYLD_LIBRARY_PATH"].split(":"))
        for p in sorted(Path(directory).glob("*.dylib"))
    }


def compile_cpp(cache, source, objc=False):
    flags = ["-std=c++17", "-O2"] + (["-fobjc-arc"] if objc else [])
    dependencies = run(["clang++", *flags, "-MM", source])
    # clang escapes spaces in repository paths. Match escaped tokens before resolving.
    words = re.findall(
        r"(?:\\.|[^\s])+", dependencies.replace("\\\n", " ").split(":", 1)[1]
    )
    deps = [Path(x.replace("\\ ", " ")) for x in words]
    identity = {
        "compiler": run(["clang++", "--version"]),
        "flags": flags,
        "dependencies": {relative(p): file_hash(p) for p in deps},
    }
    return cache.artifact(
        "cpp", identity, lambda out: run(["clang++", *flags, "-c", source, "-o", out])
    )


def executables(cache):
    scene = compile_cpp(cache, V2 / "shared/frozen_scene.cpp")
    env = compile_cpp(cache, V2 / "shared/environment_runtime.cpp")
    metal = compile_cpp(cache, V2 / "backends/metal.mm", True)

    def link(kind, objects, libs):
        result = cache.artifact(
            kind,
            {"objects": [file_hash(p) for p in objects], "libs": libs},
            lambda out: run(["clang++", *objects, *libs, "-o", out]),
        )
        result.chmod(0o755)
        return result

    return link("scene-executable", [scene, env], []), link(
        "metal-executable", [metal], ["-framework", "Metal", "-framework", "Foundation"]
    )


def tools():
    root = Path(os.environ.get("C3X_LAB_SHADER_TOOLS", str(APP / ".local")))
    glslang = root / "glslang/16.5.0/bin/glslang"
    cross = root / "spirv-cross/1.4.357.0/bin/spirv-cross"
    if not glslang.is_file() or not cross.is_file():
        raise ValueError(
            "shader tools missing; run app/bootstrap_tools.py or configure C3X_LAB_SHADER_TOOLS"
        )
    env = dict(
        os.environ,
        DYLD_LIBRARY_PATH=str(root / "glslang/16.5.0/lib")
        + ":"
        + str(root / "spirv-tools/1.4.357.0/lib"),
    )
    return glslang, cross, env


def apply_mip_bias(source, bias):
    if bias == 0:
        return source
    result = []
    start = 0
    for m in re.finditer(r"\.Sample\(", source):
        pos = m.end()
        depth = 1
        while depth and pos < len(source):
            depth += (source[pos] == "(") - (source[pos] == ")")
            pos += 1
        if depth:
            raise ValueError("malformed texture sample call")
        result.append(
            source[start : m.start()]
            + ".SampleBias("
            + source[m.end() : pos - 1]
            + ","
            + str(float(bias))
            + ")"
        )
        start = pos
    result.append(source[start:])
    return "".join(result)


def shaders(cache, path, bias, msl_version=20100):
    if msl_version not in (20100,20200):
        raise ValueError("unsupported MSL capability version")
    glslang, cross, env = tools()
    identity = {
        "closure": closure(path),
        "glslang": file_hash(glslang),
        "spirv_cross": file_hash(cross),
        "tool_libraries": tool_libraries(env),
        "mip_bias": bias,
        "msl": msl_version,
        "bindings": {"textures": 0, "samplers": 128, "constants": 130},
        "compiler_options": "auto-map, strict runtime Metal math",
    }
    outputs = {}
    for entry in ["VSMain", "VSFeature", "PSMain", "PSFeature"]:
        stage = "vert" if entry.startswith("VS") else "frag"

        def build(out):
            source = out.parent / "input.hlsl"
            # Resolve includes through the original shader directory; no source-specific behavior.
            source.write_text(apply_mip_bias(shader_source(path), bias))
            spv = out.parent / "shader.spv"
            run(
                [
                    glslang,
                    "-D",
                    "-V",
                    "-S",
                    stage,
                    "-e",
                    entry,
                    "--auto-map-bindings",
                    "--auto-map-locations",
                    "--shift-texture-binding",
                    stage,
                    "0",
                    "--shift-sampler-binding",
                    stage,
                    "128",
                    "--shift-UBO-binding",
                    stage,
                    "130",
                    "-I" + str(path.parent),
                    source,
                    "-o",
                    spv,
                ],
                env=env,
            )
            run(
                [
                    cross,
                    spv,
                    "--msl",
                    "--msl-version",
                    str(msl_version),
                    "--msl-argument-buffers",
                    "--msl-decoration-binding",
                    "--output",
                    out,
                ]
            )

        outputs[entry] = cache.artifact("shader", dict(identity, entry=entry), build)
    return outputs


def pack_identity(cache, mounts):
    hashes = {}
    for mount, name in mounts.items():
        root = local(name)
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix in (".dds", ".bin"):
                hashes[relative(p)] = file_hash(p)
    manifest = cache.artifact(
        "pack", {"files": hashes}, lambda p: p.write_bytes(canonical(hashes))
    )
    return file_hash(manifest)


def packet(cache, f, module, scene, phase, zoom, pack_hash, query=None):
    if module.get("packet_postprocessor") and not query:
        post=module["packet_postprocessor"]
        if set(post)!={"source","owner","contract"} or post["contract"]!=1:
            raise ValueError("invalid packet postprocessor contract")
        source=local(post["source"]);owned(source,post["owner"])
        if source.suffix!=".cpp":raise ValueError("packet postprocessor needs C++ source")
        obj=compile_cpp(cache,source)
        exe=cache.artifact("module-executable",{"object":file_hash(obj)},lambda out:run(["clang++",obj,"-o",out]));exe.chmod(0o755)
        base=dict(module);base.pop("packet_postprocessor")
        incoming=packet(cache,f,base,scene,phase,zoom,pack_hash)
        identity={"postprocessor_contract":1,"builder":file_hash(exe),"input":file_hash(incoming),"fixture":f,"phase":phase,"zoom":zoom}
        def process(out):
            descriptor=out.parent/"module-input.json";descriptor.write_bytes(canonical(f))
            run([exe,incoming,out,str(phase),descriptor])
            meta=Path(str(incoming)+".source.json")
            if meta.exists():shutil.copyfile(meta,Path(str(out)+".source.json"))
        return cache.artifact("geometry",identity,process)
    dependencies = []
    base_scene = scene
    if module["provider"] in ("cpp_packet", "compose"):
        source = (
            local(module["source"])
            if module["provider"] == "cpp_packet"
            else V2 / "shared/compose.cpp"
        )
        obj = compile_cpp(cache, source)
        scene = cache.artifact(
            "module-executable",
            {"object": file_hash(obj)},
            lambda out: run(["clang++", obj, "-o", out]),
        )
        scene.chmod(0o755)
    if module.get("terrain_hooks") or module.get("hydrology_hooks") or module.get("placement_hooks"):
        includes=['#include "'+str(V2/'shared/scene_hooks.h')+'"']
        bindings=[];hook_identity={}
        if module.get('continuous_normals'):
            if module['continuous_normals'] != 1:
                raise ValueError('unsupported continuous normal adapter')
            includes += ['#define LAB_V2_CONTINUOUS_NORMALS 1',
                         '#include "'+str(V2/'systems/relief/continuous_normal.h')+'"']
            hook_identity['continuous_normals'] = 1
        for family,required,optional in [('terrain_hooks',{'initialize','material_weights'},{'material_uv'}),('hydrology_hooks',{'initialize','signed_shore_distance'},{'shore_sample','coast_segment'}),('placement_hooks',{'initialize','accept_vegetation'},set())]:
            if not module.get(family):continue
            hooks=module[family];header=local(hooks['header']);owned(header,hooks.get('owner',module['owner']))
            if not (required|{'header'}).issubset(hooks) or set(hooks)-required-optional-{'header','owner'} or any(not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_:]*',hooks[k]) for k in hooks if k not in ('header','owner')):
                raise ValueError('invalid scene provider hook contract')
            includes.append('#include "'+str(header)+'"')
            bindings.extend('labv2::'+family+'.'+key+'=&'+value+';' for key,value in hooks.items() if key not in ('header','owner'))
            hook_identity[family]={'hooks':hooks,'header':file_hash(header)}
        source_text='\n'.join(includes)+'\n#define main frozen_main\n#include "'+str(V2/'shared/frozen_scene.cpp')+'"\n#undef main\nint main(int argc,char**argv){'+''.join(bindings)+'return frozen_main(argc,argv); }\n'
        wrapper=cache.artifact('provider-wrapper',hook_identity,lambda out:out.write_text(source_text))
        cpp=wrapper.with_suffix('.cpp')
        if not cpp.exists() or file_hash(cpp)!=file_hash(wrapper):shutil.copyfile(wrapper,cpp)
        obj=compile_cpp(cache,cpp);env=compile_cpp(cache,V2/'shared/environment_runtime.cpp')
        scene=cache.artifact('scene-executable',{'objects':[file_hash(obj),file_hash(env)],'libs':[]},lambda out:run(['clang++',obj,env,'-o',out]));scene.chmod(0o755)
    if module["provider"] == "compose":
        dependencies = [
            packet(cache, f, m, base_scene, phase, zoom, pack_hash)
            for m in module["modules"]
        ]
    identity = {
        "contract": file_hash(CONTRACT),
        "fixture": file_hash(local(f["terrain"])),
        "viewport": f["viewport"],
        "packs": pack_hash,
        "builder": file_hash(scene),
        "real_map": f.get("real_map"),
        "sidecars": f.get("sidecars"),
        "surface_query": file_hash(query) if query else None,
        "phase": phase,
        "zoom": zoom,
        "module": module,
        "dependencies": [file_hash(p) for p in dependencies],
        "scenarios": {
            k: file_hash(local(v)) for k, v in f.get("scenarios", {}).items()
        },
    }
    mounts = f["packs"]
    mode = {12: "noon", 18: "sunset", 0: "midnight", 6: "sunrise"}[phase]

    def build(out):
        if module["provider"] == "compose":
            run([scene, out, *dependencies])
            return
        if module["provider"] == "cpp_packet":
            descriptor = out.parent / "module-input.json"
            descriptor.write_bytes(canonical(f))
            run(
                [
                    scene,
                    out,
                    *map(str, f["viewport"]),
                    str(phase),
                    str(zoom),
                    descriptor,
                ]
            )
            return
        complete = module.get("scene") == "complete"
        args = [
            scene,
            local(mounts["terrain"]),
            "unused",
            out,
            ("beauty_complete_" + mode if complete else "beauty_lighting_" + mode)
            + ("_zoom2" if zoom == 2 and not complete else ""),
            "0.26",
            "4.0",
            "1.0",
            "72",
            "0.085",
            local(mounts["vegetation"]),
            local(mounts["decals"]),
            local(f["terrain"]),
            local(mounts["relief"]),
            local(mounts["shore"]),
        ]
        if complete:
            scenarios = f["scenarios"]
            args += [
                local(mounts["routes"]),
                local(scenarios["roads"]),
                local(mounts["bridges"]),
                local(scenarios["railroads"]),
                local(mounts["resources"]),
                local(scenarios["resources"]),
                local(mounts["cities"]),
                local(mounts["walls"]),
                local(scenarios["cities"]),
                local(mounts["improvements"]),
                local(scenarios["mines"]),
                local(mounts["improvements"]),
                local(scenarios["farms"]),
                local(mounts["objects"]),
                local(scenarios["objects"]),
                local(mounts["objects"]),
                local(scenarios["infrastructure"]),
                local(mounts["units"]),
                local(mounts["compound_units"]),
                local(scenarios["units"]),
            ]
        descriptor=out.parent/"module-input.json";descriptor.write_bytes(canonical(f))
        environment = dict(os.environ)
        environment["C3X_LAB_V2_FIXTURE_JSON"]=str(descriptor)
        environment.pop('C3X_LAB_V2_HILL_HEIGHT',None)
        environment.pop('C3X_LAB_V2_HILL_MULTIPLIER',None)
        environment.pop('C3X_LAB_V2_COMPLETE_MATERIALS',None)
        environment.pop('C3X_LAB_V2_CONTINUOUS_DESERT',None)
        environment.pop('C3X_LAB_V2_COASTAL_ROCKS',None)
        environment.pop('C3X_LAB_V2_COASTAL_ROCK_PLACEMENT',None)
        environment.pop('C3X_LAB_V2_VOLCANO_SOURCE_MAPPING',None)
        environment.pop('C3X_LAB_V2_DIRECT_HILL_SOURCE',None)
        environment.pop('C3X_LAB_V2_OMIT_REPLACED_SHADOW',None)
        environment.pop('C3X_LAB_V2_RELIEF_SCALE',None)
        environment.pop('C3X_LAB_V2_VOLCANO_SCALE',None)
        if 'relief_scale' in module:
            scale=module['relief_scale']
            if type(scale) not in (int,float) or not 1<scale<=1.6:
                raise ValueError('Relief scale must be greater than one and at most 1.6')
            environment['C3X_LAB_V2_RELIEF_SCALE']=str(scale)
        if 'volcano_scale' in module:
            scale=module['volcano_scale']
            if 'relief_scale' not in module or type(scale) not in (int,float) or not 1<scale<=1.6:
                raise ValueError('Volcano scale requires broad relief and must be greater than one and at most 1.6')
            environment['C3X_LAB_V2_VOLCANO_SCALE']=str(scale)
        if module.get('omit_replaced_shadow_surface'):
            processor=f.get('packet_postprocessor',{})
            if (module['omit_replaced_shadow_surface']!=1 or not module.get('world_positions') or
                processor.get('source')!='Renderer/terrain_lab/v2/systems/lighting/scene_shadow.cpp'):
                raise ValueError('Omitting replaced shadow surface requires Q6 world shadows')
            environment['C3X_LAB_V2_OMIT_REPLACED_SHADOW']='1'
        if module.get('direct_hill_source'):
            if module['direct_hill_source'] != 1:
                raise ValueError('Unsupported direct hill source mapping')
            environment['C3X_LAB_V2_DIRECT_HILL_SOURCE']='1'
        if module.get('volcano_source_mapping'):
            if module['volcano_source_mapping'] != 1:
                raise ValueError('Unsupported volcano source mapping')
            environment['C3X_LAB_V2_VOLCANO_SOURCE_MAPPING']='1'
        if module.get('coastal_rocks'):
            environment['C3X_LAB_V2_COASTAL_ROCKS']=str(local(module['coastal_rocks']['path']))
            environment['C3X_LAB_V2_COASTAL_ROCK_PLACEMENT']=str(module['coastal_rocks'].get('placement_version',1))
        if module.get('continuous_desert'):
            if module['continuous_desert'] not in (1,2,3):
                raise ValueError('Unsupported continuous desert version')
            environment['C3X_LAB_V2_CONTINUOUS_DESERT']=str(module['continuous_desert'])
        if module.get('complete_materials'):
            if module['complete_materials'] != 1:
                raise ValueError('unknown complete material binding version')
            environment['C3X_LAB_V2_COMPLETE_MATERIALS']='1'
        if module.get('hill_source'):
            environment['C3X_LAB_V2_HILL_HEIGHT']=str(local(module['hill_source']['path']))
            environment['C3X_LAB_V2_HILL_MULTIPLIER']=str(module['hill_source']['height_multiplier'])
        if not complete or module.get("fit_viewport"):
            environment["C3X_LAB_V2_VIEWPORT"] = "x".join(map(str, f["viewport"]))
        else:
            environment.pop("C3X_LAB_V2_VIEWPORT", None)
        environment.pop("C3X_LAB_V2_PROJECTION", None)
        if module.get("projection"):
            projection = module["projection"]
            if (set(projection) != {'origin', 'half_width'} or
                len(projection['origin']) != 2 or projection['half_width'] not in (64, 32) or
                any(type(x) not in (int, float) or not math.isfinite(x)
                    for x in projection['origin'])):
                raise ValueError('invalid pinned pixel projection')
            environment['C3X_LAB_V2_PROJECTION'] = ','.join(map(str,
                [*projection['origin'], projection['half_width']]))
        environment.pop("C3X_LAB_V2_NO_TERRITORY", None)
        if module.get("suppress_territory"):
            environment["C3X_LAB_V2_NO_TERRITORY"] = "1"
        environment.pop("C3X_LAB_V2_SURFACE_QUERY",None)
        if query:
            environment["C3X_LAB_V2_SURFACE_QUERY"] = str(query)
        environment.pop("C3X_LAB_V2_LINEAR_SCENE",None)
        if module.get("linear_adapter"):
            if module["linear_adapter"]!=1: raise ValueError("unknown linear source adapter")
            environment["C3X_LAB_V2_LINEAR_SCENE"]="1"
        environment.pop("C3X_LAB_V2_DOWNSAMPLE",None)
        if complete and zoom==2 and module.get("linear_adapter"):
            environment["C3X_LAB_V2_DOWNSAMPLE"]="2"
        environment.pop("C3X_LAB_V2_WORLD_POSITIONS",None)
        if module.get("world_positions"):
            if not module.get("linear_adapter") or module["world_positions"]!=1:raise ValueError("world positions require version1 linear source adapter")
            environment["C3X_LAB_V2_WORLD_POSITIONS"]="1"
        environment.pop("C3X_LAB_V2_HYDROLOGY_DATA",None)
        if module.get("hydrology_data"):
            if module["hydrology_data"]!=1 or not module.get("world_positions") or not module.get("hydrology_hooks",{}).get("shore_sample"):raise ValueError("hydrology data v1 needs world positions and shore_sample callback")
            environment["C3X_LAB_V2_HYDROLOGY_DATA"]="1"
        run(args, env=environment)
        if complete and zoom == 2 and not query and not module.get("linear_adapter"):
            with out.open("r+b") as stream:
                stream.seek(16)
                stream.write(struct.pack("<I", 2))

    return cache.artifact("surface-query" if query else "geometry", identity, build)


def post_shader(cache, path, out):
    glslang, cross, env = tools()
    identity = {
        "closure": closure(path),
        "glslang": file_hash(glslang),
        "cross": file_hash(cross),
        "tool_libraries": tool_libraries(env),
        "interface": "postprocess.v1",
    }

    def build(target):
        spv = target.parent / "post.spv"
        run(
            [glslang, "-D", "-V", "-S", "comp", "-e", "CSPost", path, "-o", spv],
            env=env,
        )
        run(
            [
                cross,
                spv,
                "--msl",
                "--msl-version",
                "20100",
                "--msl-decoration-binding",
                "--output",
                target,
            ]
        )

    cached = cache.artifact("postprocess", identity, build)
    folder = out / "postprocess"
    folder.mkdir(exist_ok=True)
    target = folder / "CSPost.msl"
    shutil.copyfile(cached, target)
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "tier",
        choices=["quick", "check", "compose", "promote", "validate", "capabilities"],
    )
    p.add_argument("--fixture", type=Path, default=DEFAULT)
    p.add_argument("--candidate", default="candidate-01")
    p.add_argument("--output", type=Path)
    p.add_argument("--settings", type=Path)
    p.add_argument("--hours", type=int, nargs='+', choices=[0,6,12,18],
                   help="Explicit phases for bounded compose previews only")
    a = p.parse_args()
    try:
        f, module = fixture(a.fixture.resolve())
        settings = f["settings"]
        if a.settings:
            owned(a.settings.resolve(), f["track"])
            settings = json.loads(a.settings.read_text())
            validate_settings(settings)
            f["settings"] = settings
        if not re.fullmatch(r"[A-Za-z0-9_-]+", a.candidate):
            raise ValueError("unsafe candidate name")
        if a.tier == "validate":
            print("PASS fixture, modules, ownership, settings, contract v1")
            return 0
        tier = json.loads(CONTRACT.read_text())["tiers"].get(
            a.tier, {"max_tiles": 64, "phases": [12], "zooms": [1]}
        )
        if a.hours:
            if a.tier != 'compose':
                raise ValueError('phase overrides are only allowed for compose previews')
            tier=dict(tier,phases=list(dict.fromkeys(a.hours)))
        if f["tile_count"] > tier.get("max_tiles", 192) or (
            a.tier == "promote" and f["tile_count"] != 192
        ):
            raise ValueError("fixture exceeds tier tile budget")
        owner_output = (
            APP
            if f["track"] == "Q0-platform"
            else local(package(f["track"])["owns_paths"][-1])
        )
        out = (
            a.output or owner_output / "out" / f["campaign"] / f["track"] / a.candidate
        )
        out = out.resolve()
        owned(out, f["track"])
        out.mkdir(parents=True, exist_ok=True)
        cache = Cache(APP / ".cache")
        scene, metal = executables(cache)
        caps = json.loads(run([metal, "--capabilities"]))
        if settings["samples"] not in caps["sample_counts"]:
            raise ValueError("unsupported sample count on this device")
        if a.tier == "capabilities":
            print(json.dumps(caps, indent=2))
            return 0
        shader_outputs = {}
        shaderdir = out / "shaders"
        shaderdir.mkdir(exist_ok=True)
        render_modules=module.get('modules',[module])
        mixed_linear=len(render_modules)>1 and all(m.get('color_branch')=='q6_scene_linear_premultiplied_v1' for m in render_modules)
        for index,m in enumerate(render_modules if mixed_linear else [module]):
            compiled=shaders(cache,local(m['shader']),settings['mip_bias'],m.get('msl_version',20100))
            for entry,path in compiled.items():
                key=f'm{index}/{entry}' if mixed_linear else entry
                target=shaderdir/(key+'.msl');target.parent.mkdir(exist_ok=True)
                shutil.copyfile(path,target);shader_outputs[key]=path
            if mixed_linear:
                (shaderdir/f'm{index}.hlsl').write_text(shader_source(local(m['shader'])))
        pack_hash = pack_identity(cache, f["packs"])
        cache.artifact(
            "fixture",
            {"fixture": f, "modules": [module]},
            lambda p: p.write_bytes(canonical(f)),
        )
        postprocess = "box"
        if isinstance(settings["postprocess"], dict):
            postprocess = str(
                post_shader(cache, local(settings["postprocess"]["shader"]), out)
            )
        jobs = []
        outputs = []
        for hour in tier["phases"]:
            for zoom in tier["zooms"]:
                prepared = packet(cache, f, module, scene, hour, zoom, pack_hash)
                for index, offset in enumerate(settings["camera_offsets"]):
                    name = f"h{hour:02}-z{zoom}-pan{index:02}"
                    target = out / (name + ".bmp")
                    cost = out / (name + ".cost.json")
                    jobs.append(
                        [
                            str(prepared),
                            str(shaderdir),
                            str(target),
                            str(cost),
                            str(settings["samples"]),
                            str(settings["anisotropy"]),
                            str(settings["render_scale"]),
                            str(offset[0]),
                            str(offset[1]),
                            "2" if a.tier in ("check", "promote") else "1",
                            postprocess,
                            str(settings["postprocess"].get("contract",1)) if isinstance(settings["postprocess"],dict) else "1",
                        ]
                    )
                    outputs.append(
                        {
                            "image": relative(target),
                            "cost": relative(cost),
                            "packet": relative(prepared),
                            "source_metadata": ({"path":relative(Path(str(prepared)+".source.json")),"sha256":file_hash(Path(str(prepared)+".source.json"))} if Path(str(prepared)+".source.json").exists() else None),
                            "hour": hour,
                            "zoom": zoom,
                            "offset": offset,
                        }
                    )
        batch = out / "batch.json"
        batch.write_bytes(canonical(jobs))
        run([metal, "--batch", batch])
        for result in outputs:
            target = local(result["image"])
            result["sha256"] = file_hash(target)
            repeat = Path(str(target) + ".repeat1.bmp")
            if a.tier in ("check", "promote") and file_hash(repeat) != result["sha256"]:
                raise ValueError("nondeterministic Metal output: " + target.name)
        # The capability fingerprint contains no registry ID or personal device name.
        effective = {
            "contract": file_hash(CONTRACT),
            "fixture": f,
            "module": module,
            "capabilities": caps,
            "settings": settings,
            "shader_hashes": {k: file_hash(v) for k, v in shader_outputs.items()},
            "pack_hash": pack_hash,
            "postprocess_hash": (
                file_hash(postprocess) if postprocess != "box" else "box"
            ),
        }
        report = {
            "schema": "c3x.lab_v2.run.v1",
            "tier": a.tier,
            "effective": effective,
            "render_identity": digest(canonical(effective)),
            "outputs": outputs,
            "cache_events": cache.events,
            "visual_review": "pending direct inspection",
        }
        (out / "report.json").write_bytes(canonical(report))
        if a.tier == "promote":
            from parity import d3d

            d3d(out / "report.json")
        print(
            f"PASS {a.tier}: {len(outputs)} variants; " + relative(out / "report.json")
        )
        return 0
    except (ValueError, OSError, KeyError) as e:
        print("lab-v2: " + str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
