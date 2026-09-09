"""Incremental current asset builds; preserve source art and conflicting outputs."""
from pathlib import Path
import json
import shutil
import tempfile

from .preparation import ROOT, digest, local


def jobs():
    from Renderer.native.source_fidelity import prepare as natural
    from Renderer.native.render_core import prepare_assets as cliffs

    def natural_sources():
        return json.loads((natural.HERE / "provenance.json").read_text())["source_sha256"]

    def build_natural(stage):
        record = natural.build_pack(stage / "Renderer/packs/NaturalFidelityRuntime")
        metadata = stage / "Renderer/native/source_fidelity/provenance.json"
        metadata.parent.mkdir(parents=True)
        metadata.write_text(json.dumps(record, indent=2) + "\n")
        return record["source_sha256"]

    def cliff_sources():
        record = json.loads((cliffs.OUTPUT / "manifest.json").read_text())
        return {r["source"]: r["source_sha256"] for r in record["files"]}

    def build_cliffs(stage):
        record = cliffs.prepare(output=stage / "Renderer/packs/TerrainProfileR1")
        return {r["source"]: r["source_sha256"] for r in record["files"]}

    def city_helpers():
        return {p.relative_to(ROOT).as_posix(): digest(p)
                for p in (ROOT / "Renderer/lab/shared/cities").glob("*.py")}

    def city_sources():
        record = ROOT / "Renderer/lab/.cache/assets/cities.json"
        if record.exists():
            # Full observed read closure, including meshes and catalog files,
            # not only the older pack manifest's material/source-normal pins.
            sources = json.loads(record.read_text())["inputs"]
        else:
            sources = json.loads((ROOT / "Renderer/packs/CityCompositionRuntime/manifest.json").read_text())["source_sha256"]
        return dict(sources, **city_helpers())

    def build_cities(stage):
        from Renderer.native.city_fidelity import prepare_pack as cities
        _, consumed = cities.build_pack(stage / "Renderer/packs/CityCompositionRuntime")
        consumed.update(city_helpers())
        return consumed

    def unit_sources():
        record = ROOT / "Renderer/lab/.cache/assets/units.json"
        if record.exists():
            return json.loads(record.read_text())["inputs"]
        return ("Renderer/packs/UnitAnimationRuntime/manifest.json",
                "Renderer/packs/UnitAnimationRuntime/bindings.json",
                "Renderer/packs/UnitNormalFidelity/manifest.json")

    def build_units(stage):
        from Renderer.native.environment_refresh import prepare_units as units
        evidence, consumed = units.build_pack(stage / "Renderer/packs/UnitAnimationFidelity")
        print(f"Prepared {evidence['unit_count']} units and {evidence['native_keys']} native keys", flush=True)
        return consumed

    def resource_helpers():
        paths = ["Renderer/lab/shared/resources/clip_units.json"]
        paths.extend("Renderer/tools/asset_compiler/" + name + ".py" for name in (
            "build_resource_runtime", "build_resource_animation_runtime", "normalized_animation",
            "normalized_skin", "normalized_pose_cache", "school_orientation", "unit_model_extractor"))
        return {path: digest(ROOT / path) for path in paths}

    def resource_sources(name):
        record = ROOT / "Renderer/lab/.cache/assets" / (name + ".json")
        sources = json.loads(record.read_text())["inputs"] if record.exists() else {
            "Renderer/packs/ResourceNormalized/manifest.json": None,
            "Renderer/packs/ResourceAnimatedLab/manifest.json": None}
        return dict(sources, **resource_helpers())

    def build_resources(stage):
        from Renderer.tools.asset_compiler import build_resource_runtime as resources
        consumed = {}
        resources.build(ROOT / "Renderer/packs/ResourceNormalized",
                        stage / "Renderer/packs/ResourceNormalized/resource_runtime.bin", consumed=consumed)
        consumed.update(resource_helpers())
        return consumed

    def build_resource_animation(stage):
        from Renderer.tools.asset_compiler import build_resource_animation_runtime as resources
        consumed = {}
        result = resources.build(ROOT / "Renderer/packs/ResourceAnimatedLab",
            ROOT / "Renderer/packs/ResourceNormalized", stage / "Renderer/packs/ResourceAnimationRuntime",
            consumed=consumed)
        consumed.update(resource_helpers())
        print(f"Prepared animation for {len(result['resources'])} resource families", flush=True)
        return consumed

    def site_sources():
        from Renderer.tools.asset_compiler.build_site_runtime import plan
        return plan()[2]

    def build_sites(stage):
        from Renderer.tools.asset_compiler.build_site_runtime import build
        return build(stage / "Renderer/packs/TileSitesRuntime")

    from Renderer.tools.asset_compiler import build_wave_runtime as waves

    return (
        ("coastal-waves", waves.sources, lambda stage: waves.build(stage / "Renderer/packs/CoastalWavesRuntime"),
         "Renderer/tools/asset_compiler/build_wave_runtime.py"),
        ("tile-sites", site_sources, build_sites, "Renderer/tools/asset_compiler/build_site_runtime.py"),
        ("natural", natural_sources, build_natural, "Renderer/native/source_fidelity/prepare.py"),
        ("hill-cliff", cliff_sources, build_cliffs, "Renderer/native/render_core/prepare_assets.py"),
        ("cities", city_sources, build_cities, "Renderer/native/city_fidelity/prepare_pack.py"),
        ("units", unit_sources, build_units, "Renderer/native/environment_refresh/prepare_units.py"),
        ("resources", lambda: resource_sources("resources"), build_resources,
         "Renderer/tools/asset_compiler/build_resource_runtime.py"),
        ("resource-animation", lambda: resource_sources("resource-animation"), build_resource_animation,
         "Renderer/tools/asset_compiler/build_resource_animation_runtime.py"),
    )


def source_inputs(root, sources, builder, *, fallback=None):
    try:
        names = set(sources())
    except FileNotFoundError:
        if fallback is None:
            raise
        names = set(fallback)
    required = {builder, "Renderer/lab/asset_preparation.py"}
    names.update(required)
    # An old dependency may have been removed by an edited source catalog.
    # Mark it dirty and let the builder decide whether it is still required.
    return {name: digest(local(root, name)) if name in required or local(root, name).is_file() else None
            for name in sorted(names)}


def output_path(root, name):
    path = local(root, name)
    if path != root.resolve() / name:
        raise ValueError("Generated asset path must not alias another location: " + name)
    return path


def refresh(job, *, root=ROOT, check_only=False):
    name, sources, build, builder = job
    cache = root / "Renderer/lab/.cache/assets"
    receipt = cache / (name + ".json")
    previous = json.loads(receipt.read_text()) if receipt.exists() else None
    before = source_inputs(root, sources, builder, fallback=(previous or {}).get("inputs"))
    existing = {}
    for path, expected in (previous or {}).get("outputs", {}).items():
        target = output_path(root, path)
        if target.exists():
            existing[path] = digest(target)
            if existing[path] != expected:
                raise ValueError("Preserving edited generated asset: " + path)
    fresh = previous and previous["inputs"] == before and len(existing) == len(previous["outputs"])
    if check_only:
        if not fresh:
            raise ValueError("Asset preparation is stale: " + name + "; run Renderer/renderer.py prepare, then rerender")
        return []
    if fresh:
        return []
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=name + "-", dir=cache) as directory:
        stage = Path(directory).resolve()
        consumed = build(stage)
        # Builders report bytes actually read, including new texture paths
        # introduced by a changed source bundle, rather than pinning old paths.
        if source_inputs(root, sources, builder, fallback=(previous or {}).get("inputs")) != before or any(
                (digest(local(root, path)) if local(root, path).is_file() else None) != expected
                for path, expected in consumed.items()):
            raise ValueError("Asset sources changed during preparation: " + name)
        produced = {p.relative_to(stage).as_posix(): digest(p)
                    for p in stage.rglob("*") if p.is_file()}
        if not produced or set(produced).intersection(set(consumed) | set(before)):
            raise ValueError("Asset output is empty or overlaps source inputs: " + name)
        changed = []
        for path, expected in produced.items():
            target = output_path(root, path)
            actual = digest(target) if target.exists() else None
            if actual is not None and actual != expected and actual != (previous or {}).get("outputs", {}).get(path):
                raise ValueError("Preserving unverified generated asset: " + path)
            if actual != expected or target.stat().st_nlink > 1:
                changed.append(path)
        # Validate all output conflicts before publishing anything. Replacing
        # identical hard-linked outputs detaches runtime bytes from source edits.
        for path in changed:
            target = output_path(root, path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(prefix=".asset-", dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
            try:
                shutil.copyfile(local(stage, path), temporary)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        current = dict(consumed)
        current.update({p: before[p] for p in (builder, "Renderer/lab/asset_preparation.py")})
        record = {"inputs": current, "outputs": produced}
        with tempfile.NamedTemporaryFile(mode="w", prefix=name + "-", suffix=".tmp", dir=cache, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(record, indent=2) + "\n")
        try:
            temporary.replace(receipt)
        finally:
            temporary.unlink(missing_ok=True)
    return changed


def prepare(*, selected=None, check_only=False):
    changed = []
    for job in jobs():
        if selected is not None and job[0] not in selected:
            continue
        changed.extend(refresh(job, check_only=check_only))
    return changed
