"""Route changed visual inputs to current fixtures, not historical campaigns.

Narrow routes describe actual fixture content. Unclassified code/art is shared
until its consumers are established; it must never disappear from review.
"""
import hashlib
import json


def consumers(entries):
    objects = {key for key, value in entries.items() if value["recipe"].get("objects")}
    return {
        "coastal-waves": set(entries), "natural": set(entries), "hill-cliff": set(entries),
        "tile-sites": {"huts-camps", "goody-huts", "barbarian-camps", "shadows"} & entries.keys(),
        "cities": objects | ({"cities"} & entries.keys()),
        "units": {"units", "animation", "shadows"} & entries.keys(),
        "resources": objects | ({"resources", "animation", "huts-camps", "barbarian-camps"} & entries.keys()),
        "resource-animation": objects | ({"resources", "animation"} & entries.keys()),
    }


def source_routes(entries, assets):
    groups = consumers(entries)
    routes = {}
    for name, record in assets.items():
        selected = groups.get(name, set(entries))
        for path in set(record.get("inputs", {})) | set(record.get("outputs", {})):
            routes.setdefault(path, set()).update(selected)
    return groups, routes


def owners(path, entries, groups, routes):
    # Preparation/dispatch changes can affect every builder, even if only one
    # cached read closure happens to mention the helper today.
    if path in ("Renderer/lab/asset_preparation.py", "Renderer/lab/preparation.py",
                "Renderer/lab/dependencies.py", "Renderer/renderer.py"):
        return set(entries)
    if path in routes:
        return routes[path]
    if path.startswith(("Renderer/lab/shared/cities/", "Renderer/packs/City")) or path in (
            "Renderer/lab/shared/shaders/lighting/local_facade_lights.hlsl",
            "Renderer/lab/shared/shaders/lighting/city_environment.hlsl",
            "Renderer/lab/shared/shaders/objects/city_scene_material.hlsl",
            "Renderer/lab/shared/shaders/objects/settlement_ground.hlsl",
            "Renderer/native/city_fidelity/geometry.h",
            "Renderer/native/city_fidelity/runtime.h"):
        return groups["cities"]
    if path.startswith(("Renderer/packs/Unit", "Renderer/native/environment_refresh/prepare_unit")):
        return groups["units"]
    if path.startswith(("Renderer/lab/shared/resources/", "Renderer/packs/Resource")):
        return groups["resources"]
    # Global sun/moon, shadow, terrain and transition inputs remain shared.
    # In particular do not infer a narrow scope from a filename or a single
    # category's entry point into the monolithic native renderer.
    return set(entries)


def signatures(records, entries, *, assets=None, generated_shaders=()):
    groups, routes = source_routes(entries, assets or {})
    selected = {key: {} for key in entries}
    generated = set(generated_shaders)
    for path, digest in sorted(records.items()):
        # These exact generated outputs have a checked preparation contract.
        # Their shared source/adapters, not duplicated generated text, determine
        # scope. An edited generated output still fails require_prepared().
        if path in generated:
            continue
        for key in owners(path, entries, groups, routes):
            selected[key][path] = digest
    return {key: hashlib.sha256(json.dumps({"inputs": selected[key],
                "recipe": value["recipe"], "dependencies": value["depends_on"]},
                sort_keys=True).encode()).hexdigest()
            for key, value in entries.items()}
