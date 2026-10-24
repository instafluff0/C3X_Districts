#!/usr/bin/env python3
"""Private Lab tree that lights district buildings at night from Civ VI emissive maps.

District compositions are resource-pack bodies. Civ VI ships an emissive map
for about half of their materials (windows, lamps, signs). kits.bake() packs
them into `textures/emissive_N.dds`, laid out exactly like `atlas_N.dds`; this
tree teaches Renderer64 to read them:

- c3x_renderer.cpp: an optional emissive atlas per resource texture slot;
  bodies whose slot has one (never `decal/` ground) carry material code .002;
  the feature draws bind the emissive atlases in the road slots t98-t105
  (roads draw in their own layer) and restore them afterwards;
- render_core/terrain_scene.hlsl (maintained by hand; environment_refresh and
  city_fidelity feature shaders are prepared from it): resource bodies with
  code .002 take their slot's emissive texel as `emissive`, and join the city,
  mine and infrastructure weights that the shared night line lets glow
  (environment_night_activation, emissive scale), so they light like cities.

The checkout is never written; the private tree lives in
Renderer/lab/out/districts/tree-night/ (mountains study private_tree.py).

    python3 Renderer/lab/studies/districts/night_tree.py prepare   # sync + patch
    python3 Renderer/lab/studies/districts/night_tree.py build     # VM compile
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.lab.studies.mountains import private_tree

NAME = "night"
private_tree.OUT = ROOT / "Renderer/lab/out/districts"

EMISSIVE_SAMPLE = "".join(
    f"        {'if' if slot == 0 else 'else if'} (mine_slot < {slot}.5) emissive = "
    f"road_base_texture_{slot}.Sample(material_sampler, input.uv).rgb;\n" for slot in range(7)) + \
    "        else emissive = road_base_texture_7.Sample(material_sampler, input.uv).rgb;\n"

PATCHES = {
    "Renderer/native/c3x_renderer.cpp": [
        ("    std::array<ID3D11ShaderResourceView *, 8> resource_texture_views = {};\n",
         "    std::array<ID3D11ShaderResourceView *, 8> resource_texture_views = {};\n"
         "    // Lab (studies/districts): optional night emissive atlases beside the resource atlases.\n"
         "    std::array<std::vector<std::uint8_t>, 8> resource_emissive_dds;\n"
         "    std::array<ID3D11ShaderResourceView *, 8> resource_emissive_views = {};\n"
         "    std::array<bool, 8> resource_emissive_present = {};\n"),
        ("files(resource_texture_dds);", "files(resource_texture_dds);files(resource_emissive_dds);"),
        ("        for (ID3D11ShaderResourceView *& view : resource_texture_views)\n            release(view);\n",
         "        for (ID3D11ShaderResourceView *& view : resource_texture_views)\n            release(view);\n"
         "        for (ID3D11ShaderResourceView *& view : resource_emissive_views)\n            release(view);\n"),
        ("            release(resource_texture_views[index]);\n            resource_texture_dds[index].clear();\n",
         "            release(resource_texture_views[index]);\n            resource_texture_dds[index].clear();\n"
         "            release(resource_emissive_views[index]);\n            resource_emissive_dds[index].clear();\n"
         "            resource_emissive_present[index] = false;\n"),
        ('            resource_root, "resource_runtime.bin", resource_bundle, resource_texture_dds, true);\n',
         '            resource_root, "resource_runtime.bin", resource_bundle, resource_texture_dds, true);\n'
         "        // Lab (studies/districts): emissive_N.dds beside atlas_N.dds lights bodies at night.\n"
         "        for (std::size_t index = 0; index < resource_emissive_dds.size(); ++index) {\n"
         "            resource_emissive_present[index] = false;\n"
         "            if (!resource_assets_ready) continue;\n"
         "            std::string emissive = resource_bundle.texture_paths[index];\n"
         '            auto at = emissive.find("atlas_");\n'
         "            if (at == std::string::npos) continue;\n"
         '            emissive.replace(at, 6, "emissive_");\n'
         "            resource_emissive_present[index] =\n"
         "                load_dds_bytes(resource_root.c_str(), emissive.c_str(), resource_emissive_dds[index],\n"
         "                               DXGI_FORMAT_BC1_UNORM_SRGB, DXGI_FORMAT_BC1_UNORM) ||\n"
         "                load_dds_bytes(resource_root.c_str(), emissive.c_str(), resource_emissive_dds[index],\n"
         "                               DXGI_FORMAT_BC3_UNORM_SRGB, DXGI_FORMAT_BC3_UNORM);\n"
         "            if (!resource_emissive_present[index]) resource_emissive_dds[index].clear();\n"
         "        }\n"),
        ("                    resource_texture_dds[index], resource_texture_views[index], true);\n        }\n",
         "                    resource_texture_dds[index], resource_texture_views[index], true);\n        }\n"
         "        for (std::size_t index = 0; index < resource_emissive_views.size(); ++index)\n"
         "            if (resource_emissive_present[index] && !ensure_dds_texture(\n"
         "                    resource_emissive_dds[index], resource_emissive_views[index], false))\n"
         "                resource_emissive_present[index] = false;\n"),
        ("item.u, item.v, item.rotation, item.scale, 21.0f, 0.0f, true, tile.terrain_type < 11,",
         "item.u, item.v, item.rotation, item.scale, 21.0f,\n"
         "                        resource_emissive_present[resource_bundle.assets[item.asset].texture_index & 7u] &&\n"
         '                        id.rfind("decal/", 0) != 0 ? .002f : 0.0f, true, tile.terrain_type < 11,'),
        ("                context->PSSetShaderResources(94, 4, feature_texture_views.data() + 4);\n"
         "                if (!draw(geometry_feature))\n                    return false;\n",
         "                context->PSSetShaderResources(94, 4, feature_texture_views.data() + 4);\n"
         "                // Lab (studies/districts): resource emissive atlases take the road slots for this draw.\n"
         "                context->PSSetShaderResources(98, 8, resource_emissive_views.data());\n"
         "                bool feature_drawn = draw(geometry_feature);\n"
         "                context->PSSetShaderResources(98, 8, route_texture_views.data());\n"
         "                if (!feature_drawn)\n                    return false;\n"),
        ("        context->PSSetShader(feature_pixel_shader,nullptr,0);\n#ifdef C3X_RENDERER_BENCHMARK_ORACLE\n"
         "        if(!diagnostic_animation)\n#endif\n"
         "        if(!draw_cached_geometry(geometry_feature,buffers,rectangles,settings,nullptr))return false;\n",
         "        context->PSSetShader(feature_pixel_shader,nullptr,0);\n"
         "        context->PSSetShaderResources(98,8,resource_emissive_views.data());\n"
         "#ifdef C3X_RENDERER_BENCHMARK_ORACLE\n"
         "        if(!diagnostic_animation)\n#endif\n"
         "        if(!draw_cached_geometry(geometry_feature,buffers,rectangles,settings,nullptr))return false;\n"
         "        context->PSSetShaderResources(98,8,route_texture_views.data());\n"),
    ],
    "Renderer/native/render_core/terrain_scene.hlsl": [
        ("    float city_weight = step(28.5, input.material_index);\n",
         "    // Lab (studies/districts): resource bodies flagged .002 glow at night from\n"
         "    // their slot's emissive atlas, bound in the road slots for the feature draws.\n"
         "    if (resource_weight > 0.5 && abs(material_fraction - 0.002) < 0.001)\n"
         "    {\n" + EMISSIVE_SAMPLE + "    }\n"
         "    float city_weight = step(28.5, input.material_index);\n"),
        # The shared night line only lets city, mine and infrastructure emission through.
        ("        saturate(city_weight + mine_weight + raised_infrastructure_weight) *\n",
         "        saturate(city_weight + mine_weight + raised_infrastructure_weight +\n"
         "                 resource_weight * step(abs(material_fraction - 0.002), 0.001)) *\n"),
    ],
}


def prepare() -> None:
    root = private_tree.tree(NAME)
    private_tree.own(NAME, list(PATCHES))
    # Owned files start from the checkout each time, so patches apply to its current text.
    for relative in PATCHES:
        (root / relative).write_bytes((ROOT / relative).read_bytes())
    private_tree.sync(NAME)
    for relative, edits in PATCHES.items():
        path = root / relative
        text = path.read_text()
        for old, new in edits:
            if text.count(old) != 1:
                raise SystemExit(f"{relative}: patch anchor not found once: {old[:70]!r}")
            text = text.replace(old, new)
        path.write_text(text)
    print("patched", len(PATCHES), "files in", root.relative_to(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("prepare", "build"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        private_tree.build(NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
