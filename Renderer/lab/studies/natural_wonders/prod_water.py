"""Production water, foam-collar and effect edits for natural-wonder kits (prod_tree.py merges these).

Derived from the Lab's stamp_tree.py patches: the kit comes from the tile's topology
relief slot (terrain-like water and collars) or its record's name (effects, which
appear with exploration like every object).
"""
from Renderer.lab.studies.natural_wonders import stamp_tree

PACK = "::c3x_renderer::render_core::NaturalWonderPack"
LAB = stamp_tree.PATCHES


def derive(text: str, replacements: list[tuple[str, str]]) -> str:
    for old, new in replacements:
        if text.count(old) != 1:
            raise ValueError(f"Lab patch text moved: {old[:60]!r}")
        text = text.replace(old, new)
    if "Lab only" in text:
        raise ValueError("a Lab-only comment survived the port")
    return text


_renderer = LAB["Renderer/native/c3x_renderer.cpp"]
WATER = derive(_renderer[5][1], [
    ("                    // Lab only (studies/natural_wonders): a wonder's lake (7) or channel (8) is a\n",
     "                    // A natural wonder's lake (7) or fjord channel (8) is a\n"),
    ("                    if(!natural.wonder_effects.empty() && queries.tile(wonder_c,wonder_r).wonder){\n"
     "                        auto const& records=natural.wonder_effects;\n",
     f"                    auto const* wonder_kit=natural.wonder_pack.relief(queries.tile(wonder_c,wonder_r).wonder);\n"
     "                    if(wonder_kit && !wonder_kit->effects.empty()){\n"
     "                        auto const& records=wonder_kit->effects;\n"),
    ("                            if(e[0]<6.5f || e[0]>8.5f || (e[0]>7.5f && layer!=geometry_water))continue;\n"
     "                            bool const channel=e[0]>7.5f;\n",
     f"                            bool const channel=e[0]==float({PACK}::effect_channel);\n"
     f"                            if((e[0]!=float({PACK}::effect_lake) && !channel) || (channel && layer!=geometry_water))continue;\n"),
    ("float distance=natural.wonder_water(x,y,", "float distance=natural.wonder_water(*wonder_kit,x,y,"),
])
RIVER_DECL = derive(_renderer[2][1], [
    ("            // Lab only (studies/natural_wonders): a wonder's channel as river water.\n",
     "            // A natural wonder's fjord channel, drawn as river water.\n")])

_objects = LAB["Renderer/native/object_preparation.h"]
OBJECT_GATE = derive(_objects[0][1], [
    ("    // Lab only: a natural wonder's attached effects (stamp_tree.py) on its mountain-hosted tile.\n"
     "    bool const wonder=capture.real_terrain_type==6 && input.city_ready && !natural.wonder_effects.empty();\n",
     "    // A natural wonder's attached effects (its record names the kit: explored tiles only).\n"
     "    auto const* wonder_kit=input.city_ready?natural.wonder_pack.find(capture.natural_wonder):nullptr;\n"
     "    bool const wonder=wonder_kit && !wonder_kit->effects.empty();\n")])
OBJECT_EFFECTS = derive(_objects[3][1], [
    ("    if(wonder && library.effect_material<library.materials.size() && queries.tile(nc,nr).wonder){\n",
     "    if(wonder && library.effect_material<library.materials.size()){\n"),
    ("natural.wonder_effects,route_height", "wonder_kit->effects,route_height")])

_geometry = LAB["Renderer/native/source_fidelity/geometry.h"]
GEOMETRY_EFFECTS = derive(_geometry[1][1], [
    ("    // Lab only: a natural wonder's attached effects (stamp_tree.py) on its rendered relief.\n"
     "    if(!prepared_objects && city_profile && cities.ready && queries.tile(nc,nr).wonder && !natural.wonder_effects.empty()){\n",
     "    // A natural wonder's attached effects on its rendered relief.\n"
     "    auto const* wonder_kit=natural.wonder_pack.relief(queries.tile(nc,nr).wonder);\n"
     "    if(!prepared_objects && city_profile && cities.ready && wonder_kit && !wonder_kit->effects.empty()){\n"),
    ("natural.wonder_effects,relief_at", "wonder_kit->effects,relief_at")])

WAVES = f"""struct WavePoint {{ Point position; float distance,along,coverage; }};
// A water natural wonder's foam collars round its islets: its kit's ribbon points (five
// floats each: x, y offsets from the tile centre, distance, along, coverage) as a
// triangle list, found by the tile's topology relief slot.
inline std::vector<WavePoint> wonder_waves(WorldCoast const& coast,int c,int r,{PACK} const* wonders){{
    std::vector<WavePoint> out;
    auto const* kit=wonders?wonders->relief(coast.world().tile(c,r).wonder):nullptr;
    if(!kit)return out;
    auto const& points=kit->waves;
    for(std::size_t i=0;i+5<=points.size();i+=5)
        out.push_back({{Point{{c+.5+points[i],r+.5+points[i+1]}},points[i+2],points[i+3],points[i+4]}});
    return out;
}}"""

PATCHES = {
    "Renderer/native/c3x_renderer.cpp": [
        _renderer[0], _renderer[1],
        (_renderer[2][0], RIVER_DECL), _renderer[3],
        (_renderer[5][0], WATER),
        ("auto ribbon=indexed_wave_ribbon(coastal_wave_ribbon(world_coast,c,r,.30f+.70f*seed));\n",
         "auto ribbon=indexed_wave_ribbon(coastal_wave_ribbon(world_coast,c,r,.30f+.70f*seed,&natural.wonder_pack));\n"),
        ("auto ribbon=indexed_wave_ribbon(coastal_wave_ribbon(world_coast,c,r,.30f+.70f*seed));if(ribbon.empty())continue;",
         "auto ribbon=indexed_wave_ribbon(coastal_wave_ribbon(world_coast,c,r,.30f+.70f*seed,&natural.wonder_pack));"
         "if(ribbon.empty())continue;")],
    "Renderer/native/render_core/coastal_waves.h": [
        ('#include "world_coast.h"\n', '#include "world_coast.h"\n#include "natural_wonder_pack.h"\n'),
        ("struct WavePoint { Point position; float distance,along,coverage; };", WAVES),
        ("inline std::vector<WavePoint> coastal_wave_ribbon(WorldCoast const& coast,int c,int r,float scale) {",
         f"inline std::vector<WavePoint> coastal_wave_ribbon(WorldCoast const& coast,int c,int r,float scale,\n"
         f"        {PACK} const* wonders=nullptr) {{"),
        ("    if(!coastal_wave_site(coast,c,r,chosen) || !coastal_wave_spaced(coast,c,r,chosen))return {};",
         "    auto wonder=wonder_waves(coast,c,r,wonders);\n"
         "    if(!coastal_wave_site(coast,c,r,chosen) || !coastal_wave_spaced(coast,c,r,chosen))return wonder;"),
        ("    if(!active)return {};", "    if(!active)return wonder;"),
        ("        for(auto i:{a,b,e,a,e,d})result.push_back(grid[i]);\n    }\n    return result;",
         "        for(auto i:{a,b,e,a,e,d})result.push_back(grid[i]);\n    }\n"
         "    result.insert(result.end(),wonder.begin(),wonder.end());\n    return result;")],
    "Renderer/native/object_preparation.h": [(_objects[0][0], OBJECT_GATE), _objects[1], (_objects[3][0], OBJECT_EFFECTS)],
    "Renderer/native/source_fidelity/geometry.h": [(_geometry[1][0], GEOMETRY_EFFECTS)],
}
