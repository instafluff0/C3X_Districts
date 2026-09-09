"""Execute actual site capture/ownership and verify imported surface normals."""
from pathlib import Path
import math
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class TileSites(unittest.TestCase):
    def test_visibility_capture_and_exclusive_ownership(self):
        source = (ROOT / "injected_code.c").read_text()
        start = source.index("\t\t\trecord->barbarian_tribe_id = -1;")
        end = source.index("\t\t\tif (tile->vtable->m18_Check_Mines", start)
        capture = source[start:end]
        start = source.index("bool\nvalidate_custom_renderer_replacement_ownership")
        end = source.index("\n// Compact topology", start)
        harness = r'''
#include "c3x_renderer_api.h"
#include <cassert>
#include <cstddef>
enum {SQ_Forest=7,SQ_Jungle=8,SQ_Swamp=9,SQ_Volcano=10};
struct State {int custom_renderer_tile_count; c3x_renderer_tile_v1* custom_renderer_tiles;};
State state,*is=&state;
struct Tile;
struct Vtable {bool (*m15_Check_Goody_Hut)(Tile*,int,int); bool (*m7_Check_Barbarian_Camp)(Tile*,int,int); short (*m44_Get_Barbarian_TribeID)(Tile*);};
struct Tile {Vtable* vtable; bool hut,camp; int allowed_viewer;short tribe;};
int queries=0;
bool hut(Tile* t,int,int viewer){++queries;return viewer==t->allowed_viewer && t->hut;}
bool camp(Tile* t,int,int viewer){++queries;return viewer==t->allowed_viewer && t->camp;}
short tribe(Tile* t){return t->tribe;}
void capture(c3x_renderer_tile_v1* record,Tile* tile,int visible_to_civ_id){int __=0;
''' + capture + "\n}\n" + source[start:end] + r'''
int main(){
    Vtable table={hut,camp,tribe};Tile native={&table,true,true,7,29};
    c3x_renderer_tile_v1 tile={};unsigned flags=0;state={1,&tile};
    c3x_renderer_output_v1 out={};out.replacement_tile_count=1;out.replacement_tile_flags=&flags;
    for(int viewer:{0,6,7,8}){
        tile={};tile.tile_flags=C3X_RENDERER_TILE_RENDER;capture(&tile,&native,viewer);
        flags=C3X_RENDERER_TILE_CUSTOM_TERRAIN_REPLACED;
        if(viewer==7){
            assert(tile.barbarian_tribe_id==29);
            assert(!validate_custom_renderer_replacement_ownership(&out));
            flags|=C3X_RENDERER_TILE_CUSTOM_HUT_REPLACED;
            assert(!validate_custom_renderer_replacement_ownership(&out));
            flags|=C3X_RENDERER_TILE_CUSTOM_CAMP_REPLACED;
            assert(validate_custom_renderer_replacement_ownership(&out));
        }else{
            assert(tile.improvement_flags==0 && tile.barbarian_tribe_id==-1);
            assert(validate_custom_renderer_replacement_ownership(&out));
            flags|=C3X_RENDERER_TILE_CUSTOM_CAMP_REPLACED;
            assert(!validate_custom_renderer_replacement_ownership(&out));
        }
    }
    native.hut=native.camp=false;tile={};tile.tile_flags=C3X_RENDERER_TILE_RENDER;
    capture(&tile,&native,7);assert(!tile.improvement_flags && tile.barbarian_tribe_id==-1);
    flags=C3X_RENDERER_TILE_CUSTOM_TERRAIN_REPLACED;assert(validate_custom_renderer_replacement_ownership(&out));
    assert(queries==10);
}
'''
        harness = harness.replace('#include <cassert>', '#include <cassert>\n#include <initializer_list>')
        with tempfile.TemporaryDirectory() as folder:
            cpp = Path(folder) / "sites.cpp"; cpp.write_text(harness)
            exe = Path(folder) / "sites"
            subprocess.run(["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "Renderer/native"), str(cpp), "-o", str(exe)], check=True)
            subprocess.run([str(exe)], check=True)

    def test_compound_and_height_transform_preserve_normal_perpendicularity(self):
        from Renderer.tools.asset_compiler.build_site_runtime import transform_mesh
        mesh = {"vertices": [{"position": p, "normal": [-1, 1, 1], "uv0": [0, 0]} for p in ([0, 0, 0], [1, 0, 1], [0, 1, -1])]}
        for angle in range(0, 360, 15):
            c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
            matrix = [2*c, 2*s, 0, 0, -s, c, 0, 0, 0, 0, .5, 0, 3, -2, 4, 1]
            vertices = transform_mesh(mesh, matrix)["vertices"]
            for i in (1, 2):
                tangent = [a-b for a, b in zip(vertices[i]["position"], vertices[0]["position"])]
                self.assertAlmostEqual(0, sum(a*b for a, b in zip(tangent, vertices[0]["normal"])), places=10)

    def test_runtime_pack_keeps_all_body_materials_and_deterministic_bytes(self):
        from Renderer.tools.asset_compiler.build_site_runtime import build, plan
        if not (ROOT / "Renderer/packs/TileObjectsNormalized/manifest.json").exists():
            self.skipTest("Local normalized site art unavailable")
        groups, textures, consumed = plan()
        self.assertEqual({"hut_0", "hut_1", "hut_2", "camp"}, set(groups))
        self.assertGreater(len(groups["camp"]), 1)
        self.assertTrue(0 < len(textures) <= 8)
        self.assertTrue(consumed)
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder) / "a", Path(folder) / "b"
            build(a); build(b)
            self.assertEqual((a / "sites.bin").read_bytes(), (b / "sites.bin").read_bytes())
