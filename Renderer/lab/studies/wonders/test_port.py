#!/usr/bin/env python3
"""Contract tests for the constructed-wonder port in the checkout (stage 2).

- Lamp modes ride in a night light's seed (seed + 64 * mode). Every production
  city effect seed must stay below 64, so every existing record is mode 0.
- The patched city effect shader decodes the mode only for kind 2, keeps mode 0's
  seed expression exactly as before, and takes a lamp's flicker phase from
  fmod(seed, 64).
"""
from __future__ import annotations

import struct
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.lab.studies.city_readability import runtime_pack as rp

SHADER = "Renderer/lab/shared/shaders/objects/city_scene_material.hlsl"
MODE_ZERO_SEED = "clamp(p.city_ao_uv.x,0,64)"
PACK = "Renderer/native/render_core/natural_wonder_pack.h"


class LampModes(unittest.TestCase):
    def test_production_city_seeds_are_mode_zero(self):
        library = rp.decode(ROOT / "Renderer/packs/CityCompositionRuntime/city.bin")
        seeds = [effect[6] for template in library["templates"] for instance in template["instances"]
                 for effect in instance["effects"]]
        self.assertTrue(seeds)
        self.assertLess(max(seeds), 64.0)

    def test_patched_shader_keeps_mode_zero(self):
        text = (ROOT / SHADER).read_text()
        self.assertIn("lamp_mode=kind>1.5 && kind<2.5?floor(max(p.city_ao_uv.x,0)/64):0", text)
        # Mode 0 (every city light) takes the seed exactly as before the patch.
        self.assertIn(f"seed=lamp_mode>.5?fmod(p.city_ao_uv.x,64):{MODE_ZERO_SEED}", text)
        self.assertIn("if(lamp_mode>.5){", text)


class RiverAndSlots(unittest.TestCase):
    def test_river_priority_matches_c3x(self):
        """The renderer's river table is C3X's get_primary_river_direction: its priority
        list, direction_to_neighbor_bit and direction_to_offset (raw Civ III offsets turned
        into world column/row: c = (dx + dy) / 2, r = (dx - dy) / 2). A C3X change fails here."""
        import re
        source = (ROOT / "injected_code.c").read_text()
        body = source[source.index("get_primary_river_direction (Tile * tile"):]
        order = re.findall(r"DIR_\w+", re.search(r"dirs\[\] = \{([^}]*)\}", body).group(1))
        bits = dict(re.findall(r"case (DIR_\w+):\s*return (\d+);",
                               source[source.index("direction_to_neighbor_bit (enum direction dir)"):][:600]))
        offsets = {name: (int(dx), int(dy)) for name, dx, dy in re.findall(
            r"case (DIR_\w+):\s*dx =\s*(-?\d+); dy =\s*(-?\d+);",
            source[source.index("direction_to_offset (enum direction dir"):][:900])}
        expected = [(int(bits[name]), (offsets[name][0] + offsets[name][1]) // 2, (offsets[name][0] - offsets[name][1]) // 2)
                    for name in order]
        table = re.search(r"order\[8\]=\{(.*?)\};", (ROOT / PACK).read_text()).group(1)
        renderer = [tuple(int(v) for v in t) for t in re.findall(r"\{(-?\d+),(-?\d+),(-?\d+)\}", table)]
        self.assertEqual(order, ["DIR_E", "DIR_W", "DIR_SE", "DIR_NE", "DIR_SW", "DIR_NW", "DIR_S", "DIR_N"])
        self.assertEqual(renderer, expected)

    def test_relief_slots_merge(self):
        """Constructed stamps follow the natural kits in one registry: natural slots never
        move, constructed ones continue, and overflow drops only constructed stamps (and
        their aliases)."""
        from Renderer.native.native_cpp_test import run_cpp
        header = ROOT / PACK
        run_cpp(r'''
#include <cassert>
#include <cmath>
#include "''' + header.relative_to(ROOT).as_posix() + r'''"
using P=c3x_renderer::render_core::NaturalWonderPack;
P pack(unsigned stamps,char const* prefix){
    P p;for(unsigned i=0;i<stamps;++i){P::Kit k;k.key=std::string(prefix)+std::to_string(i);k.flags=P::flag_stamp;p.kits.push_back(k);
        p.aliases.emplace_back(k.key,i);}
    return p;
}
int main(){
    P natural;assert(natural.append(pack(13,"n"))==0);
    for(unsigned i=0;i<13;++i)assert(natural.kits[i].relief_slot==i+1);
    assert(natural.append(pack(3,"wonder|c"))==1);
    assert(natural.kits.size()==15 && natural.kits[13].relief_slot==14 && natural.kits[14].relief_slot==15);
    assert(natural.find("wonder|c0") && natural.find("wonder|c1") && !natural.find("wonder|c2"));
    assert(natural.relief(14)==&natural.kits[13] && natural.relief(1)==&natural.kits[0]);
    // River frames: a south-west edge (bit 5) runs along world x at the tile's low row edge.
    float ox,oy,ax,ay;assert(P::river_frame(32,ox,oy,ax,ay));
    assert(ox==0 && oy==-.5f && std::abs(ax-1)<1e-6f && std::abs(ay)<1e-6f);
    // East (bit 2) outranks south-west, as in C3X.
    assert(P::river_frame(32|4,ox,oy,ax,ay) && ox==.5f && oy==.5f);
    assert(!P::river_frame(0,ox,oy,ax,ay));
    return 0;
}
''')

    def test_follow_river_reach_stays_in_owner_window(self):
        """Moved to its river edge, a follow-river kit's reach (the patched cap) keeps
        centre +/- reach inside the owner's 3x3 for every river bit, and the gorge field
        is already zero at that reach (no cut at the window bound)."""
        from Renderer.native.native_cpp_test import run_cpp
        from Renderer.tools.asset_compiler import build_wonder_pack as wp
        import numpy as np
        shape = (ROOT / "Renderer/lab/shared/natural/mountain_shape.h").read_text()
        self.assertIn("if(k.flags&Pack::flag_follow_river)p.reach=std::min(p.reach,1.5f-std::max(std::abs(ox),std::abs(oy)));", shape)
        header = ROOT / PACK
        run_cpp(r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include "''' + header.relative_to(ROOT).as_posix() + r'''"
using P=c3x_renderer::render_core::NaturalWonderPack;
int main(){
    for(unsigned bits=1;bits<256;++bits){
        float ox,oy,ax,ay;assert(P::river_frame(bits,ox,oy,ax,ay));
        float reach=std::min(std::min(1.06f,.40f*''' + str(wp.GORGE["span"]) + r'''f),1.5f-std::max(std::abs(ox),std::abs(oy)));
        assert(std::abs(ox)+reach<=1.5f+1e-6f && std::abs(oy)+reach<=1.5f+1e-6f);
    }
    return 0;
}
''')
        size = wp.GORGE["size"]
        height = np.frombuffer(wp.gorge_fields(size, wp.GORGE["channel"], wp.GORGE["crest"], wp.GORGE["outer"])[0],
                               np.uint8).reshape(size, size)
        axis = (np.arange(size) + .5) / size * 2 - 1
        u, v = np.meshgrid(axis, axis)
        reach = 1.0 / (wp.GORGE["span"] / 2)        # 1.0 tile in field halves
        self.assertEqual(int(height[np.sqrt(u * u + v * v) >= reach - 1.0 / size].max()), 0)
        self.assertGreater(int(height.max()), 200)

    def test_river_wonders_follow_c3x_facts(self):
        """A river wonder is C3X's buildable_on_rivers fact (wonder_flags bit 2), never pack
        or name data: the fact reads it, and only a mountain tile with it takes the one
        generic river valley the pack builds."""
        from Renderer.native.native_cpp_test import run_cpp
        from Renderer.tools.asset_compiler import build_wonder_pack as wp
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text(encoding="latin-1")
        slot = source[source.index("unsigned wonder_tile_slot(c3x_renderer_tile_v1 const& tile) const {"):][:700]
        self.assertIn(f'wonder_slot("{wp.RIVER_VALLEY}")', slot)
        self.assertIn("tile.real_terrain_type!=6", slot)
        self.assertIn("!fact.on_river", slot)
        run_cpp(r'''
#include <cassert>
#include <cstring>
#include "Renderer/native/city_fidelity/wonders.h"
using namespace c3x_renderer::city_fidelity;
int main(){
    c3x_renderer_tile_v1 tile{};WonderFact fact;
    assert(!wonder_fact(tile,fact));
    std::strcpy(tile.wonder,"Hoover Dam");tile.wonder_state=C3X_RENDERER_WONDER_COMPLETE;
    assert(wonder_fact(tile,fact) && fact.alias=="hoover dam" && !fact.on_river && !fact.alternate);
    tile.wonder_flags=C3X_RENDERER_WONDER_ON_RIVER;assert(wonder_fact(tile,fact) && fact.on_river && !fact.alternate);
    tile.wonder_flags|=C3X_RENDERER_WONDER_ALTERNATE;assert(wonder_fact(tile,fact) && fact.on_river && fact.alternate);
    return 0;
}
''')

    def test_world_wonders_prefetch_before_a_visit(self):
        """A Wonder District in world state has its group (era district, next stage)
        loaded and pinned off the frame before any tile asks, so a camera's first visit
        finds it resident; groups the world no longer shows are released, and an
        interrupted pass stays due (2026-10-25: first visits stalled 150-490 ms)."""
        from Renderer.native.native_cpp_test import run_cpp
        if not (ROOT / "Renderer/packs/WonderRuntime/index.bin").is_file():
            self.skipTest("build the wonder pack first")
        run_cpp(r'''
#include <cassert>
#include <cstring>
#include <fstream>
#include <iterator>
#include "Renderer/native/city_fidelity/wonders.h"
using namespace c3x_renderer::city_fidelity;
static bool read_file(std::string const& path,std::vector<std::uint8_t>& out){
    std::ifstream file(path,std::ios::binary);if(!file)return false;
    out.assign(std::istreambuf_iterator<char>(file),{});return true;
}
static WonderFact fact(char const* name,unsigned state,unsigned progress){
    c3x_renderer_tile_v1 tile{};std::strcpy(tile.wonder,name);tile.wonder_state=state;tile.wonder_progress=progress;
    WonderFact out;assert(wonder_fact(tile,out));return out;
}
int main(){
    WonderStore store;std::vector<std::uint8_t> index;
    assert(read_file("Renderer/packs/WonderRuntime/index.bin",index) && store.parse(index));
    std::vector<Material> materials(wonder_material_slots);
    std::vector<std::array<ID3D11ShaderResourceView*,7>> views(wonder_material_slots);
    store.slots.used.assign(wonder_material_slots,false);store.slots.materials=&materials;store.slots.views=&views;
    store.read=[](std::string const& path,std::vector<std::uint8_t>& out){return read_file(path,out);};
    store.upload=[](std::vector<std::uint8_t> const&,ID3D11ShaderResourceView*& view){view=nullptr;return true;};
    store.ready=true;
    auto const pyramids=fact("The Pyramids",C3X_RENDERER_WONDER_COMPLETE,0);
    auto const gardens=fact("The Hanging Gardens",C3X_RENDERER_WONDER_BUILDING,120);
    std::vector<std::string> keys;store.needs(pyramids,keys);store.needs(gardens,keys);
    assert(!store.resident(store.key(pyramids)) && !store.resident(store.key(gardens)));
    // The world shows them: due, then loaded and pinned with no tile holding them.
    store.want(keys);assert(store.prefetch_due.exchange(false));
    std::atomic<bool> go{false},stop{true};
    store.prefetch(go);
    assert(store.resident(store.key(pyramids)) && store.resident(store.key(gardens)) && store.resident(store.district(gardens)));
    // The same world again is not due again.
    store.want(keys);assert(!store.prefetch_due.load());
    // The Hanging Gardens leave the world: released (nothing else holds it).
    std::vector<std::string> fewer;store.needs(pyramids,fewer);store.want(fewer);
    assert(store.resident(store.key(pyramids)) && !store.resident(store.key(gardens)));
    // An interrupted pass loads nothing and stays due.
    store.want(keys);assert(store.prefetch_due.exchange(false));
    store.prefetch(stop);
    assert(!store.resident(store.key(gardens)) && store.prefetch_due.load());
    return 0;
}
''')

    def test_combined_relief_slots_fit(self):
        """The packs' relief kits (natural stamps and wave kits, then constructed stamps) fit
        the 15 slots of topology bits 28-31, with none dropped."""
        from Renderer.tools.asset_compiler import build_natural_wonder_pack as natural
        packs = ROOT / "Renderer/packs"
        if not (packs / "NaturalWonderRuntime/wonders.bin").is_file() or not (packs / "WonderRuntime/stamps.bin").is_file():
            self.skipTest("build both packs first")
        kits = natural.read_wonders((packs / "NaturalWonderRuntime/wonders.bin").read_bytes())["kits"]
        relief = sum(1 for kit in kits if (kit["flags"] & 1) or kit["waves"])   # the parser's rule: stamp or waves
        stamps = struct.unpack_from("<I", (packs / "WonderRuntime/stamps.bin").read_bytes(), 12)[0]
        self.assertLessEqual(relief + stamps, 15)


if __name__ == "__main__":
    unittest.main()
