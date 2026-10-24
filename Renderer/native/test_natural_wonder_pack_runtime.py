"""The renderer reads NaturalWonderRuntime/wonders.bin exactly as the pack compiler writes it."""
import struct
import types
import unittest

import numpy as np

from Renderer.native.native_cpp_test import run_cpp
from Renderer.tools.asset_compiler import build_natural_wonder_pack as pack
from Renderer.tools.asset_compiler import build_resource_compositions as rc


def sample_pack() -> bytes:
    kits = types.SimpleNamespace(PICKS={"peak": {}, "grove": {}, "bay": {"winter_wet": 24.0}}, SNOW_CAPPED={"peak"})
    field = np.linspace(0, 1, 16).reshape(4, 4)
    stamp = {"field": field, "ground": field * .5, "special": 1 - field, "span": 2.5, "units": 90.0,
             "turn": .75, "kind": 4, "effects": [[4.0] + [0.25] * 9, [6.0] + [0.5] * 9]}
    records = [pack.kit_record("peak", {0: {"stamp": stamp, "waves": []}}, [], kits),
               pack.kit_record("grove", {0: {"stamp": None, "waves": []}}, [("NW_grove", 15)], kits),
               pack.kit_record("bay", {0: {"stamp": None, "waves": [0.1, 0.2, 0.0, 0.5, 2.3]}},
                               [("NW_bay~11", 11), ("NW_bay~4", 4)], kits)]
    data = bytearray(pack.MAGIC) + struct.pack("<IIII", pack.VERSION, 3, 2, rc.TEXTURE_SLOTS)
    for season in pack.SEASONS:
        data += struct.pack(f"<{rc.TEXTURE_SLOTS}I", *[8 + season if slot == 1 and season in (1, 3) else slot
                                                        for slot in range(rc.TEXTURE_SLOTS)])
    data += b"".join(records)
    for name, index in (("Great Peak", 0), ("Sea Bay", 2)):
        data += rc.bundle_string(name) + struct.pack("<I", index)
    return bytes(data)


class NaturalWonderPackRuntimeTests(unittest.TestCase):
    def test_parse_aliases_relief_slots_and_rejections(self):
        data = sample_pack()
        self.assertEqual(len(pack.read_wonders(data)["kits"]), 3)        # the compiler's own reader agrees
        array = ",".join(str(b) for b in data)
        run_cpp(r'''
#include <cassert>
#include <vector>
#include "Renderer/native/render_core/natural_wonder_pack.h"
int main(){
 using c3x_renderer::render_core::NaturalWonderPack;
 std::vector<std::uint8_t> data={''' + array + r'''};
 NaturalWonderPack p;assert(p.parse(data.data(),data.size()));
 assert(p.kits.size()==3 && p.aliases.size()==2);
 auto const* peak=p.find("great PEAK");assert(peak && peak->key=="peak" && peak->stamp());
 assert(peak->flags==(NaturalWonderPack::flag_stamp|NaturalWonderPack::flag_snow) && peak->region_kind==4);
 assert(peak->field_size==4 && peak->height.size()==16 && peak->height[15]==255 && peak->special[15]==0);
 assert(peak->effects.size()==20 && peak->effects[0]==4.f && peak->effects[10]==9.f);   // spray left kind 6
 // Relief slots: stamped or collared kits in pack order; the grove has neither.
 assert(peak->relief_slot==1 && p.relief(1)==peak);
 auto const* bay=p.find("Sea Bay");assert(bay && bay->relief_slot==2 && p.relief(2)==bay && bay->waves.size()==5);
 assert(bay->snow_floor==24.f && bay->bodies.size()==2 && bay->bodies[1].second==4);
 assert(p.kits[1].relief_slot==0 && !p.relief(3) && !p.relief(0) && !p.find("Great Peaks") && !p.find(""));
 assert(p.season_slots[0][1]==1 && p.season_slots[1][1]==9 && p.season_slots[3][1]==11);
 // Trailing bytes, truncation and a foreign effect kind are refused whole.
 auto longer=data;longer.push_back(0);assert(!p.parse(longer.data(),longer.size()) && p.kits.empty());
 assert(!p.parse(data.data(),data.size()-1));
 auto broken=data;bool found=false;
 for(std::size_t i=0;i+4<=broken.size();++i){float v;std::memcpy(&v,&broken[i],4);
  if(v==9.f){v=6.f;std::memcpy(&broken[i],&v,4);found=true;break;}}
 assert(found && !p.parse(broken.data(),broken.size()));
}
''')


if __name__ == "__main__":
    unittest.main()
