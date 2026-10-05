"""Shadow caster changes are journaled for retained-raster repair.

Retained rasters bake the shadow field they sampled. A reveal can publish
neighbor casters a moment after a repair drew the old field, and contributor
proofs cannot see that. The production refresh journals the screen footprints
of casters that entered or left; re-borrowing an unchanged set after a
geometry retirement journals nothing; a wholesale replacement raises the floor
so older rasters refresh completely.
"""
import unittest
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


def method(source, header):
    start = source.index(header)
    depth = 0
    for i in range(source.index('{', start), len(source)):
        if source[i] == '{':
            depth += 1
        elif source[i] == '}':
            depth -= 1
            if depth == 0:
                return source[start:i + 1]
    raise AssertionError(header)


class ShadowChangeJournalTests(unittest.TestCase):
    def test_refresh_journals_entering_and_leaving_footprints(self):
        source = (ROOT / 'Renderer/sandbox/fresh_pipeline.h').read_text()
        refresh = method(source, '    bool refresh_casters(std::uint64_t scene) {')
        run_cpp(r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <unordered_set>
#include <vector>
struct ID3D11Buffer;
enum DXGI_FORMAT{DXGI_FORMAT_UNKNOWN};using UINT=unsigned;
struct Shadow{struct Caster{int id=0;std::array<int,4> source{};};};
struct AtlasInputs{using Key=std::array<std::uint64_t,2>;
 struct Hash{std::size_t operator()(Key const& k)const{return std::size_t(k[0]*31+k[1]);}};};
struct Harness{
 struct{struct{void IASetVertexBuffers(int,int,ID3D11Buffer**,UINT*,UINT*){}void IASetIndexBuffer(ID3D11Buffer*,DXGI_FORMAT,int){}}context_value;
  decltype(context_value)* context=&context_value;
  struct{int publish(){return 1;}}geometry_vertex_buffers;
  std::vector<Shadow::Caster> world;
  template<class View>void collect_shadow_casters(View&,std::vector<Shadow::Caster>& out){out=world;}}renderer;
 std::vector<Shadow::Caster> caster_inputs;std::vector<AtlasInputs::Key> input_keys;std::vector<std::array<int,4>> input_sources;
 struct ShadowChange{std::uint64_t serial=0;std::array<int,4> source{};};
 std::vector<ShadowChange> shadow_changes;std::uint64_t change_serial=0,change_floor=0;
 std::uint64_t caster_signature=~std::uint64_t(0),caster_collections=0,caster_reuses=0;int caster_lease=0;
 AtlasInputs::Key caster_key(Shadow::Caster const& c)const{return {std::uint64_t(c.id),std::uint64_t(c.source[0])};}
''' + refresh + r'''
};
Shadow::Caster caster(int id,int x){return {id,{x,0,x+128,64}};}
int main(){
 Harness h;
 // Initial field: one wrapped copy shares its footprint with the original.
 h.renderer.world={caster(1,0),caster(2,128),caster(3,256),caster(13,256)};
 assert(h.refresh_casters(1)&&h.change_serial==1&&h.shadow_changes.size()==3&&h.change_floor==0);
 // Geometry retirement clears borrowed casters, not their value keys.
 h.caster_inputs.clear();
 assert(h.refresh_casters(2)&&h.change_serial==1&&h.caster_inputs.size()==4&&h.shadow_changes.size()==3);
 // Caster 2 leaves and caster 4 enters elsewhere: both footprints repair.
 h.renderer.world={caster(1,0),caster(4,512),caster(3,256),caster(13,256)};
 assert(h.refresh_casters(3)&&h.change_serial==2&&h.shadow_changes.size()==5);
 assert(h.shadow_changes[3].serial==2&&h.shadow_changes[4].serial==2);
 std::vector<int> lefts={h.shadow_changes[3].source[0],h.shadow_changes[4].source[0]};std::sort(lefts.begin(),lefts.end());
 assert((lefts==std::vector<int>{128,512}));
 // Unchanged set: a reuse, no serial.
 assert(h.refresh_casters(4)&&h.change_serial==2&&h.caster_reuses==1);
 // A wholesale replacement raises the floor instead of journaling.
 h.renderer.world.clear();for(int i=0;i<20000;++i)h.renderer.world.push_back(caster(100+i,i*8));
 assert(h.refresh_casters(5)&&h.change_serial==3&&h.change_floor==3&&h.shadow_changes.size()==5);
 std::printf("PASS shadow change journal: entering_leaving=2 reborrow_silent=1 wholesale_floor=1\n");
}
''')


if __name__ == '__main__':
    unittest.main()
