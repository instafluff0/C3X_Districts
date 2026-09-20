"""Prepared-world backing preserves exact mesh and dependency values."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class WorldBackingTests(unittest.TestCase):
    def test_windows_compression_concurrent_roundtrip_and_retirement(self):
        run_cpp(r'''
#include <windows.h>
#include "Renderer/native/render_core/compressed_world_store.h"
#include <thread>
#include <cassert>
using Store=c3x_renderer::render_core::CompressedWorldStore<std::array<unsigned,2>>;
int main(){
 Store store;std::array<std::thread,4> lanes;
 for(unsigned lane=0;lane<4;++lane)lanes[lane]=std::thread([&,lane]{
  for(unsigned n=0;n<32;++n){
   std::vector<unsigned char> raw(32768);for(unsigned i=0;i<raw.size();++i)raw[i]=(i+lane+n)%251;
   std::array<unsigned,2> key{lane,n};assert(store.put(key,raw));assert(store.get(key)==raw);
   assert(store.put(key,raw));assert(store.contains(key));
  }
 });
 for(auto& lane:lanes)lane.join();auto stats=store.statistics();
 assert(stats.records==128 && stats.writes==128 && stats.reads==128);
 assert(stats.bytes<stats.raw && stats.raw==128*32768);
 store.invalidate({0,0});assert(!store.contains({0,0}));
 assert(store.put({0,0},{9,8,7}) && store.get({0,0})==std::vector<unsigned char>({9,8,7}));
 assert(store.get({8,9}).empty());
 store.clear();assert(!store.contains({0,0}) && store.get({0,0}).empty());
 assert(!store.statistics().bytes && !store.statistics().records);
 assert(store.put({0,0},{1,2,3}) && store.get({0,0})==std::vector<unsigned char>({1,2,3}));
 assert(!store.put({1,1},{}));
}
''')

    def test_codec_retains_payload_proofs_and_discards_runtime_receipts(self):
        run_cpp(r'''
using UINT=unsigned;
#include "Renderer/native/world_backing_codec.h"
#include <cassert>
using namespace c3x_renderer;
int main(){
 PreparedWorld world;world.ground=std::make_unique<fidelity::PreparedGround>();
 world.terrain=std::make_unique<fidelity::TerrainSurfaces>();world.objects=std::make_unique<objects::PreparedObjects>();
 auto& g=*world.ground;auto& t=*world.terrain;auto& o=*world.objects;
 render_core::PreparedMesh mesh;mesh.vertex_stride=92;mesh.index_stride=2;mesh.index_count=3;
 mesh.vertices.resize(276);mesh.indices={0,0,1,0,2,0};for(unsigned i=0;i<mesh.vertices.size();++i)mesh.vertices[i]=i%251;
 mesh.world_low={1,2,3};mesh.world_high={4,5,6};mesh.bounds={-7,8,9,10};mesh.projected_bounds.include(7,8,9);
 g.meshes[0]=mesh;t.meshes[2]=mesh;o.layers[1].mesh=mesh;
 g.water_coverage=true;g.world={{5,42},{8,17}};g.coast={{3,99}};g.topology={{6,72}};
 t.world=g.world;t.coast=g.coast;o.world=g.world;o.coast=g.coast;o.topology=g.topology;
 auto cell=std::make_shared<fidelity::NaturalWorld::CellContent>();cell->values={1,7,99};
 cell->inputs=std::make_shared<fidelity::NaturalWorld::PageInputs>();cell->inputs->values={{18,7},{25,9}};cell->inputs->flow={2,3};
 cell->inputs->current=true;cell->inputs->checked_revision=123;
 fidelity::NaturalWorld::CellKey key{1,2,3,4};g.rivers.emplace(key,cell);t.rivers.emplace_back(key,cell);o.rivers=t.rivers;
 o.city.resize(2);auto light=std::make_shared<city_fidelity::Lighting>();light->lights.resize(1);light->lights[0].owner=9;
 light->blockers.push_back({{1,2,3,4},{5,6,7,8}});
 for(auto& p:o.city){p.mesh=mesh;p.material=42;p.environment=true;p.terrain_conforming=true;p.atlas={1,2,3,4};p.lighting=light;}
 o.composition=7;o.instances=11;o.routes=3;
 auto bytes=WorldBackingCodec::encode(world);auto copied=WorldBackingCodec::decode(bytes);assert(copied);
 assert(copied->ground->meshes[0].vertices==mesh.vertices && copied->terrain->meshes[2].indices==mesh.indices);
 assert(copied->ground->world==g.world && copied->objects->coast==o.coast && copied->objects->topology==o.topology);
 assert(copied->ground->water_coverage && copied->objects->composition==7 && copied->objects->instances==11 && copied->objects->routes==3);
 assert(copied->objects->city[1].lighting==copied->objects->city[0].lighting && copied->objects->city[0].lighting!=light);
 assert(copied->objects->city[1].lighting->lights[0].owner==9 && copied->objects->city[1].atlas==o.city[1].atlas);
 auto const& proof=copied->terrain->rivers.front();assert(proof.first==key && proof.second->values==cell->values);
 assert(proof.second->inputs->values==cell->inputs->values && proof.second->inputs->flow==cell->inputs->flow);
 assert(!proof.second->inputs->checked_world && proof.second->inputs->checked_revision==-1 && !proof.second->inputs->current);
 assert(!copied->buffer && !copied->objects->buffer && !copied->terrain->vertex_buffer && !copied->gpu_bytes);
 for(std::size_t size=0;size<bytes.size();size+=31){auto truncated=bytes;truncated.resize(size);assert(!WorldBackingCodec::decode(truncated));}
 auto invalid=bytes;invalid[0]=99;assert(!WorldBackingCodec::decode(invalid));
 invalid=bytes;invalid.push_back(1);assert(!WorldBackingCodec::decode(invalid));
 g.legacy_shadow.resize(1);assert(WorldBackingCodec::encode(world).empty());
}
''')


if __name__ == '__main__':
    unittest.main()
