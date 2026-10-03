"""Prepared-world RAM backing preserves values under pressure and retirement."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class WorldBackingTests(unittest.TestCase):
    def test_portable_ram_capacity_unique_publication_and_key_ownership(self):
        run_cpp(r'''
#include "Renderer/native/render_core/compressed_world_store.h"
#include <cassert>
#include <thread>
using namespace c3x_renderer::render_core;
struct Codec {
 bool encode(std::vector<unsigned char> const& raw,WorldStoreBuffer& out){out.resize(raw.size());out.bytes=raw;return true;}
 bool decode(std::vector<unsigned char> const& packed,std::vector<unsigned char>& raw,WorldStoreMemory&){raw=packed;return true;}
};
struct Key {unsigned id;std::vector<std::uint64_t> recipe;bool operator<(Key const& b)const{return id<b.id;}};
int main(){
 using Store=CompressedWorldStore<unsigned,Codec>;
 Store store(256*1024);auto empty_owner=store.statistics().resident_bytes;assert(empty_owner<=1024);
 std::array<std::thread,4> lanes;
 for(unsigned lane=0;lane<4;++lane)lanes[lane]=std::thread([&,lane]{
  for(unsigned n=0;n<32;++n){std::vector<unsigned char> raw(1024,unsigned(lane*32+n));
   auto key=lane*32+n;assert(store.put(key,raw));assert(store.put(key,raw));assert(store.get(key)==raw);
   assert(store.put(999,std::vector<unsigned char>(1024,7)));}
 });
 for(auto& lane:lanes)lane.join();auto full=store.statistics();
 assert(full.records==129 && full.writes==129 && full.reads==128 && !full.in_flight_bytes);
 assert(full.live==129*1024 && full.resident_bytes>full.live && full.resident_bytes<=full.limit);
 assert(full.peak_bytes>=full.resident_bytes && !full.generated_world_file_reads && !full.generated_world_file_writes);
 assert(!store.configure(full.resident_bytes-1) && store.statistics().records==129);
 auto frozen=store.statistics();assert(frozen.capacity_pressure && frozen.limit==frozen.resident_bytes);
 assert(!store.put(1000,std::vector<unsigned char>(1024,19)) && store.contains(999));
 assert(store.configure(2ull*1024*1024*1024)); // No hidden one-GiB policy cap.
 assert(!store.statistics().capacity_pressure);
 unsigned visited=0;store.inspect_keys([&](unsigned key){assert(key<128 || key==999);++visited;});assert(visited==129);
 store.clear();assert(store.statistics().resident_bytes==empty_owner && !store.statistics().in_flight_bytes);
 Store pressured(12000);std::vector<unsigned char> bytes(2048,17);unsigned admitted=0;
 while(admitted<32 && pressured.put(admitted,bytes))++admitted;
 assert(admitted>=3 && admitted<=5 && !pressured.put(100,bytes));
 auto refused=pressured.statistics();assert(refused.capacity_refusals && refused.records==admitted && refused.resident_bytes<=refused.limit);
 for(unsigned key=0;key<admitted;++key)assert(pressured.get(key)==bytes); // Refusal never evicts useful recipes.
 pressured.invalidate(0);assert(pressured.put(100,bytes) && pressured.get(100)==bytes);
 Store race(50000);assert(race.put(1,bytes));auto sampled=race.statistics();
 std::thread publisher([&]{assert(race.put(2,bytes));});publisher.join();
 assert(!race.configure(sampled.resident_bytes));auto current=race.statistics();
 assert(current.capacity_pressure && current.limit==current.resident_bytes && current.limit>sampled.resident_bytes);
 assert(!race.put(3,bytes) && race.get(1)==bytes && race.get(2)==bytes);
 CompressedWorldStore<Key,Codec> owned(16000);auto empty_owned=owned.statistics().resident_bytes;assert(empty_owned<=1024);
 assert(owned.configure(16000,[](Key const& key){return key.recipe.capacity()*sizeof(std::uint64_t);}));
 Key key{4,std::vector<std::uint64_t>(128,9)};assert(owned.put(key,bytes));
 assert(owned.configure(32000,[](Key const& key){return key.recipe.capacity()*sizeof(std::uint64_t);}));
 auto used=owned.statistics();assert(used.resident_bytes>=bytes.size()+key.recipe.size()*sizeof(std::uint64_t));
 key.recipe.clear();key.recipe.shrink_to_fit();assert(owned.get(Key{4,{}})==bytes);
 bool inspected=false;owned.inspect_keys([&](Key const& retained){assert(retained.recipe.size()==128 && retained.recipe[127]==9);inspected=true;});
 assert(inspected);owned.clear();assert(owned.statistics().resident_bytes==empty_owned);
}
''')

    def test_portable_compact_capacity_excludes_codec_reservation(self):
        run_cpp(r'''
#include "Renderer/native/render_core/compressed_world_store.h"
#include <cassert>
using namespace c3x_renderer::render_core;
struct Codec {
 bool encode(std::vector<unsigned char> const& raw,WorldStoreBuffer& out){
  if(raw.size()!=32768)return false;
  for(unsigned i=256;i<raw.size();++i)if(raw[i]!=raw[i%256])return false;
  out.resize(raw.size()+512);std::copy(raw.begin(),raw.begin()+256,out.bytes.begin());out.bytes.resize(256);return true;}
 bool decode(std::vector<unsigned char> const& packed,std::vector<unsigned char>& raw,WorldStoreMemory&){
  if(packed.size()!=256)return false;for(unsigned i=0;i<raw.size();++i)raw[i]=packed[i%256];return true;}
};
int main(){
 CompressedWorldStore<unsigned,Codec> store(4096);auto empty_owner=store.statistics().resident_bytes;assert(empty_owner<=1024);
 std::vector<unsigned char> raw(32768);
 for(unsigned key=0;key<8;++key){for(unsigned i=0;i<raw.size();++i)raw[i]=(i+key)%256;
  assert(store.put(key,raw) && store.get(key)==raw);}
 auto packed=store.statistics();assert(packed.records==8 && packed.raw==8*32768 && packed.live==8*256);
 assert(packed.live<packed.raw && packed.resident_bytes<=4096 && !packed.in_flight_bytes);
 // The old reservation and compact copy coexist before publication.
 assert(packed.peak_bytes>=empty_owner+32768+512+256);
 for(unsigned key=0;key<8;++key){for(unsigned i=0;i<raw.size();++i)raw[i]=(i+key)%256;assert(store.get(key)==raw);}
 store.clear();auto cleared=store.statistics();assert(cleared.resident_bytes==empty_owner && !cleared.in_flight_bytes && !cleared.live);
}
''')

    def test_portable_ram_read_pin_survives_invalidate_and_reset(self):
        run_cpp(r'''
#include "Renderer/native/render_core/compressed_world_store.h"
#include <cassert>
#include <thread>
#include <condition_variable>
using namespace c3x_renderer::render_core;
struct Codec {
 std::mutex mutex;std::condition_variable cv;bool entered=false,released=false;
 bool encode(std::vector<unsigned char> const& raw,WorldStoreBuffer& out){out.resize(raw.size());out.bytes=raw;return true;}
 bool decode(std::vector<unsigned char> const& packed,std::vector<unsigned char>& raw,WorldStoreMemory&){
  std::unique_lock<std::mutex> lock(mutex);entered=true;cv.notify_all();cv.wait(lock,[&]{return released;});raw=packed;return true;}
};
int main(){
 auto codec=std::make_shared<Codec>();CompressedWorldStore<unsigned,Codec> store(16000,codec);
 auto empty_owner=store.statistics().resident_bytes;assert(empty_owner<=1024);
 std::vector<unsigned char> bytes(4096,31),restored;assert(store.put(1,bytes));
 std::thread reader([&]{restored=store.get(1);});
 {std::unique_lock<std::mutex> lock(codec->mutex);assert(codec->cv.wait_for(lock,std::chrono::seconds(5),[&]{return codec->entered;}));}
 store.invalidate(1);auto pinned=store.statistics();
 assert(!pinned.records && !pinned.live && !pinned.raw && pinned.pinned_bytes==4096);
 assert(pinned.resident_bytes>=4096 && pinned.in_flight_bytes>=4096 && pinned.peak_bytes>=8192);
 assert(!store.configure(0));auto frozen=store.statistics();
 assert(frozen.capacity_pressure && frozen.limit==frozen.resident_bytes && frozen.pinned_bytes==4096);
 store.clear();auto reset=store.statistics();assert(!reset.records && reset.pinned_bytes==4096 && reset.in_flight_bytes>=4096);
 {std::lock_guard<std::mutex> lock(codec->mutex);codec->released=true;}codec->cv.notify_all();reader.join();
 assert(restored==bytes);auto retired=store.statistics();
 assert(retired.resident_bytes==empty_owner && !retired.in_flight_bytes && !retired.pinned_bytes && !retired.reads);
 assert(!retired.generated_world_file_reads && !retired.generated_world_file_writes);
}
''')

    def test_portable_ram_reset_rejects_old_publish_and_optional_lock_skips(self):
        run_cpp(r'''
#include "Renderer/native/render_core/compressed_world_store.h"
#include <cassert>
#include <thread>
#include <condition_variable>
using namespace c3x_renderer::render_core;
struct Codec {
 std::mutex mutex;std::condition_variable cv;bool block=false,entered=false,released=false,corrupt=false;
 bool encode(std::vector<unsigned char> const& raw,WorldStoreBuffer& out){
  out.resize(raw.size());out.bytes=raw;std::unique_lock<std::mutex> lock(mutex);
  if(block){entered=true;cv.notify_all();cv.wait(lock,[&]{return released;});}return true;}
 bool decode(std::vector<unsigned char> const& packed,std::vector<unsigned char>& raw,WorldStoreMemory&){raw=packed;if(corrupt)raw[0]^=1;return true;}
};
int main(){
 auto codec=std::make_shared<Codec>();CompressedWorldStore<unsigned,Codec> store(16000,codec);
 auto empty_owner=store.statistics().resident_bytes;assert(empty_owner<=1024);
 std::vector<unsigned char> bytes(4096,41);codec->block=true;bool published=true;
 std::thread old([&]{published=store.put(1,bytes);});
 {std::unique_lock<std::mutex> lock(codec->mutex);assert(codec->cv.wait_for(lock,std::chrono::seconds(5),[&]{return codec->entered;}));}
 assert(store.statistics().in_flight_bytes>=4096);store.clear();assert(!store.contains(1));
 {std::lock_guard<std::mutex> lock(codec->mutex);codec->released=true;}codec->cv.notify_all();old.join();
 assert(!published && store.statistics().resident_bytes==empty_owner && !store.statistics().in_flight_bytes);
 auto fresh=std::make_shared<Codec>();CompressedWorldStore<unsigned,Codec> optional(16000,fresh);
 auto empty_optional=optional.statistics().resident_bytes;assert(empty_optional<=1024);
 assert(optional.put(2,bytes));fresh->corrupt=true;assert(optional.get(2).empty());fresh->corrupt=false;
 std::mutex gate;std::condition_variable cv;bool entered=false,released=false;
 std::thread inspector([&]{optional.inspect_keys([&](unsigned key){assert(key==2);
  std::unique_lock<std::mutex> lock(gate);entered=true;cv.notify_all();cv.wait(lock,[&]{return released;});});});
 {std::unique_lock<std::mutex> lock(gate);assert(cv.wait_for(lock,std::chrono::seconds(5),[&]{return entered;}));}
 auto begin=std::chrono::steady_clock::now();assert(!optional.put_optional(3,bytes));
 assert(std::chrono::steady_clock::now()-begin<std::chrono::seconds(1));
 {std::lock_guard<std::mutex> lock(gate);released=true;}cv.notify_all();inspector.join();
 std::atomic<bool> stop{true};assert(!optional.put_optional(3,bytes,&stop));
 assert(!optional.put_optional(3,{}));assert(optional.statistics().optional_skips==3);
 stop=false;assert(optional.put_optional(3,bytes,&stop));auto saved=optional.statistics();
 assert(saved.optional_writes==1 && saved.records==2 && optional.get(2)==bytes && optional.get(3)==bytes);
 optional.clear();assert(optional.statistics().resident_bytes==empty_optional && !optional.statistics().in_flight_bytes);
}
''')

    def test_portable_allocation_reuses_generations_and_recovers_fragmentation(self):
        run_cpp(r'''
#include "Renderer/native/render_core/backing_allocation.h"
#include <cassert>
#include <map>
using c3x_renderer::render_core::BackingAllocation;
int main(){
 BackingAllocation a(100);std::uint64_t offsets[5]{};
 for(auto& offset:offsets)assert(a.allocate(20,offset));
 assert(a.high_water()==100 && a.live_bytes()==100);
 a.release(offsets[0],20);a.release(offsets[2],20);
 std::uint64_t p=999;assert(!a.allocate(30,p)); // Enough free bytes, fragmented.
 assert(a.reset_layout({{0,20},{20,20},{40,20}}));
 assert(a.high_water()==60 && a.live_bytes()==60 && a.allocate(30,p) && p==60);
 assert(!a.reset_layout({{0,30},{20,30}})); // Overlap cannot replace a valid layout.
 assert(a.high_water()==90 && a.live_bytes()==90);
 a.release(60,30);a.release(40,20);a.release(20,20);a.release(0,20);
 assert(!a.high_water() && !a.live_bytes());
 assert(a.allocate(100,p) && p==0);a.release(p,100);
 assert(!a.allocate(101,p) && !a.allocate(0,p));
 BackingAllocation ring(1024);std::map<unsigned,std::pair<std::uint64_t,std::uint64_t>> records;
 std::uint32_t random=57;
 for(unsigned sweep=0;sweep<1000;++sweep){
  for(unsigned key=0;key<16;++key){
   auto found=records.find(key);if(found!=records.end()){ring.release(found->second.first,found->second.second);records.erase(found);}
   random=random*1664525u+1013904223u;auto size=std::uint64_t(16+(random%32));
   std::uint64_t offset=0;
   if(!ring.allocate(size,offset)){
    std::uint64_t cursor=0;std::vector<std::pair<std::uint64_t,std::uint64_t>> compacted;
    for(auto& item:records){item.second.first=cursor;compacted.emplace_back(cursor,item.second.second);cursor+=item.second.second;}
    assert(ring.reset_layout(compacted) && ring.allocate(size,offset));
   }
   records.emplace(key,std::make_pair(offset,size));
   std::vector<std::pair<std::uint64_t,std::uint64_t>> ranges;std::uint64_t live=0;
   for(auto const& item:records){ranges.push_back(item.second);live+=item.second.second;}
   std::sort(ranges.begin(),ranges.end());std::uint64_t end=0;
   for(auto const& range:ranges){assert(range.first>=end);end=range.first+range.second;}
   assert(end<=ring.high_water() && ring.high_water()<=1024 && ring.live_bytes()==live);
  }
 }
 for(auto const& item:records)ring.release(item.second.first,item.second.second);
 assert(!ring.high_water() && !ring.live_bytes());
}
''')

    def test_windows_ram_capacity_and_generation_plateau(self):
        run_cpp(r'''
#include <windows.h>
#include "Renderer/native/render_core/compressed_world_store.h"
#include <cassert>
using Store=c3x_renderer::render_core::CompressedWorldStore<unsigned>;
int main(){
 auto noise=[](unsigned n,unsigned seed){std::vector<unsigned char> v(n);
  for(auto& b:v){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;b=seed>>24;}return v;};
 Store store(30000);auto empty_owner=store.statistics().resident_bytes;assert(empty_owner<=1024);
 std::vector<std::vector<unsigned char>> raw;
 for(unsigned i=0;i<16;++i){auto bytes=noise(4096,99+i);
  if(!store.put(i,bytes))break;raw.push_back(bytes);}
 assert(raw.size()>=5 && raw.size()<=7);
 for(unsigned i=0;i+1<raw.size();i+=2)store.invalidate(i);
 auto big=noise(8192,171);assert(store.put(100,big));
 auto admitted=store.statistics();assert(!admitted.compactions && admitted.resident_bytes<=admitted.limit);
 assert(!admitted.generated_world_file_reads && !admitted.generated_world_file_writes);
 assert(store.get(100)==big);
 for(unsigned i=1;i<raw.size();i+=2)assert(store.get(i)==raw[i]);
 assert(store.get(unsigned(raw.size()-1))==raw.back());
 for(unsigned generation=0;generation<500;++generation){
  store.invalidate(100);big=noise(8192,171+generation);assert(store.put(100,big));assert(store.get(100)==big);
  auto s=store.statistics();assert(s.bytes<=30000 && s.live<=s.bytes && s.limit==30000);
 }
 for(unsigned i=0;i<raw.size();++i)store.invalidate(i);store.invalidate(100);
 auto retired=store.statistics();assert(!retired.records && !retired.raw && !retired.live && !retired.in_flight_bytes);
 store.clear();assert(store.statistics().resident_bytes==empty_owner);
 assert(store.put(7,noise(16000,887)) && store.get(7)==noise(16000,887));
}
''')

    def test_windows_optional_ram_pressure_and_cancellation(self):
        run_cpp(r"""
#include <windows.h>
#include "Renderer/native/render_core/compressed_world_store.h"
#include <cassert>
using Store=c3x_renderer::render_core::CompressedWorldStore<unsigned>;
int main(){
 auto noise=[](unsigned n,unsigned seed){std::vector<unsigned char> raw(n);
  for(auto& b:raw){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;b=seed>>24;}return raw;};
 Store store(30000);auto empty_owner=store.statistics().resident_bytes;assert(empty_owner<=1024);unsigned count=0;
 for(;count<16;++count)if(!store.put(count,noise(4096,99+count)))break;
 assert(count>=5 && count<=7);
 auto big=noise(8192,171);auto before=store.statistics();
 assert(!store.put_optional(100,big));auto skipped=store.statistics();
 assert(skipped.compactions==before.compactions && skipped.writes==before.writes && !store.contains(100));
 std::atomic<bool> stop{true};assert(!store.put_optional(101,noise(512,123),&stop));
 for(unsigned i=0;i+1<count;i+=2)store.invalidate(i);
 // Freed RAM is reusable without moving or evicting another recipe.
 assert(store.put(100,big) && store.get(100)==big);
 auto admitted=store.statistics();assert(!admitted.compactions && admitted.capacity_refusals);
 stop=false;auto small_blob=noise(512,123);assert(store.put_optional(101,small_blob,&stop));
 assert(store.get(101)==small_blob);auto written=store.statistics();
 assert(written.optional_writes==1 && written.optional_skips==2 && !written.compactions);
 assert(!written.generated_world_file_reads && !written.generated_world_file_writes);
 for(unsigned i=1;i<count;i+=2)assert(store.get(i)==noise(4096,99+i));
 store.clear();auto empty=store.statistics();assert(empty.bytes==empty_owner && !empty.optional_writes && !empty.optional_skips);
}
""")

    def test_windows_compression_concurrent_roundtrip_and_retirement(self):
        run_cpp(r'''
#include <windows.h>
#include "Renderer/native/render_core/compressed_world_store.h"
#include <thread>
#include <cassert>
using Store=c3x_renderer::render_core::CompressedWorldStore<std::array<unsigned,2>>;
int main(){
 Store store;auto empty_owner=store.statistics().resident_bytes;assert(empty_owner<=1024);std::array<std::thread,4> lanes;
 for(unsigned lane=0;lane<4;++lane)lanes[lane]=std::thread([&,lane]{
  for(unsigned n=0;n<32;++n){
   std::vector<unsigned char> raw(32768);for(unsigned i=0;i<raw.size();++i)raw[i]=(i+lane+n)%251;
   std::array<unsigned,2> key{lane,n};assert(store.put(key,raw));assert(store.get(key)==raw);
   assert(store.put(key,raw));assert(store.contains(key));
  }
 });
 for(auto& lane:lanes)lane.join();auto stats=store.statistics();
 assert(stats.records==128 && stats.writes==128 && stats.reads==128);
 assert(stats.live<stats.raw && stats.bytes<stats.raw && stats.raw==128*32768);
 assert(stats.resident_bytes==stats.bytes && stats.live<=stats.resident_bytes && !stats.in_flight_bytes);
 assert(stats.peak_bytes>=stats.resident_bytes && !stats.generated_world_file_reads && !stats.generated_world_file_writes);
 store.invalidate({0,0});assert(!store.contains({0,0}));
 assert(store.put({0,0},{9,8,7}) && store.get({0,0})==std::vector<unsigned char>({9,8,7}));
 assert(store.get({8,9}).empty());
 store.clear();assert(!store.contains({0,0}) && store.get({0,0}).empty());
 assert(store.statistics().bytes==empty_owner && !store.statistics().records);
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
 o.rigid.resize(1);o.rigid[0].family=objects::mine_family;o.rigid[0].asset=2;o.rigid[0].layer=objects::mine_layer;
 o.rigid[0].instance.place[7]=12.5f;o.rigid[0].bounds={1,2,3,4};o.rigid[0].material=21.18f;
 o.draws={{objects::feature_layer,0,3,~0u},{objects::mine_layer,0,0,0}};
 auto bytes=WorldBackingCodec::encode(world);auto copied=WorldBackingCodec::decode(bytes);assert(copied);
 assert(copied->ground->meshes[0].vertices==mesh.vertices && copied->terrain->meshes[2].indices==mesh.indices);
 assert(copied->ground->world==g.world && copied->objects->coast==o.coast && copied->objects->topology==o.topology);
 assert(copied->ground->water_coverage && copied->objects->composition==7 && copied->objects->instances==11 && copied->objects->routes==3);
 assert(copied->objects->rigid.size()==1 && !std::memcmp(&copied->objects->rigid[0],&o.rigid[0],sizeof(objects::PreparedRigid)));
 assert(copied->objects->draws.size()==2 && copied->objects->draws[0].count==3 && copied->objects->draws[1].rigid==0);
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
