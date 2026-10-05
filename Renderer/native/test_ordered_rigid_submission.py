"""Compile the production ordered rigid packet owner with observable D3D resources."""
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.test_fresh_shared_submission import method


STUB = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>
using UINT=unsigned;
constexpr unsigned D3D11_USAGE_IMMUTABLE=1,D3D11_USAGE_DEFAULT=2,
 D3D11_BIND_VERTEX_BUFFER=1,D3D11_BIND_INDEX_BUFFER=2,D3D11_BIND_SHADER_RESOURCE=4,
 D3D11_RESOURCE_MISC_BUFFER_STRUCTURED=8,DXGI_FORMAT_R32_UINT=1;
bool FAILED(int value){return value<0;}
struct D3D11_BUFFER_DESC {unsigned ByteWidth=0,Usage=0,BindFlags=0,MiscFlags=0,StructureByteStride=0;};
struct D3D11_SUBRESOURCE_DATA {void const* pSysMem=nullptr;};
struct D3D11_BOX {unsigned left,top,front,right,bottom,back;};
unsigned live_buffers=0,live_views=0;
struct ID3D11Buffer {
 std::vector<unsigned char> bytes;
 void Release(){assert(live_buffers);--live_buffers;delete this;}
};
struct ID3D11ShaderResourceView {
 ID3D11Buffer* buffer=nullptr;
 void Release(){assert(live_views);--live_views;delete this;}
};
struct ID3D11InputLayout{};struct ID3D11VertexShader{};
struct ID3D11Device {
 unsigned creates=0,fail_at=0;
 int CreateBuffer(D3D11_BUFFER_DESC const* d,D3D11_SUBRESOURCE_DATA const* data,ID3D11Buffer** b){
  if(++creates==fail_at)return -1;
  *b=new ID3D11Buffer;(*b)->bytes.resize(d->ByteWidth);++live_buffers;
  if(data)std::memcpy((*b)->bytes.data(),data->pSysMem,d->ByteWidth);return 0;
 }
 int CreateShaderResourceView(ID3D11Buffer* b,void*,ID3D11ShaderResourceView** v){
  if(++creates==fail_at)return -1;
  *v=new ID3D11ShaderResourceView;(*v)->buffer=b;++live_views;return 0;
 }
};
struct Receipt {
 std::array<unsigned char,32> vertex{};std::array<unsigned char,64> placement{};
 bool operator==(Receipt const& other)const{return vertex==other.vertex&&placement==other.placement;}
};
struct ID3D11DeviceContext {
 ID3D11Buffer* geometry=nullptr;ID3D11ShaderResourceView* placements=nullptr;
 unsigned index_offset=0,selection_offset=0,draws=0,copies=0;
 std::vector<Receipt> primitives;
 void CopySubresourceRegion(ID3D11Buffer* dst,unsigned,unsigned offset,unsigned,unsigned,ID3D11Buffer* src,unsigned,D3D11_BOX const* b){
  assert(b->top==0&&b->front==0&&b->bottom==1&&b->back==1&&b->right-b->left==64);
  assert(offset+64<=dst->bytes.size()&&b->right<=src->bytes.size());
  std::memcpy(dst->bytes.data()+offset,src->bytes.data()+b->left,64);++copies;
 }
 void IASetInputLayout(ID3D11InputLayout*){}void VSSetShader(ID3D11VertexShader*,void*,unsigned){}
 void IASetVertexBuffers(unsigned slot,unsigned count,ID3D11Buffer* const* b,UINT const* strides,UINT const* offsets){
  assert(slot==0&&count==2&&b[0]==b[1]);if(!b[0]){geometry=nullptr;return;}
  assert(strides[0]==32&&strides[1]==4&&offsets[0]==0);
  geometry=b[0];selection_offset=offsets[1];
 }
 void IASetIndexBuffer(ID3D11Buffer* b,unsigned format,unsigned offset){assert(b==geometry&&format==DXGI_FORMAT_R32_UINT);index_offset=offset;}
 void VSSetShaderResources(unsigned slot,unsigned count,ID3D11ShaderResourceView* const* v){assert(slot==15&&count==1);placements=*v;}
 void DrawIndexed(unsigned count,unsigned first,int base){
  assert(base==0&&count%3==0);++draws;
  for(unsigned i=0;i<count;++i){unsigned vertex=0,row=0;
   std::memcpy(&vertex,geometry->bytes.data()+index_offset+(first+i)*4,4);
   std::memcpy(&row,geometry->bytes.data()+selection_offset+vertex*4,4);
   Receipt receipt;std::memcpy(receipt.vertex.data(),geometry->bytes.data()+vertex*32,32);
   std::memcpy(receipt.placement.data(),placements->buffer->bytes.data()+row*64,64);primitives.push_back(receipt);
  }
 }
};
'''


def harness():
    header = (ROOT / "Renderer/native/render_core/ordered_rigid_submission.h").read_text()
    return STUB + header.replace("#include <d3d11.h>", "") + r'''
using Owner=c3x_renderer::render_core::OrderedRigidSubmission<std::array<std::uint64_t,4>>;
struct Vertex {float words[8];};
struct Fixture {
 ID3D11Device device;ID3D11DeviceContext context;ID3D11Buffer source;Owner owner;
 std::array<Vertex,4> a{},b{};std::array<unsigned,6> ia={0,1,2,2,1,3};std::array<unsigned,3> ib={2,0,1};
 Fixture(){source.bytes.resize(8*64);for(unsigned row=0;row<8;++row)for(unsigned w=0;w<16;++w){
  unsigned value=1000+row*100+w;std::memcpy(source.bytes.data()+row*64+w*4,&value,4);}
  for(unsigned i=0;i<4;++i)for(unsigned w=0;w<8;++w){a[i].words[w]=float(i*8+w);b[i].words[w]=float(100+i*8+w);}}
 Owner::Input input(unsigned key,unsigned row,bool second=false){return {{key,1,2,3},second?b.data():a.data(),4,
  second?ib.data():ia.data(),second?3u:6u,row};}
 std::vector<Receipt> expected(Owner::Input const* inputs,unsigned count){std::vector<Receipt> result;
  for(unsigned n=0;n<count;++n)for(unsigned i=0;i<inputs[n].index_count;++i){Receipt receipt;
   std::memcpy(receipt.vertex.data(),static_cast<unsigned char const*>(inputs[n].vertices)+inputs[n].indices[i]*32,32);
   std::memcpy(receipt.placement.data(),source.bytes.data()+inputs[n].placement*64,64);result.push_back(receipt);}return result;}
 void issue(Owner::Range const* ranges,unsigned count){unsigned records=0,indices=0;
  owner.issue(&context,nullptr,nullptr,ranges,count,[&](unsigned i,unsigned r){indices+=i;records+=r;});
  assert(records==count&&indices);}
};
'''


def production_planner_harness():
    header = (ROOT / "Renderer/native/render_core/ordered_rigid_submission.h").read_text()
    source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
    methods = "\n".join(method(source, signature) for signature in (
        "    c3x_renderer::render_core::SharedInstanceSubmission::Key shared_instance_draw_key(",
        "    OrderedRigidKey ordered_rigid_key(",
        "    void retire_ordered_rigid_packets(",
        "    bool ordered_rigid_input(",
        "    void prepare_ordered_rigid_packets("))
    return STUB + header.replace("#include <d3d11.h>", "") + r'''
#include <climits>
#include "Renderer/native/render_core/geometry_draws.h"
namespace c3x_renderer {namespace render_core {
struct DrawParameterStream {static constexpr unsigned limit=256;};
struct SharedInstanceSubmission {
 using Key=std::array<std::uint64_t,24>;
 struct Range {unsigned first=0,count=0;explicit operator bool()const{return count!=0;}};
 struct Front {ID3D11Buffer* buffer=nullptr;unsigned records=0;std::map<Key,Range> ranges;mutable unsigned finds=0;
  Range find(Key const& key)const{++finds;auto f=ranges.find(key);return f==ranges.end()?Range{}:f->second;}};
 using Lease=std::shared_ptr<Front>;
};
}}
namespace c3x_renderer {namespace objects {enum {bridge_family,site_family,mine_family,farm_family,city_family,wall_family};}}
enum GeometryLayer {geometry_route=0,geometry_feature,geometry_city,geometry_wall,geometry_mine,geometry_farm,geometry_site};
constexpr unsigned geometry_layer_count=7;
struct Vertex {float words[8];};
struct Asset {std::vector<Vertex> vertices;std::vector<unsigned> indices;};
struct Bundle {std::vector<Asset> assets;};
namespace c3x_renderer {using FeatureBundle=::Bundle;}
struct Mesh {
 struct Bounds {int left=0,top=0,right=100,bottom=100;}bounds;
 ID3D11Buffer *buffer=nullptr,*indices=nullptr,*resource_instance=nullptr;
 ID3D11ShaderResourceView* animation_texture=nullptr;
 unsigned vertex_stride=32,vertex_offset=0,index_offset=128,index_count=6,index_format=DXGI_FORMAT_R32_UINT,
  projection_kind=2,source_tile_width=128,city_material=0xffffffffu;
 std::uint64_t version=1;bool rigid_source=true;float instance_material=21.31f;
 std::shared_ptr<std::vector<unsigned> const> instances=std::make_shared<std::vector<unsigned> const>(1,1);
 int translation_x=0,translation_y=0;float natural_projection[4]={0,0,128,256};
};
using GeometryDrawView=c3x_renderer::render_core::GeometryDrawView<Mesh,7>;
using GeometryDrawReference=GeometryDrawView::Reference;
struct Planner {
 using OrderedRigidKey=std::array<std::uint64_t,32>;
 using OrderedRigidPackets=c3x_renderer::render_core::OrderedRigidSubmission<OrderedRigidKey>;
 OrderedRigidPackets ordered_rigid_packets;
 ID3D11Device device_value;ID3D11DeviceContext context_value;
 ID3D11Device* device=&device_value;ID3D11DeviceContext* context=&context_value;
 std::uint64_t device_generation=1,content_revision=1;
 struct Topology {std::uint64_t scope=1;std::uint64_t scope_sequence()const{return scope;}}topology_cache;
 struct Rigid {struct Source {ID3D11Buffer* buffer;unsigned count,index_offset;};
  std::array<std::vector<Source>,6> meshes;}rigid_sources;
 Bundle bridge_bundle,site_bundle,farm_bundle,mine_bundle,city_bundle,wall_bundle;
 bool ensure_ordered_rigid_layout(){return true;}
''' + methods + r'''
};
'''


class OrderedRigidSubmissionTests(unittest.TestCase):
    def test_membership_retirement_preserves_draw_leases_and_reopens_capacity(self):
        run_cpp(harness() + r'''
int main(){
 {Fixture f;auto empty=f.owner.bytes();std::array<Owner::Input,2> inputs={f.input(1,0),f.input(2,1,true)};
  auto page=f.owner.append(&f.device,&f.context,&f.source,8,inputs.data(),2);assert(page);page.reset();
  auto held=f.owner.find(inputs[0].key);auto third=f.input(3,2);auto other=f.owner.append(&f.device,&f.context,&f.source,8,&third,1);assert(other);
  auto reuses=f.owner.reuses;assert(f.owner.contains(inputs[0].key)&&!f.owner.contains(f.input(4,3).key));assert(f.owner.reuses==reuses);
  auto full=f.owner.bytes();auto leased=held.page;
  f.owner.retire_missing([](auto const& key){return key[0]!=1;});
  assert(f.owner.page_count()==2&&!f.owner.find(inputs[0].key)&&f.owner.find(inputs[1].key));
  f.owner.retire_missing([](auto const& key){return key[0]==3;});
  assert(f.owner.page_count()==1&&f.owner.retired_entries==2&&!f.owner.find(inputs[1].key));
  // An admitted command still owns its exact immutable resources and charge.
  assert(f.owner.bytes()==full);auto expected=f.expected(inputs.data(),1);f.issue(&held,1);assert(f.context.primitives==expected);
  held={};leased.reset();assert(f.owner.bytes()<full&&f.owner.bytes()>empty);
  auto builds=f.owner.builds;auto cached=f.owner.find(third.key);
  f.owner.retire_missing([](auto const& key){return key[0]==3;});
  assert(f.owner.page_count()==1&&f.owner.builds==builds&&f.owner.retired_entries==2);
  auto recovered=f.owner.append(&f.device,&f.context,&f.source,8,inputs.data(),2);assert(recovered);
  assert(f.owner.page_count()==2&&f.owner.builds==builds+1);
  cached={};other.reset();recovered.reset();f.owner.clear();assert(f.owner.bytes()==empty);
 }assert(!live_buffers&&!live_views);
}
''')

    def test_adjacent_index_batch_keeps_primitive_order_and_operand_boundaries(self):
        run_cpp(harness() + r'''
struct Parameters {float projection[4]={};float translation[2]={};};
struct Chunk {
 ID3D11Buffer* buffer=nullptr,*indices=nullptr,*resource_instance=nullptr;
 ID3D11ShaderResourceView* animation_texture=nullptr;
 unsigned vertex_stride=32,vertex_offset=0,index_offset=0,index_count=3,index_format=1,city_material=0xffffffffu;
 bool rigid_source=false,city_environment=false;float city_atlas[4]={},visual_time=-1;
};
struct Draw {Chunk mesh;bool visible=true;Chunk const& content()const{return mesh;}bool water_visible()const{return visible;}};
int main(){
 ID3D11Buffer geometry;Draw a,b,c;a.mesh.buffer=a.mesh.indices=&geometry;b=a;c=a;
 b.mesh.index_offset=12;c.mesh.index_offset=24;Parameters p;
 auto joins=[&](Draw const& first,Draw const& next,Parameters const& x,Parameters const& y,unsigned stride=4){
  return c3x_renderer::render_core::compatible_ordered_index_range(first,next,x,y,stride);};
 assert(joins(a,b,p,p)&&joins(b,c,p,p));
 // Actual ordered primitive interpretation is identical to separate ranges,
 // including repeated vertices and arbitrary alpha/order-sensitive overlap.
 std::vector<unsigned> indices={2,0,1,1,0,2,2,2,1},separate,joined;
 for(auto const& draw:{a,b,c})for(unsigned i=0;i<draw.mesh.index_count;++i)separate.push_back(indices[draw.mesh.index_offset/4+i]);
 for(unsigned i=0;i<a.mesh.index_count+b.mesh.index_count+c.mesh.index_count;++i)joined.push_back(indices[i]);
 assert(separate==joined);
 for(unsigned field=0;field<13;++field){auto changed=b;auto q=p;
  switch(field){case 0:changed.mesh.buffer=nullptr;break;case 1:changed.mesh.indices=nullptr;break;
   case 2:changed.mesh.vertex_stride=88;break;case 3:changed.mesh.vertex_offset=32;break;
   case 4:changed.mesh.index_offset=24;break;case 5:changed.mesh.index_format=2;break;
   case 6:changed.mesh.city_material=2;break;case 7:changed.mesh.city_environment=true;break;
   case 8:changed.mesh.city_atlas[3]=.5f;break;case 9:changed.mesh.visual_time=0;break;
   case 10:changed.visible=false;break;case 11:changed.mesh.rigid_source=true;break;
   case 12:q.translation[0]=1;break;}
  assert(!joins(a,changed,p,q));
 }
 auto changed=b;changed.mesh.animation_texture=reinterpret_cast<ID3D11ShaderResourceView*>(1);assert(!joins(a,changed,p,p));
 changed=b;changed.mesh.resource_instance=&geometry;assert(!joins(a,changed,p,p));
 b.mesh.index_offset=6;assert(joins(a,b,p,p,2)); // Same contract for 16-bit indices.
 a.mesh.index_offset=0xfffffff8u;b.mesh.index_offset=4;assert(!joins(a,b,p,p)); // No offset wrap admission.
}
''')

    def test_production_range_selection_preserves_city_emission_and_packet_barriers(self):
        fresh = (ROOT / 'Renderer/sandbox/fresh_pipeline.h').read_text()
        start = fresh.index('                bool city_emission=')
        end = fresh.index('                if(mesh.city_material!=0xffffffffu){', start)
        production = fresh[start:end]
        run_cpp(harness() + r'''
constexpr unsigned DXGI_FORMAT_R16_UINT=2;
struct Parameters {float x=0;};
struct Chunk {
 ID3D11Buffer* buffer=nullptr,*indices=nullptr,*resource_instance=nullptr;
 ID3D11ShaderResourceView* animation_texture=nullptr;
 unsigned vertex_stride=32,vertex_offset=0,index_offset=0,index_count=3,index_format=1,city_material=0xffffffffu;
 bool rigid_source=false,city_environment=false;float city_atlas[4]={},visual_time=-1;
};
struct Draw {Chunk mesh;bool visible=true;Chunk const& content()const{return mesh;}bool water_visible()const{return visible;}};
struct Renderer {struct Cities {struct Material {bool ground=false;};
 struct Library {std::vector<Material> materials{{}};}library;
 bool emission=false;bool emits(unsigned)const{return emission;}}cities;}renderer;
struct Result {unsigned end,count;bool emission;};
Result select(std::vector<Draw> const& selected,std::array<Parameters,3> const& values,
        std::array<bool,3> const& packets,bool original_emission){
 unsigned i=0;auto const& mesh=selected[i].content();
''' + production + r'''
 return {end,index_count,city_emission};
}
int main(){
 ID3D11Buffer geometry;std::vector<Draw> rows(3);std::array<Parameters,3> values{};std::array<bool,3> packets{};
 for(unsigned n=0;n<3;++n){rows[n].mesh.buffer=rows[n].mesh.indices=&geometry;rows[n].mesh.index_offset=n*12;}
 auto result=select(rows,values,packets,false);assert(result.end==3&&result.count==9&&!result.emission);
 for(auto& row:rows)row.mesh.city_material=0;
 renderer.cities.emission=true;result=select(rows,values,packets,false);assert(result.end==1&&result.count==3&&result.emission);
 renderer.cities.emission=false;result=select(rows,values,packets,false);assert(result.end==3&&!result.emission);
 result=select(rows,values,packets,true);assert(result.end==1&&result.emission);
 renderer.cities.library.materials[0].ground=true;result=select(rows,values,packets,true);assert(result.end==3&&!result.emission);
 packets[1]=true;assert(select(rows,values,packets,false).end==1);packets[1]=false;
 values[1].x=1;assert(select(rows,values,packets,false).end==1);values[1].x=0;
 rows[1].visible=false;assert(select(rows,values,packets,false).end==1);rows[1].visible=true;
 rows[1].mesh.index_offset+=4;assert(select(rows,values,packets,false).end==1);
}
''')

    def test_actual_planner_capacity_misses_skip_front_search_and_preserve_cached_order(self):
        run_cpp(production_planner_harness() + r'''
int main(){
 for(bool entry_pressure:{true,false}){
  Planner p;ID3D11Buffer source,mesh_a,mesh_b,mesh_large;source.bytes.resize(8*64);
  for(unsigned n=0;n<source.bytes.size();++n)source.bytes[n]=static_cast<unsigned char>(n);
  Asset a;a.vertices.resize(4);a.indices={0,1,2,2,1,3};auto b=a;b.indices={2,0,1};
  for(unsigned n=0;n<4;++n)for(unsigned word=0;word<8;++word){a.vertices[n].words[word]=float(n*8+word);b.vertices[n].words[word]=float(100+n*8+word);}
  auto large=a;large.indices.assign(1800000,0);
  p.farm_bundle.assets={a,b,large};p.rigid_sources.meshes[c3x_renderer::objects::farm_family]=
   {{&mesh_a,6,128},{&mesh_b,3,128},{&mesh_large,unsigned(large.indices.size()),128}};
  Mesh ma,mb,ml;ma.buffer=ma.indices=&mesh_a;mb.buffer=mb.indices=&mesh_b;mb.index_count=3;
  ml.buffer=ml.indices=&mesh_large;ml.index_count=unsigned(large.indices.size());
  GeometryDrawView::Record first(ma),second(mb),miss(entry_pressure?ma:ml);
  first.owner={1,4};second.owner={2,5};miss.owner={3,6};
  using Shared=c3x_renderer::render_core::SharedInstanceSubmission;auto front=std::make_shared<Shared::Front>();front->buffer=&source;front->records=8;
  front->ranges.emplace(p.shared_instance_draw_key(geometry_farm,first),Shared::Range{0,1});
  front->ranges.emplace(p.shared_instance_draw_key(geometry_farm,second),Shared::Range{1,1});
  front->ranges.emplace(p.shared_instance_draw_key(geometry_farm,miss),Shared::Range{2,1});
  using Packets=Planner::OrderedRigidPackets;std::array<Packets::Range,256> ranges{};
  std::vector<GeometryDrawReference> selected={first,second};
  p.prepare_ordered_rigid_packets(geometry_farm,selected,front,ranges);
  assert(ranges[0]&&ranges[1]&&front->finds==2); // Observable eligible input searches.
  auto original_first=ranges[0],original_second=ranges[1];
  auto filler=[&](unsigned n,bool big){auto const& asset=big?large:a;Packets::Input input;
   input.key[0]=100000+n;input.vertices=asset.vertices.data();input.vertex_count=asset.vertices.size();
   input.indices=asset.indices.data();input.index_count=asset.indices.size();input.placement=3;return input;};
  if(entry_pressure){
   std::array<Packets::Input,Packets::record_limit> inputs;
   for(unsigned first_key=2;first_key<Packets::entry_limit;){
    auto count=std::min<unsigned>(inputs.size(),Packets::entry_limit-first_key);
    for(unsigned n=0;n<count;++n)inputs[n]=filler(first_key+n,false);
    assert(p.ordered_rigid_packets.append(p.device,p.context,&source,8,inputs.data(),count));first_key+=count;
   }
  }else{
   auto geometry=std::size_t(large.vertices.size())*36+large.indices.size()*4;unsigned n=0;
   while(p.ordered_rigid_packets.can_append(geometry,1)){
    auto input=filler(n++,true);assert(p.ordered_rigid_packets.append(p.device,p.context,&source,8,&input,1));
   }assert(n&&n<Packets::entry_limit);
  }
  auto const& absent=miss.content();auto geometry=std::size_t(absent.index_offset/32)*36+std::size_t(absent.index_count)*4;
  assert(!p.ordered_rigid_packets.can_append(geometry,1));
  auto builds=p.ordered_rigid_packets.builds,uploaded=p.ordered_rigid_packets.uploaded_bytes;auto bytes=p.ordered_rigid_packets.bytes();
  auto creates=p.device_value.creates,copies=p.context_value.copies;front->finds=0;
  // A missing occurrence remains a fallback gap. Reordered/duplicated cached
  // ranges on either side still select their original immutable resources.
  selected={second,miss,first,second};
  for(unsigned repeat=0;repeat<32;++repeat){ranges={};p.prepare_ordered_rigid_packets(geometry_farm,selected,front,ranges);
   assert(ranges[0].page==original_second.page&&ranges[0].first==original_second.first&&!ranges[1]);
   assert(ranges[2].page==original_first.page&&ranges[2].first==original_first.first);
   assert(ranges[3].page==original_second.page&&ranges[3].first==original_second.first);
  }
  assert(front->finds==0&&p.device_value.creates==creates&&p.context_value.copies==copies);
  assert(p.ordered_rigid_packets.builds==builds&&p.ordered_rigid_packets.uploaded_bytes==uploaded&&p.ordered_rigid_packets.bytes()==bytes);
  auto expected=[&](Asset const& asset,unsigned row){std::vector<Receipt> result;
   for(auto index:asset.indices){Receipt receipt;std::memcpy(receipt.vertex.data(),&asset.vertices[index],32);
    std::memcpy(receipt.placement.data(),source.bytes.data()+row*64,64);result.push_back(receipt);}return result;};
  auto want=expected(b,1),part=expected(a,0);want.insert(want.end(),part.begin(),part.end());
  part=expected(b,1);want.insert(want.end(),part.begin(),part.end());
  p.ordered_rigid_packets.issue(p.context,nullptr,nullptr,ranges.data(),selected.size(),[](unsigned,unsigned){});
  assert(p.context_value.primitives==want&&p.context_value.draws==2);
  assert(p.ordered_rigid_packets.bytes()==p.ordered_rigid_packets.metadata_bytes()+p.ordered_rigid_packets.gpu_bytes());
  assert(p.ordered_rigid_packets.peak_bytes()<=Packets::budget);
 }assert(!live_buffers&&!live_views);
}
''')

    def test_actual_planner_uses_all_shared_source_families_with_ordered_ranges(self):
        run_cpp(production_planner_harness() + r'''
int main(){
 {Planner p;ID3D11Buffer source;source.bytes.resize(8*64);std::array<ID3D11Buffer,6> meshes;
  for(unsigned n=0;n<source.bytes.size();++n)source.bytes[n]=static_cast<unsigned char>(n);
  std::array<Bundle*,6> bundles={&p.bridge_bundle,&p.site_bundle,&p.mine_bundle,&p.farm_bundle,&p.city_bundle,&p.wall_bundle};
  std::array<unsigned,6> layers={geometry_route,geometry_site,geometry_mine,geometry_farm,geometry_city,geometry_wall};
  using Shared=c3x_renderer::render_core::SharedInstanceSubmission;auto front=std::make_shared<Shared::Front>();front->buffer=&source;front->records=8;
  for(unsigned family=0;family<6;++family){
   Asset asset;asset.vertices.resize(4);asset.indices={2,0,1};
   for(unsigned i=0;i<4;++i)for(unsigned w=0;w<8;++w)asset.vertices[i].words[w]=float(family*100+i*8+w);
   bundles[family]->assets.push_back(asset);p.rigid_sources.meshes[family].push_back({&meshes[family],3,128});
   Mesh mesh;mesh.buffer=mesh.indices=&meshes[family];mesh.index_count=3;mesh.instance_material=10.f+family;
   GeometryDrawView::Record first(mesh),second(mesh);first.owner={family+1,family+10};second.owner=first.owner;second.translation_x=32;
   front->ranges.emplace(p.shared_instance_draw_key(layers[family],first),Shared::Range{0,1});
   front->ranges.emplace(p.shared_instance_draw_key(layers[family],second),Shared::Range{1,1});
   std::vector<GeometryDrawReference> selected={first,second};std::array<Planner::OrderedRigidPackets::Range,256> ranges{};
   p.prepare_ordered_rigid_packets(layers[family],selected,front,ranges);assert(ranges[0]&&ranges[1]&&ranges[0].contiguous(ranges[1]));
   auto builds=p.ordered_rigid_packets.builds;auto copies=p.context_value.copies;
   selected={second,first,second};ranges={};p.prepare_ordered_rigid_packets(layers[family],selected,front,ranges);
   assert(ranges[0]&&ranges[1]&&ranges[2]&&p.ordered_rigid_packets.builds==builds&&p.context_value.copies==copies);
   p.context_value.primitives.clear();p.ordered_rigid_packets.issue(p.context,nullptr,nullptr,ranges.data(),selected.size(),[](unsigned,unsigned){});
   std::vector<Receipt> expected;
   for(unsigned row:{1,0,1})for(auto index:asset.indices){Receipt receipt;std::memcpy(receipt.vertex.data(),&asset.vertices[index],32);
    std::memcpy(receipt.placement.data(),source.bytes.data()+row*64,64);expected.push_back(receipt);}
   assert(p.context_value.primitives==expected);
   Planner::OrderedRigidPackets::Input invalid;mesh.city_material=0;assert(!p.ordered_rigid_input(layers[family],first,front,invalid));
   mesh.city_material=0xffffffffu;mesh.animation_texture=reinterpret_cast<ID3D11ShaderResourceView*>(1);
   assert(!p.ordered_rigid_input(layers[family],first,front,invalid));mesh.animation_texture=nullptr;
   mesh.resource_instance=&source;assert(!p.ordered_rigid_input(layers[family],first,front,invalid));
  }
 }assert(!live_buffers&&!live_views);
}
''')

    def test_actual_membership_retirement_selects_exact_main_and_reflection_occurrences(self):
        run_cpp(production_planner_harness() + r'''
int main(){
 {Planner p;ID3D11Buffer source,mesh_buffer;source.bytes.resize(8*64);
  p.ordered_rigid_packets.select_scope({1,1,1});Mesh mesh;mesh.buffer=mesh.indices=&mesh_buffer;
  GeometryDrawView::Record main(mesh),mirror(mesh),missing(mesh),newcomer(mesh);
  main.owner=mirror.owner=missing.owner=newcomer.owner={1,8};
  main.translation_x=-6400;mirror.translation_x=6400;missing.translation_x=12800;newcomer.translation_x=25600;
  std::array<Vertex,4> vertices{};std::array<unsigned,3> indices={2,0,1};
  auto input=[&](GeometryDrawView::Record const& record,unsigned row){return Planner::OrderedRigidPackets::Input{
   p.ordered_rigid_key(geometry_city,record),vertices.data(),unsigned(vertices.size()),indices.data(),unsigned(indices.size()),row};};
  auto a=input(main,0),b=input(mirror,1),c=input(missing,2);std::array<Planner::OrderedRigidPackets::Input,3> inputs={a,b,c};
  auto page=p.ordered_rigid_packets.append(p.device,p.context,&source,8,inputs.data(),3);assert(page);page.reset();
  auto old=p.ordered_rigid_packets.find(c.key);GeometryDrawView::Records records;
  records[geometry_city]={main,mirror,main,newcomer}; // Duplicate and unadmitted keys cannot add residency.
  auto builds=p.ordered_rigid_packets.builds;auto bytes=p.ordered_rigid_packets.bytes();auto reuses=p.ordered_rigid_packets.reuses;
  p.retire_ordered_rigid_packets(records);
  assert(p.ordered_rigid_packets.contains(a.key)&&p.ordered_rigid_packets.contains(b.key)&&!p.ordered_rigid_packets.contains(c.key));
  assert(!p.ordered_rigid_packets.contains(input(newcomer,3).key)&&p.ordered_rigid_packets.retired_entries==1);
  assert(p.ordered_rigid_packets.builds==builds&&p.ordered_rigid_packets.reuses==reuses&&p.ordered_rigid_packets.bytes()==bytes);
  // Erasing the last owners retires the page, but a previously admitted draw
  // keeps its exact source/placement resources until that reader finishes.
  records={};p.retire_ordered_rigid_packets(records);assert(!p.ordered_rigid_packets.page_count()&&p.ordered_rigid_packets.bytes()==bytes);
  p.ordered_rigid_packets.issue(p.context,nullptr,nullptr,&old,1,[](unsigned,unsigned){});assert(p.context_value.primitives.size()==3);
  old={};assert(p.ordered_rigid_packets.bytes()<bytes);
 }assert(!live_buffers&&!live_views);
}
''')

    def test_more_than_64_small_pages_are_owned_until_scope_retirement(self):
        run_cpp(harness() + r'''
int main(){
 {Fixture f;auto empty=f.owner.bytes();Owner::Range held;
  // Separate immutable occurrences can form small packets; page count must
  // not refuse them while the charged byte and exact-key budgets have room.
  for(unsigned n=0;n<192;++n){auto input=f.input(n+1,n%8,n%2);
   auto geometry=std::size_t(input.vertex_count)*36+std::size_t(input.index_count)*4;
   assert(f.owner.can_append(geometry,1));
   auto page=f.owner.append(&f.device,&f.context,&f.source,8,&input,1);assert(page);
   if(!n)held=f.owner.find(input.key);
   assert(f.owner.bytes()==f.owner.metadata_bytes()+f.owner.gpu_bytes());
   assert(f.owner.bytes()<=Owner::budget&&f.owner.peak_bytes()<=Owner::budget);
  }
  assert(f.owner.page_count()==192&&f.owner.builds==192&&!f.owner.refusals);
  auto last=f.input(192,7,true);auto selected=f.owner.find(last.key);assert(selected);
  auto expected=f.expected(&last,1);f.issue(&selected,1);assert(f.context.primitives==expected);
  // Owner-map references survive every local append lease; navigation uses
  // their unchanged resources without another upload or placement copy.
  auto creates=f.device.creates,copies=f.context.copies;selected={};
  for(unsigned n=0;n<192;++n){auto input=f.input(n+1,n%8,n%2);assert(f.owner.find(input.key));}
  assert(f.device.creates==creates&&f.context.copies==copies);
  f.owner.clear();assert(f.owner.page_count()==0&&!f.owner.find(last.key));
  assert(f.owner.bytes()>empty&&f.owner.gpu_bytes()); // External draw lease still owns its charge.
  f.context.primitives.clear();f.issue(&held,1);held={};
  assert(f.owner.bytes()==empty&&f.owner.gpu_bytes()==0);
 }assert(!live_buffers&&!live_views);
}
''')

    def test_exact_key_capacity_refuses_without_changing_owned_packets(self):
        run_cpp(harness() + r'''
int main(){
 {Fixture f;auto empty=f.owner.bytes();std::array<Owner::Input,Owner::record_limit> inputs;
  for(unsigned first=0;first<Owner::entry_limit;first+=inputs.size()){
   for(unsigned n=0;n<inputs.size();++n)inputs[n]=f.input(first+n+1,n%8,n%2);
   assert(f.owner.append(&f.device,&f.context,&f.source,8,inputs.data(),inputs.size()));
  }
  auto old=f.input(1,0);auto held=f.owner.find(old.key);auto last=f.input(Owner::entry_limit,7,true);
  assert(held&&f.owner.find(last.key));auto bytes=f.owner.bytes();auto creates=f.device.creates,copies=f.context.copies;
  auto refused=f.input(Owner::entry_limit+1,0);assert(!f.owner.can_append(168,1));
  assert(!f.owner.append(&f.device,&f.context,&f.source,8,&refused,1));
  assert(f.owner.bytes()==bytes&&f.device.creates==creates&&f.context.copies==copies&&f.owner.find(old.key));
  assert(f.owner.page_count()==Owner::entry_limit/Owner::record_limit&&f.owner.peak_bytes()<=Owner::budget);
  f.owner.clear();assert(f.owner.page_count()==0&&f.owner.bytes()>empty);
  held={};assert(f.owner.bytes()==empty&&f.owner.can_append(168,1));
 }assert(!live_buffers&&!live_views);
}
''')

    def test_partial_publication_rollback_releases_strong_map_references(self):
        run_cpp(harness() + r'''
#include <stdexcept>
struct ThrowKey {
 unsigned value=0;static bool armed;
 bool operator<(ThrowKey const& other)const{
  if(armed&&(value==3||other.value==3)){armed=false;throw std::runtime_error("injected map publication failure");}
  return value<other.value;
 }
};bool ThrowKey::armed=false;
using ThrowOwner=c3x_renderer::render_core::OrderedRigidSubmission<ThrowKey>;
int main(){
 {Fixture f;ThrowOwner owner;auto source=f.input(1,0);
  ThrowOwner::Input old={{1},source.vertices,source.vertex_count,source.indices,source.index_count,0};
  auto page=owner.append(&f.device,&f.context,&f.source,8,&old,1);assert(page);page={};
  auto bytes=owner.bytes(),gpu=owner.gpu_bytes();auto buffers=live_buffers,views=live_views;
  std::array<ThrowOwner::Input,3> pending={old,old,old};
  for(unsigned n=0;n<pending.size();++n){pending[n].key={n+2};pending[n].placement=n+1;}
  ThrowKey::armed=true;
  assert(!owner.append(&f.device,&f.context,&f.source,8,pending.data(),pending.size()));
  assert(!ThrowKey::armed); // Failure occurs after key 2 has already published.
  assert(owner.find({1})&&!owner.find({2})&&!owner.find({3})&&!owner.find({4}));
  assert(owner.page_count()==1&&owner.builds==1&&owner.bytes()==bytes&&owner.gpu_bytes()==gpu);
  assert(live_buffers==buffers&&live_views==views&&owner.bytes()==owner.metadata_bytes()+owner.gpu_bytes());
  assert(owner.append(&f.device,&f.context,&f.source,8,pending.data(),pending.size()));
  assert(owner.page_count()==2&&owner.find({2})&&owner.find({3})&&owner.find({4}));owner.clear();
 }assert(!live_buffers&&!live_views);
}
''')

    def test_actual_production_planner_reuses_navigation_and_rejects_state_changes(self):
        run_cpp(production_planner_harness() + r'''
int main(){
 {Planner p;ID3D11Buffer source,mesh_a,mesh_b;source.bytes.resize(8*64);
  for(unsigned i=0;i<source.bytes.size();++i)source.bytes[i]=static_cast<unsigned char>(i);
  Asset a;a.vertices.resize(4);a.indices={0,1,2,2,1,3};auto b=a;b.indices={1,0,2};
  p.farm_bundle.assets={a,b};p.rigid_sources.meshes[c3x_renderer::objects::farm_family]={{&mesh_a,6,128},{&mesh_b,3,128}};
  Mesh ma,mb;ma.buffer=ma.indices=&mesh_a;mb.buffer=mb.indices=&mesh_b;mb.index_count=3;mb.version=2;
  GeometryDrawView::Record first(ma),second(mb),wrapped(ma);first.owner={1,4};second.owner={2,5};wrapped.owner=first.owner;
  wrapped.translation_x=6400;wrapped.translation_y=-3200;
  std::vector<GeometryDrawReference> selected={first,second,wrapped};
  using Shared=c3x_renderer::render_core::SharedInstanceSubmission;auto front=std::make_shared<Shared::Front>();front->buffer=&source;front->records=8;
  for(unsigned i=0;i<selected.size();++i)front->ranges.emplace(p.shared_instance_draw_key(geometry_farm,selected[i]),Shared::Range{i,1});
  std::array<Planner::OrderedRigidPackets::Range,256> ranges{};
  p.prepare_ordered_rigid_packets(geometry_farm,selected,front,ranges);assert(ranges[0]&&ranges[1]&&ranges[2]&&p.ordered_rigid_packets.builds==1);
  auto creates=p.device_value.creates,copies=p.context_value.copies;
  selected={wrapped,first};ranges={};p.prepare_ordered_rigid_packets(geometry_farm,selected,front,ranges);
  assert(ranges[0]&&ranges[1]&&p.device_value.creates==creates&&p.context_value.copies==copies);
  // A new wrapped occurrence appends only one placement/source mesh range.
  auto new_wrap=wrapped;new_wrap.translation_x+=6400;selected={first,new_wrap};
  front->ranges.emplace(p.shared_instance_draw_key(geometry_farm,selected[1]),Shared::Range{3,1});
  ranges={};p.prepare_ordered_rigid_packets(geometry_farm,selected,front,ranges);
  assert(ranges[0]&&ranges[1]&&p.ordered_rigid_packets.builds==2&&p.context_value.copies==copies+1);
  // Nonrigid state, native city materials, animation/resource bindings and
  // malformed source geometry use ordinary draws; no cached range crosses them.
  auto check=[&](Mesh const& changed){GeometryDrawReference draw(changed);std::vector<GeometryDrawReference> one={draw};
   front->ranges[p.shared_instance_draw_key(geometry_farm,draw)]={0,1};ranges={};
   p.prepare_ordered_rigid_packets(geometry_farm,one,front,ranges);assert(!ranges[0]);};
  auto changed=ma;changed.rigid_source=false;check(changed);changed=ma;changed.city_material=0;check(changed);
  ID3D11ShaderResourceView view;changed=ma;changed.animation_texture=&view;check(changed);
  changed=ma;changed.resource_instance=&source;check(changed);changed=ma;changed.vertex_stride=120;check(changed);
  changed=ma;changed.index_offset=0;check(changed);
  auto before=p.device_value.creates;ranges={};p.prepare_ordered_rigid_packets(geometry_route,selected,front,ranges);
  assert(!ranges[0]&&!ranges[1]&&p.device_value.creates==before);
  ++p.device_generation;ranges={};p.prepare_ordered_rigid_packets(geometry_farm,selected,front,ranges);
  assert(ranges[0]&&ranges[1]&&p.device_value.creates>before);
 }assert(!live_buffers&&!live_views);
}
''')

    def test_exact_primitives_material_rows_and_navigation_subranges(self):
        run_cpp(harness() + r'''
int main(){
 {Fixture f;std::array<Owner::Input,3> inputs={f.input(1,4),f.input(2,1,true),f.input(3,7)};
  auto expected=f.expected(inputs.data(),3);auto page=f.owner.append(&f.device,&f.context,&f.source,8,inputs.data(),3);assert(page);
  std::array<Owner::Range,3> ranges={f.owner.find(inputs[0].key),f.owner.find(inputs[1].key),f.owner.find(inputs[2].key)};
  auto creates=f.device.creates;f.issue(ranges.data(),3);assert(f.context.draws==1&&f.context.primitives==expected);
  // A front replacement cannot change the compact immutable placement rows.
  std::fill(f.source.bytes.begin(),f.source.bytes.end(),0);f.context.primitives.clear();f.issue(ranges.data(),3);
  assert(f.context.primitives==expected&&f.device.creates==creates&&f.owner.builds==1);
  // Navigation chooses existing ranges, with an omitted primitive range left out.
  std::array<Owner::Range,2> selected={ranges[0],ranges[2]};f.context.primitives.clear();auto before=f.context.draws;
  f.issue(selected.data(),2);assert(f.context.draws==before+2&&f.device.creates==creates);
  auto subset=expected;subset.erase(subset.begin()+6,subset.begin()+9);assert(f.context.primitives==subset);
  // Reordering and duplicates keep the requested native alpha/primitive order.
  std::array<Owner::Range,3> order={ranges[2],ranges[1],ranges[2]};f.context.primitives.clear();before=f.context.draws;f.issue(order.data(),3);
  std::vector<Receipt> reordered;reordered.insert(reordered.end(),expected.begin()+9,expected.end());
  reordered.insert(reordered.end(),expected.begin()+6,expected.begin()+9);reordered.insert(reordered.end(),expected.begin()+9,expected.end());
  assert(f.context.primitives==reordered&&f.context.draws==before+2);
  assert(f.owner.bytes()<=Owner::budget&&f.owner.peak_bytes()<=Owner::budget);
 }assert(!live_buffers&&!live_views);
}
''')

    def test_reset_retirement_and_new_content_only(self):
        run_cpp(harness() + r'''
int main(){
 {Fixture f;auto input=f.input(1,2);auto page=f.owner.append(&f.device,&f.context,&f.source,8,&input,1);assert(page);
  auto range=f.owner.find(input.key);auto resident=f.owner.bytes();auto gpu=f.owner.gpu_bytes();assert(gpu);
  f.owner.select_scope({2,3,4});assert(!f.owner.find(input.key)&&f.owner.page_count()==0);
  assert(f.owner.bytes()==resident&&f.owner.gpu_bytes()==gpu); // Retired lease is still charged.
  auto next=f.input(2,3,true);auto replacement=f.owner.append(&f.device,&f.context,&f.source,8,&next,1);assert(replacement);
  assert(f.owner.builds==2&&f.owner.placement_copies==2&&f.owner.bytes()>resident);
  page.reset();range={};assert(f.owner.gpu_bytes()<gpu+replacement->geometry->bytes.size()+replacement->placements->bytes.size());
  auto chosen=f.owner.find(next.key);auto creates=f.device.creates;f.issue(&chosen,1);assert(f.device.creates==creates);
  replacement.reset();chosen={};f.owner.clear();assert(f.owner.gpu_bytes()==0&&f.owner.metadata_bytes()==f.owner.bytes());
 }assert(!live_buffers&&!live_views);
}
''')

    def test_invalid_input_and_allocation_refusal_preserve_fallback(self):
        run_cpp(harness() + r'''
int main(){
 {Fixture f;auto input=f.input(1,0);auto page=f.owner.append(&f.device,&f.context,&f.source,8,&input,1);assert(page);
  auto before=f.owner.bytes();auto creates=f.device.creates;auto bad=f.input(2,8);
  assert(!f.owner.append(&f.device,&f.context,&f.source,8,&bad,1));bad=f.input(2,0);unsigned invalid[]={0,4,1};bad.indices=invalid;bad.index_count=3;
  assert(!f.owner.append(&f.device,&f.context,&f.source,8,&bad,1));assert(f.device.creates==creates&&f.owner.bytes()==before);
  for(unsigned stage=1;stage<=3;++stage){bad=f.input(2,0,true);f.device.fail_at=f.device.creates+stage;
   assert(!f.owner.append(&f.device,&f.context,&f.source,8,&bad,1));assert(!f.owner.find(bad.key)&&f.owner.bytes()==before&&f.owner.find(input.key));}
  f.device.fail_at=0;auto good=f.input(2,0,true);assert(f.owner.append(&f.device,&f.context,&f.source,8,&good,1));
 }assert(!live_buffers&&!live_views);
}
''')

    def test_bounded_pinned_pressure_and_eviction(self):
        run_cpp(harness() + r'''
int main(){
 {Fixture f;std::vector<unsigned> many(1800000,0);std::vector<Owner::Lease> pins;
  auto geometry=std::size_t(4)*36+many.size()*4;
  // Enough maximal pages to exceed the owner's charged budget, whatever that
  // budget is (each page retains one geometry copy after staging).
  unsigned const attempts=unsigned(Owner::budget/geometry)+2;
  for(unsigned n=0;n<attempts;++n){auto input=f.input(n+1,0);input.indices=many.data();input.index_count=unsigned(many.size());
   auto page=f.owner.append(&f.device,&f.context,&f.source,8,&input,1);if(!page)break;pins.push_back(page);
   assert(f.owner.bytes()==f.owner.metadata_bytes()+f.owner.gpu_bytes());
   assert(f.owner.bytes()<=Owner::budget&&f.owner.peak_bytes()<=Owner::budget);}
  assert(!pins.empty()&&pins.size()<attempts&&f.owner.refusals);
  assert(!f.owner.can_append(geometry,1));auto creates=f.device.creates;
  auto refused=f.input(100,0);refused.indices=many.data();refused.index_count=unsigned(many.size());
  assert(!f.owner.append(&f.device,&f.context,&f.source,8,&refused,1)&&f.device.creates==creates);
  auto held=f.owner.gpu_bytes();f.owner.clear();assert(f.owner.gpu_bytes()==held&&!f.owner.can_append(geometry,1));
  pins.clear();assert(f.owner.gpu_bytes()==0&&f.owner.can_append(geometry,1));
  auto small=f.input(100,1);assert(f.owner.append(&f.device,&f.context,&f.source,8,&small,1));
 }assert(!live_buffers&&!live_views);
}
''')


if __name__ == "__main__":
    unittest.main()
