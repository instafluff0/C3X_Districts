"""Host executable contracts for exact, bounded hill-decal preparation."""
import os
from pathlib import Path
import unittest

from Renderer.native.native_cpp_test import run_cpp

GEOMETRY_SOURCE = Path(__file__).resolve().parents[2] / "Renderer/native/source_fidelity/geometry.h"


PREAMBLE = r'''
#include "Renderer/native/source_fidelity/terrain_compiler.h"
#include "Renderer/lab/shared/natural/patterns.h"
#include <cassert>
#include <cstring>
namespace c3x_renderer {
float stable_random(std::uint32_t value){return patterns::stable_random(value);}
std::uint32_t stable_hash(std::uint32_t value){return patterns::feature_hash(value);}
}
using namespace c3x_renderer;
using namespace c3x_renderer::fidelity;
void same_mesh(render_core::PreparedMesh const& a,render_core::PreparedMesh const& b){
 assert(a.vertices==b.vertices && a.indices==b.indices && a.bounds==b.bounds);
 assert(a.world_low==b.world_low && a.world_high==b.world_high);
 assert(a.projected_bounds.extent==b.projected_bounds.extent);
 assert(a.vertex_stride==b.vertex_stride && a.index_stride==b.index_stride);
 assert(a.index_count==b.index_count && a.shared_grid==b.shared_grid);
}
void same_terrain(TerrainSurfaces const& a,TerrainSurfaces const& b){
 assert(a.world==b.world && a.coast==b.coast && a.proof_bytes==b.proof_bytes);
 assert(a.rivers.size()==b.rivers.size());
 for(unsigned i=0;i<a.rivers.size();++i){auto const& x=a.rivers[i];auto const& y=b.rivers[i];
  assert(x.first==y.first && x.second->values==y.second->values);
  assert(x.second->inputs->values==y.second->inputs->values && x.second->inputs->flow==y.second->inputs->flow);}
 for(unsigned i=0;i<3;++i)same_mesh(a.meshes[i],b.meshes[i]);
}
NaturalData natural_fixture(){
 NaturalData natural;natural.fields.resize(1);auto& field=natural.fields[0];
 field.width=field.height=16;field.pixels.resize(256);
 for(unsigned i=0;i<256;++i)field.pixels[i]=std::uint8_t(i*31);
 field.minimum=0;field.maximum=1;
 natural.bodies.resize(1);natural.bodies[0].vertices={
  {{-.5f,-.5f,0},{0,0,1},{0,0}},{{.5f,-.5f,0},{0,0,1},{1,0}},{{.5f,.5f,0},{0,0,1},{1,1}}};
 natural.recipes.resize(35);natural.recipes[0]={0,1,0,180,0,0,2,1,0};
 natural.recipes[25]={0,1,0,121,0,0,2,1,0};return natural;
}
TerrainCompileInput hill_input(){
 TerrainCompileInput input;input.tile_x=8;input.tile_y=4;input.real_terrain_type=5;input.ground=2;
 input.tile_width=128;input.tile_height=64;input.target_height=480;input.world_revision=1;return input;
}
'''


@unittest.skipIf(os.name == "nt", "Host-only contracts never dispatch Windows or VM tools")
class HillDecalCompactionTests(unittest.TestCase):
    def test_first_reference_order_uses_all_168_bytes(self):
        run_cpp(PREAMBLE + r'''
int main(){
 std::vector<MapVertex> expanded,unique;std::vector<unsigned> indices;
 MapVertex a{},b{},c{},d{},e{};b.x=b.world_x=1;c.y=c.world_y=1;
 // These fields are outside the92-byte natural GPU layout. They still
 // belong to the original168-byte dedup identity and must not be remapped.
 d=b;d.panel=1;e=a;e.normal_x=-0.f;
 MapVertex triangles[][3]={{a,b,c},{b,d,c},{a,e,c}};
 HillDecalOutput output(unique,indices);
 auto admit=[&](std::size_t extra){return output.scratch_bytes()+unique.capacity()*sizeof(MapVertex)+
  indices.capacity()*sizeof(unsigned)+extra<=8u*1024u*1024u;};
 assert(output.initialize(9,admit));
 for(auto const& triangle:triangles){expanded.insert(expanded.end(),triangle,triangle+3);assert(output.append(triangle,admit));}
 assert(unique.size()==5 && (indices==std::vector<unsigned>{0,1,2,1,3,2,0,4,2}));
 assert(!std::memcmp(&unique[3],&d,sizeof(d)) && !std::memcmp(&unique[4],&e,sizeof(e)));
 render_core::PreparedMesh original,compacted;render_core::MeshFormat format;format.natural=true;
 assert(render_core::prepare_mesh(expanded,nullptr,format,original,[]{return false;}));
 output.release_scratch();assert(output.scratch_bytes()==0);
 assert(render_core::prepare_mesh(unique,&indices,format,compacted,[]{return false;}));same_mesh(original,compacted);
}
''')

    def test_rock_stream_and_later_floor_stream_share_first_references(self):
        run_cpp(PREAMBLE + r'''
int main(){
 for(unsigned seed=0;seed<12;++seed){
  std::vector<MapVertex> receiver,expanded,unique;std::vector<unsigned> topology,indices;
  for(unsigned y=0;y<=16;++y)for(unsigned x=0;x<=16;++x){MapVertex v{};
   v.world_x=x/16.f;v.world_y=1-y/16.f;v.world_z=.1f+.01f*x;
   v.x=x;v.y=y;v.z=x+y;v.normal_z=1;receiver.push_back(v);}
  for(unsigned y=0;y<16;++y)for(unsigned x=0;x<16;++x){unsigned a=y*17+x;
   unsigned triangle[]={a,a+1,a+18,a,a+18,a+17};topology.insert(topology.end(),triangle,triangle+6);}
  Tile owner{int(seed*2),8,0,0,5};HillDecalOutput output(unique,indices);
  auto admit=[&](std::size_t extra){return output.scratch_bytes()+unique.capacity()*sizeof(MapVertex)+
   indices.capacity()*sizeof(unsigned)+extra<=8u*1024u*1024u;};
  assert(output.initialize(topology.size()*10+6,admit));
  emit_hill_decals(owner,0,0,receiver,&topology,expanded);assert(!expanded.empty());
  assert(emit_hill_decal_triangles(owner,0,0,receiver,&topology,[&](MapVertex const* v){return output.append(v,admit);}));
  // The same layer receives canopy floors after rocks; no later stream may
  // bypass the index output or restart first-reference numbering.
  MapVertex floor[]={receiver[0],receiver[1],receiver[18]};
  for(auto& v:floor)v.material_plains=3;
  for(unsigned i=0;i<2;++i){expanded.insert(expanded.end(),floor,floor+3);assert(output.append(floor,admit));}
  render_core::PreparedMesh original,compacted;render_core::MeshFormat format;format.natural=true;
  assert(render_core::prepare_mesh(expanded,nullptr,format,original,[]{return false;}));output.release_scratch();
  assert(render_core::prepare_mesh(unique,&indices,format,compacted,[]{return false;}));same_mesh(original,compacted);
 }
}
''')

    def test_actual_compiler_preserves_hill_canopy_and_current_proofs(self):
        run_cpp(PREAMBLE + r'''
int main(){
 auto natural=natural_fixture();std::array<ReliefFields,14> assets;
 render_core::WorldCoast world;std::vector<std::uint32_t> data(128,2+(5<<8));
 render_core::World dimensions{16,16,false,false};world.update(dimensions,data.data(),data.size(),1);
 int c=6,r=2;
 for(auto offset:std::array<std::array<int,2>,4>{{{{-1,0}},{{1,0}},{{0,-1}},{{0,1}}}})
  data[world.world().index(c+offset[0],r+offset[1])]=2+(7<<8);
 world.update(dimensions,data.data(),data.size(),2);
 auto input=hill_input();input.world_revision=2;TerrainCompileScratch original_scratch,compact_scratch;
 auto original=compile_terrain_surfaces(natural,assets,world,input,original_scratch,[]{return false;},false);
 auto compacted=compile_terrain_surfaces(natural,assets,world,input,compact_scratch,[]{return false;},true);
 assert(original && compacted);same_terrain(*original,*compacted);
 unsigned rocks=0,floors=0;auto const& mesh=compacted->meshes[1];
 for(std::size_t at=0;at<mesh.vertices.size();at+=mesh.vertex_stride){float material=0;
  std::memcpy(&material,mesh.vertices.data()+at+13*sizeof(float),sizeof(float));rocks+=material==2;floors+=material==3;}
 assert(rocks>0 && floors>0);
 unsigned checks=0;assert(!compile_terrain_surfaces(natural,assets,world,input,compact_scratch,[&]{return ++checks==20;},true));
 assert(checks==20);auto retry=compile_terrain_surfaces(natural,assets,world,input,compact_scratch,[]{return false;},true);
 assert(retry);same_terrain(*retry,*original);
 input.indexed=false;
 auto legacy=compile_terrain_surfaces(natural,assets,world,input,original_scratch,[]{return false;},false);
 assert(legacy);
 for(unsigned i=0;i<3;++i){assert(legacy->meshes[i].vertices==original->meshes[i].vertices);
  assert(legacy->meshes[i].indices==original->meshes[i].indices);assert(legacy->meshes[i].shared_grid==0);}
 auto legacy_retry=compile_terrain_surfaces(natural,assets,world,input,compact_scratch,[]{return false;},false);
 assert(legacy_retry);same_terrain(*legacy,*legacy_retry);
}
''', timeout=60)

    def test_actual_legacy_geometry_include_keeps_expanded_foreground_path(self):
        # Compile the actual foreground prefix and its shared body, through
        # the legacy !cpu_terrain_enabled include. City/forest body submission
        # follows this prefix and is outside this terrain-emission change.
        prefix = GEOMETRY_SOURCE.read_text().split('#include "../city_fidelity/geometry.h"', 1)[0]
        prefix = prefix.replace('"terrain_mesh_body.h"',
                                '"' + str(GEOMETRY_SOURCE.with_name("terrain_mesh_body.h")) + '"')
        run_cpp(PREAMBLE + r'''
bool legacy_geometry(NaturalData const& data,std::array<ReliefFields,14> const& assets,
        render_core::WorldCoast const& world_coast,TerrainCompileInput const& input,
        TerrainSurfaces& destination,bool cpu_terrain_enabled){
 using Vertex=MapVertex;
 std::array<std::vector<Vertex>,3> natural_vertices;
 std::array<std::vector<unsigned>,2> natural_grid_indices;
 NaturalWorld natural;static_cast<NaturalData&>(natural)=data;
 natural.update_rivers(world_coast.world(),input.world_revision);
 NaturalWorld::CellInputs river_inputs;NaturalWorld::DependencyScope scope(natural,&river_inputs);
 render_core::ExactPointCache<render_core::ShoreSample> shores;
 render_core::ExactPointCache<render_core::GroundSample> pickup;
 auto observe_world=[&](std::size_t index,std::uint32_t value){destination.world.emplace(index,value);};
 auto observe_coast=[&](std::uint64_t index,std::uint64_t revision){destination.coast.emplace(index,revision);};
 SurfaceQueries queries(world_coast,shores,input.tile_x,input.tile_y,observe_world,observe_coast,input.skip_flat_shore,&natural);
 auto world_lookup=[&](int c,int r){return queries.tile(c,r);};
 auto shore_sample_at=[&](float x,float y){return queries.shore(x,y);};
 auto material_weights_for=[&](float x,float y){return queries.weights(x,y);};
 auto relief_sample=[&](int kind,unsigned variant,int channel,float u,float v){return relief_source(assets,true,kind,variant,channel,u,v);};
 auto river=[&](int c,int r,float u,float v){auto const& world=world_coast.world();auto i=world.index(c,r);auto value=world.at(i);
  if(i!=std::size_t(-1))observe_world(i,value);
  if(value==0xffffffffu || !(value>>16&170u) || !input.river_ready)return 1000.f;
  return float(natural.river_sample({float(c)+u,float(r)+1-v}).distance);};
 auto dune=[](float,float){return 0.f;};
 auto activity=[&](int c,int r){auto const& world=world_coast.world();auto i=world.index(c,r);auto value=world.at(i);
  if(i!=std::size_t(-1))observe_world(i,value);return value!=0xffffffffu && (value>>24)!=0?1.f:0.f;};
 std::size_t height_queries=0;
 ReliefSurface pickup_surface(world_coast.world().dimensions(),(input.tile_x+input.tile_y)/2,(input.tile_x-input.tile_y)/2,
  shore_sample_at(queries.center_u,queries.center_v).distance,world_lookup,relief_sample,shore_sample_at,river,dune,activity,
  pickup,height_queries,input.separate_relief);
 auto pickup_height=[&](float x,float y){return pickup_surface.height(x,y);};
 auto natural_height_at=[&](float x,float y,float* support){return queries.height(natural,pickup_height,x,y,support);};
 auto const& tile=input;auto const& frame=input;int ground=input.ground;
 float half_w=input.tile_width*.5f,half_h=input.tile_height*.5f;
 float relief_projection_scale=float(input.tile_width)/224.f*.82f;
 bool fidelity_profile=true,river_assets_ready=input.river_ready,index_natural_grids=input.indexed;
 auto cancelled=[]{return false;};auto begin_natural_phase=[]{};auto record_natural_phase=[](unsigned){};
 auto patch_detail=input.detail;PatchLayouts patch_layouts;
 // Legacy foreground: no retained ground and an empty canopy clearing.
 bool retained_ground_terrain=false;auto topology_lookup=[](int,int){return nullptr;};
 auto canopy_clearing=[](auto const&,auto const&){return CanopyClearing{};};
''' + prefix + r'''
 }
 if(!export_border_ground_mesh(input.tile_x,input.tile_y,natural_vertices,natural_grid_indices))return false;
 for(unsigned layer=0;layer<3;++layer){auto topology=input.indexed && layer!=1?&natural_grid_indices[layer==0?0:1]:nullptr;
  render_core::MeshFormat format;format.natural=true;
  format.shared_grid=render_core::shared_mesh_grid(natural_vertices[layer].size(),topology,patch_layouts);
  if(!render_core::prepare_mesh(natural_vertices[layer],topology,format,destination.meshes[layer],[]{return false;}))return false;}
 destination.rivers.assign(river_inputs.begin(),river_inputs.end());destination.proof_bytes=natural.proof_bytes(destination.rivers);
 return true;
}
int main(){
 auto natural=natural_fixture();std::array<ReliefFields,14> assets;render_core::WorldCoast world;
 std::vector<std::uint32_t> data(128,2+(5<<8));world.update({16,16,false,false},data.data(),data.size(),1);
 auto input=hill_input();TerrainCompileScratch scratch;
 for(bool indexed:{false,true}){input.indexed=indexed;TerrainSurfaces legacy;
  assert(legacy_geometry(natural,assets,world,input,legacy,false));
  auto original=compile_terrain_surfaces(natural,assets,world,input,scratch,[]{return false;},false);
  assert(original);same_terrain(legacy,*original);assert(!legacy.meshes[1].empty());
  TerrainSurfaces skipped;assert(legacy_geometry(natural,assets,world,input,skipped,true));
  for(auto const& mesh:skipped.meshes)assert(mesh.empty());
 }
}
''', timeout=60)

    def test_unbounded_larger_input_keeps_recovery_semantics(self):
        run_cpp(PREAMBLE + r'''
int main(){
 auto natural=natural_fixture();std::array<ReliefFields,14> assets;render_core::WorldCoast world;
 std::vector<std::uint32_t> data(128,2+(5<<8));world.update({16,16,false,false},data.data(),data.size(),1);
 // A supported default64 hill in legacy expanded mode exceeds the bounded
 // transient cap, while unbounded foreground recovery must remain available.
 auto input=hill_input();input.indexed=false;TerrainCompileScratch scratch;
 auto recovery=compile_terrain_surfaces(natural,assets,world,input,scratch,[]{return false;},false);
 assert(recovery && !recovery->meshes[1].empty());
 assert(recovery->bytes()<TerrainPreparation::byte_limit);
 assert(!compile_terrain_surfaces(natural,assets,world,input,scratch,[]{return false;},true));
 auto retry=compile_terrain_surfaces(natural,assets,world,input,scratch,[]{return false;},false);
 assert(retry);same_terrain(*retry,*recovery);
}
''', timeout=60)

    def test_budget_counts_reallocation_overlap_and_rejects_before_allocation(self):
        run_cpp(PREAMBLE + r'''
int main(){
 {
  std::vector<MapVertex> vertices;std::vector<unsigned> indices;HillDecalOutput output(vertices,indices);
  assert(!output.initialize(6,[](std::size_t){return false;}));assert(output.rejected() && output.scratch_bytes()==0);
 }
 for(bool vertex_overlap:{false,true}){
  std::vector<MapVertex> vertices;std::vector<unsigned> indices;HillDecalOutput output(vertices,indices);
  std::size_t budget=vertex_overlap?1500:260,denied_request=0;
  auto live=[&]{return output.scratch_bytes()+vertices.capacity()*sizeof(MapVertex)+indices.capacity()*sizeof(unsigned);};
  auto admit=[&](std::size_t allocation){bool fits=live()+allocation<=budget;if(!fits)denied_request=live()+allocation;return fits;};
  assert(output.initialize(6,admit));MapVertex first[3]{};
  if(vertex_overlap){first[1].x=1;first[2].x=2;}
  assert(output.append(first,admit));MapVertex second[3]{};
  if(vertex_overlap)for(unsigned i=0;i<3;++i)second[i].x=3+i;
  assert(!output.append(second,admit));assert(output.rejected() && live()<=budget && denied_request>budget);
  if(vertex_overlap){
   // Final capacity would fit; current plus the entire replacement does not.
   assert(vertices.capacity()==4);assert(live()-4*sizeof(MapVertex)+8*sizeof(MapVertex)<=budget);
  }else{
   assert(indices.capacity()==3);assert(live()-3*sizeof(unsigned)+6*sizeof(unsigned)<=budget);
  }
  output.release_scratch();assert(output.scratch_bytes()==0);
 }
 {
  std::vector<MapVertex> vertices;std::vector<unsigned> indices;HillDecalOutput output(vertices,indices);
  auto admit=[&](std::size_t extra){return output.scratch_bytes()+vertices.capacity()*sizeof(MapVertex)+indices.capacity()*sizeof(unsigned)+extra<=8u*1024u*1024u;};
  assert(!output.initialize(8u*1024u*1024u,admit));assert(output.rejected() && output.scratch_bytes()==0);
 }
}
''')


if __name__ == "__main__":
    unittest.main()
