"""Execute viewport admission with production quality and source dependency proofs."""
import unittest
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.test_fresh_shared_submission import method

class GeometryDependencyReuseTests(unittest.TestCase):
    def test_changed_remote_appearance_can_pass_view_match_but_reject_retained_mesh(self):
        source=(ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
        match=method(source,'    bool geometry_matches(CachedGeometry const & candidate,')
        same=method(source,'    bool same_terrain_content(c3x_renderer_tile_v1 const & left,')
        valid=method(source,'    bool tile_content_valid(CachedTileGeometry& cached,')
        run_cpp(r'''
#define C3X_RENDERER64_FRESH
#include "Renderer/native/render_core/captured_scene.h"
#include <array>
#include <cassert>
#include <vector>
#include <utility>
#include <cstring>
#include <memory>
namespace c3x_renderer {
 struct TerrainFrameSignature {std::uint64_t geometry=19;};
 namespace render_core {
 struct ForegroundSelection {bool operator==(ForegroundSelection const&)const{return true;}
 bool preserves(c3x_renderer_tile_v1 const&,c3x_renderer_tile_v1 const&)const{return true;}};
 }
}
struct CachedGeometry {
 bool valid=true;c3x_renderer::TerrainFrameSignature signature;
 c3x_renderer::render_core::ForegroundSelection selection;
 std::vector<c3x_renderer_tile_v1> tiles;
};
struct CachedTileGeometry {
 bool shared_natural=false,ground_component=false,world_ground=true,world_objects=true,validity=false;
 struct Handle {unsigned generation=0;} natural_content;
 struct Proof {};struct Mesh {std::shared_ptr<Proof> proof=std::make_shared<Proof>();};
 std::shared_ptr<Mesh> mesh=std::make_shared<Mesh>();
 std::uint64_t validity_epoch=0,validity_world_sequence=0;int validity_anchor_x=0,validity_anchor_y=0,source_tile_width=128;
 c3x_renderer::render_core::RasterDependencyRevisions::Checkpoint validity_revision{};
 std::vector<std::pair<std::uint64_t,std::uint64_t>> appearance_dependencies,dependencies,coast_dependencies;
 std::vector<std::pair<std::size_t,std::uint32_t>> world_dependencies;
 std::vector<std::pair<std::uint64_t,std::array<int,2>>> anchor_dependencies;
 std::vector<int> river_dependencies;
};
struct Harness {
 bool gpu_output_mode=true,scene_surface_requested=true;unsigned shadow_tile_width=128;
 unsigned frame_tile_invalid_shared=0,frame_tile_invalid_appearance=0,frame_tile_invalid_semantic=0,frame_tile_invalid_coast=0,frame_tile_invalid_world=0,frame_tile_invalid_anchor=0,frame_tile_invalid_river=0;
 struct Content {CachedTileGeometry* resolve(CachedTileGeometry::Handle){return nullptr;}} resident_content;
 struct Scene {
  std::uint64_t appearance=1,observations=1,world_epoch=1;
  struct Current {c3x_renderer_tile_v1 occurrence{};std::uint64_t semantic=1;} present;
  std::uint64_t observation_sequence()const{return observations;}
  std::uint64_t world_input_sequence()const{return world_epoch;}
  std::uint64_t world_appearance_revision(std::uint64_t)const{return appearance;}
  Scene const& compilation_view(bool)const{return *this;}
  Current const* current(std::uint64_t)const{return &present;}
 } topology_cache;
 struct Coast {
  std::uint64_t node_revision(std::uint64_t)const{return 0;}
  Coast const& world()const{return *this;}
  std::uint32_t at(std::size_t)const{return 0;}
 } world_coast;
 struct Natural {template<class T> bool valid(T const&)const{return true;}} natural;
 c3x_renderer::render_core::RasterDependencyRevisions raster_dependency_revisions;
 // Shared natural content is not exercised here; its proof has its own test.
 bool raster_content_valid(CachedTileGeometry::Proof const&){return true;}
''' + same+'\n'+match+'\n'+valid+r'''
};
using Domain=c3x_renderer::render_core::RasterDependencyRevisions::Domain;
int main(){
 Harness h;c3x_renderer_tile_v1 tile{};tile.tile_x=88;tile.tile_y=14;tile.anchor_x=1120;tile.anchor_y=630;
 tile.terrain_type=tile.real_terrain_type=7;tile.tile_flags=C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_EXPLORED;
 c3x_renderer_frame_v1 frame{};frame.tiles=&tile;frame.tile_count=1;
 CachedGeometry assembly;assembly.tiles.push_back(tile);c3x_renderer::TerrainFrameSignature signature;
 c3x_renderer::render_core::ForegroundSelection selection;int x=0,y=0;
 CachedTileGeometry mesh;mesh.appearance_dependencies.push_back({0xabcdef,1});
 assert(h.geometry_matches(assembly,frame,signature,selection,x,y,false));
 assert(h.tile_content_valid(mesh,tile));
 // A newly published remote forest dependency changes while captured camera
 // facts, native anchors, compiler profile and light remain identical.
 // Every authoritative producer edit touches the raster revision stream (577069a9).
 ++h.topology_cache.appearance;h.raster_dependency_revisions.touch(Domain::appearance,0xabcdef);
 ++h.topology_cache.observations;++h.topology_cache.world_epoch;
 assert(h.geometry_matches(assembly,frame,signature,selection,x,y,false));
 assert(!h.tile_content_valid(mesh,tile) && h.frame_tile_invalid_appearance==1);
 // Publication alone does not forbid reuse when all concrete inputs match.
 // A freshly published owner carries no revision memo.
 mesh.appearance_dependencies[0].second=h.topology_cache.appearance;mesh.validity_revision={};
 ++h.topology_cache.observations;++h.topology_cache.world_epoch;
 assert(h.geometry_matches(assembly,frame,signature,selection,x,y,false));
 assert(h.tile_content_valid(mesh,tile));
}
''')

    def test_admission_uses_current_quality_and_selected_source_lifetimes(self):
        source=(ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
        self.assertIn('        if(reuse_geometry && fresh_scene_path && !covered_membership)',source)
        valid=method(source,'    bool tile_content_valid(CachedTileGeometry& cached,')
        raster=method(source,'    bool raster_content_valid(CachedGeometryProof const& proof)')
        guard=method(source,'        if(reuse_geometry && fresh_scene_path && !covered_membership)')
        quality_start=source.index('        std::array<std::uint64_t,2> const compile_quality=')
        quality_end=source.index(';',quality_start)+1
        quality=source[quality_start:quality_end]
        run_cpp(r'''
#include "Renderer/native/c3x_renderer_api.h"
#include "Renderer/native/render_core/prepared_world_validity.h"
#include "Renderer/native/render_core/raster_dependency_revisions.h"
#include <array>
#include <cassert>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
struct Handle {unsigned generation=0;};
struct CachedGeometryProof {
 std::uint64_t scope=1,assets=7;bool ground_semantics=false;
 mutable c3x_renderer::render_core::RasterDependencyRevisions::Checkpoint validated_revision{};
 std::vector<std::pair<std::uint64_t,std::uint64_t>> appearance_dependencies,dependencies,coast_dependencies;
 std::vector<std::pair<std::size_t,std::uint32_t>> world_dependencies;
 std::vector<int> river_dependencies;
};
struct Mesh {std::shared_ptr<CachedGeometryProof> proof=std::make_shared<CachedGeometryProof>();};
struct CachedTileGeometry {
 bool shared_natural=false,ground_component=false,world_ground=true,world_objects=true,validity=false;
 Handle natural_content;
 c3x_renderer::render_core::RasterDependencyRevisions::Checkpoint validity_revision{};
 std::array<std::uint64_t,20> compile_context{};
 std::shared_ptr<Mesh> mesh=std::make_shared<Mesh>();
 std::uint64_t validity_epoch=0,validity_world_sequence=0;int validity_anchor_x=0,validity_anchor_y=0,source_tile_width=128;
 std::vector<std::pair<std::uint64_t,std::uint64_t>> appearance_dependencies,dependencies,coast_dependencies;
 std::vector<std::pair<std::size_t,std::uint32_t>> world_dependencies;
 std::vector<std::pair<std::uint64_t,std::array<int,2>>> anchor_dependencies;
 std::vector<int> river_dependencies;
};
struct Harness {
 bool fresh_scene_path=true,covered_membership=false,world_ground=true,world_objects=true,canonical_world_content=true;
 unsigned shadow_tile_width=128,content_revision=7,base_ground_grid=16,draw_record_count=648;
 struct Detail {unsigned value=3;unsigned identity()const{return value;}} patch_detail;
 unsigned frame_tile_invalid_shared=0,frame_tile_invalid_appearance=0,frame_tile_invalid_semantic=0,frame_tile_invalid_coast=0,frame_tile_invalid_world=0,frame_tile_invalid_anchor=0,frame_tile_invalid_river=0;
 std::array<unsigned,7> raster_proof_rejections{};
 struct Geometry {std::vector<Handle> tile_keys={{1}};} geometry_cache;
 struct Content {
  std::array<CachedTileGeometry*,3> owners{};unsigned resolves=0;
  CachedTileGeometry* resolve(Handle handle){++resolves;return owners[handle.generation];}
 } resident_content;
 struct Scene {
  std::uint64_t appearance=1,observations=1,world_epoch=1,scope=1;mutable unsigned appearance_reads=0;
  bool present=true;
  struct Current {c3x_renderer_tile_v1 occurrence{};std::uint64_t semantic=1,semantic_revision=1;std::int32_t ground=0;} item;
  std::uint64_t observation_sequence()const{return observations;}
  std::uint64_t world_input_sequence()const{return world_epoch;}
  std::uint64_t scope_sequence()const{return scope;}
  std::uint64_t world_appearance_revision(std::uint64_t)const{++appearance_reads;return appearance;}
  Scene const& compilation_view(bool)const{return *this;}
  Scene const& world_view()const{return *this;}
  Current const* current(std::uint64_t)const{return present?&item:nullptr;}
  Current const* retained(std::uint64_t)const{return current(0);}
 } topology_cache;
 struct Coast {
  std::uint64_t node_revision(std::uint64_t)const{return 0;}
  Coast const& world()const{return *this;}
  std::uint32_t at(std::size_t)const{return 0;}
 } world_coast;
 struct Natural {template<class T> bool valid(T const&)const{return true;}} natural;
 c3x_renderer::render_core::RasterDependencyRevisions raster_dependency_revisions;
''' + raster+'\n'+valid+r'''
 std::array<std::uint64_t,2> quality(c3x_renderer_frame_v1 const& frame)const{
''' + quality + r'''
  return compile_quality;
 }
 bool admits(c3x_renderer_frame_v1 const& frame,bool reuse_geometry=true){
  auto compile_quality=quality(frame);
''' + guard + r'''
  return reuse_geometry;
 }
 void publish(){++topology_cache.observations;++topology_cache.world_epoch;}
 // Every authoritative producer edit touches the raster revision stream (577069a9).
 void edit(c3x_renderer::render_core::RasterDependencyRevisions::Domain domain,std::uint64_t id){
  raster_dependency_revisions.touch(domain,id);publish();}
};
using Domain=c3x_renderer::render_core::RasterDependencyRevisions::Domain;
int main(){
 Harness h;CachedTileGeometry tile_owner,shared;
 h.resident_content.owners[1]=&tile_owner;h.resident_content.owners[2]=&shared;shared.shared_natural=true;
 c3x_renderer_tile_v1 tile{};tile.tile_x=-2;tile.tile_y=14;tile.anchor_x=1120;tile.anchor_y=630;
 c3x_renderer_frame_v1 frame{};frame.tiles=&tile;frame.tile_count=1;
 frame.tile_width=128;frame.tile_height=64;frame.world_width_tiles=100;frame.world_wrap_x=1;
 auto quality=h.quality(frame);tile_owner.compile_context[14]=quality[0];tile_owner.compile_context[15]=quality[1];
 tile_owner.appearance_dependencies.push_back({0xabcdef,1});
 assert(h.admits(frame));auto reads=h.topology_cache.appearance_reads;
 assert(h.admits(frame) && h.topology_cache.appearance_reads==reads);
 // Camera/light-only requests leave concrete source proof memoized; native
 // wrapped occurrence coordinates are not mistaken for compiler coordinates.
 frame.hour=18;++tile.anchor_x;assert(h.admits(frame));
 reads=h.topology_cache.appearance_reads;assert(h.admits(frame) && h.topology_cache.appearance_reads==reads);
 ++h.topology_cache.appearance;h.edit(Domain::appearance,0xabcdef);assert(!h.admits(frame));
 // A producer's freshly published immutable owner restores ordinary admission.
 // It carries no revision memo.
 tile_owner.appearance_dependencies[0].second=h.topology_cache.appearance;tile_owner.validity_revision={};h.publish();assert(h.admits(frame));
 for(unsigned field:{14u,15u}){auto original=tile_owner.compile_context[field];
  ++tile_owner.compile_context[field];assert(!h.admits(frame));tile_owner.compile_context[field]=original;}
 assert(h.admits(frame));h.canonical_world_content=false;assert(!h.admits(frame));h.canonical_world_content=true;
 // The early guard also proves actual source scope/assets, not merely a live
 // wrapper handle. Invalidated generations cannot borrow memoized validity.
 ++tile_owner.mesh->proof->scope;assert(!h.admits(frame));--tile_owner.mesh->proof->scope;
 ++tile_owner.mesh->proof->assets;assert(!h.admits(frame));--tile_owner.mesh->proof->assets;
 tile_owner.world_ground=false;assert(!h.admits(frame));tile_owner.world_ground=true;
 tile_owner.world_objects=false;assert(!h.admits(frame));tile_owner.world_objects=true;
 auto mesh=tile_owner.mesh;tile_owner.mesh.reset();assert(!h.admits(frame));tile_owner.mesh=mesh;
 auto proof=mesh->proof;mesh->proof.reset();assert(!h.admits(frame));mesh->proof=proof;
 h.resident_content.owners[1]=nullptr;assert(!h.admits(frame));h.resident_content.owners[1]=&tile_owner;
 // Shared natural content has its own source proof. The wrapper's memo alone
 // cannot admit a still-live but stale published natural generation.
 tile_owner.natural_content={2};shared.mesh->proof->appearance_dependencies.push_back({0xcdef,h.topology_cache.appearance});
 assert(h.admits(frame));++h.topology_cache.appearance;h.raster_dependency_revisions.touch(Domain::appearance,0xcdef);
 tile_owner.appearance_dependencies[0].second=h.topology_cache.appearance;tile_owner.validity_revision={};h.edit(Domain::appearance,0xabcdef);
 assert(!h.admits(frame) && h.raster_proof_rejections[1]);
 shared.mesh->proof->appearance_dependencies[0].second=h.topology_cache.appearance;assert(h.admits(frame));
 h.resident_content.owners[2]=nullptr;assert(!h.admits(frame));h.resident_content.owners[2]=&shared;
 auto source_mesh=shared.mesh;shared.mesh.reset();assert(!h.admits(frame));shared.mesh=source_mesh;
 auto source_proof=shared.mesh->proof;shared.mesh->proof.reset();assert(!h.admits(frame));shared.mesh->proof=source_proof;
 // Zero-valued semantic absence remains covered by the wrapper's complete
 // selection proof even though the raster proof skips absence itself.
 h.topology_cache.present=false;tile_owner.dependencies.push_back({73,0});h.publish();assert(h.admits(frame));
 h.topology_cache.present=true;h.edit(Domain::semantic,73);assert(!h.admits(frame) && h.frame_tile_invalid_semantic);
 tile_owner.dependencies.back().second=h.topology_cache.item.semantic;tile_owner.validity_revision={};h.publish();assert(h.admits(frame));
 // Index alignment is exact, including fallback/no-binding slots. Only the
 // captured selected handles are resolved, not every global world record.
 h.geometry_cache.tile_keys.clear();assert(!h.admits(frame));h.geometry_cache.tile_keys={{0}};
 auto resolves=h.resident_content.resolves;assert(h.admits(frame) && h.resident_content.resolves==resolves);
 h.geometry_cache.tile_keys={{1}};assert(h.admits(frame));assert(h.resident_content.resolves-resolves==3);
}
''')

if __name__=='__main__':unittest.main()
