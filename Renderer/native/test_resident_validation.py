"""Execute production resident proof reuse across camera motion and mutations."""
import unittest
from Renderer.native.native_cpp_test import ROOT, run_cpp


def method(source, name):
    start = source.index('    bool ' + name + '(')
    brace = source.index('{', start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


class ResidentValidationTests(unittest.TestCase):
    def test_camera_reuse_mutations_absence_scope_and_native_anchors(self):
        source = (ROOT / 'Renderer/native/c3x_renderer.cpp').read_text()
        methods = '\n'.join(method(source, name) for name in (
            'raster_content_valid', 'tile_content_valid'))
        run_cpp(r'''
#include "Renderer/native/c3x_renderer_api.h"
#include "Renderer/native/render_core/raster_dependency_revisions.h"
#include <array>
#include <vector>
#include <memory>
#include <cassert>
using Revisions=c3x_renderer::render_core::RasterDependencyRevisions;
using Domain=Revisions::Domain;
struct CachedGeometryProof {
 std::vector<std::pair<std::uint64_t,std::uint64_t>> appearance_dependencies,dependencies,coast_dependencies,world_dependencies;
 std::vector<int> river_dependencies;
 unsigned scope=1,assets=2;bool ground_semantics=false;
 mutable Revisions::Checkpoint validated_revision{};
};
struct CachedTileGeometry {
 struct Mesh {std::shared_ptr<CachedGeometryProof> proof=std::make_shared<CachedGeometryProof>();};
 std::shared_ptr<Mesh> mesh=std::make_shared<Mesh>();
 bool shared_natural=false,ground_component=false,world_ground=true,world_objects=true,validity=false;
 unsigned source_tile_width=128;
 struct Handle {unsigned generation=0;} natural_content;
 std::uint64_t validity_epoch=0,validity_world_sequence=0;
 Revisions::Checkpoint validity_revision{};
 int validity_anchor_x=0,validity_anchor_y=0;
 std::vector<std::pair<std::uint64_t,std::uint64_t>> appearance_dependencies,dependencies,coast_dependencies,world_dependencies;
 std::vector<std::pair<std::uint64_t,std::array<int,2>>> anchor_dependencies;
 std::vector<int> river_dependencies;
};
struct Scene {
 struct Observation {std::uint64_t semantic=8,semantic_revision=1;c3x_renderer_tile_v1 occurrence{};};
 unsigned scope=1,epoch=1,world_epoch=1,visits=0;std::uint64_t appearance=7;
 Observation observation;bool present=true;
 unsigned scope_sequence()const{return scope;}
 unsigned observation_sequence()const{return epoch;}
 unsigned world_input_sequence()const{return world_epoch;}
 std::uint64_t world_appearance_revision(std::uint64_t){++visits;return appearance;}
 Observation const* current(std::uint64_t){++visits;return present?&observation:nullptr;}
 Observation const* retained(std::uint64_t id){return current(id);}
 Scene& compilation_view(bool){return *this;}
 Scene& world_view(){return *this;}
};
namespace c3x_renderer::render_core {
std::uint64_t ground_topology_value(Scene::Observation const* p){return p?p->semantic:0;}
}
struct World {
 std::uint64_t value=10;unsigned visits=0;
 std::uint64_t at(std::uint64_t){++visits;return value;}
};
struct Coast {std::uint64_t value=9;unsigned visits=0;World inputs;
 std::uint64_t node_revision(std::uint64_t){++visits;return value;}
 World& world(){return inputs;}
};
struct Natural {bool correct=true;unsigned visits=0;
 bool valid(std::vector<int> const&){++visits;return correct;}
};
struct Harness {
 Scene topology_cache;Coast world_coast;Natural natural;Revisions raster_dependency_revisions;
 unsigned content_revision=2,shadow_tile_width=128;
 std::array<unsigned,7> raster_proof_rejections{};
 unsigned frame_tile_invalid_shared=0,frame_tile_invalid_appearance=0,frame_tile_invalid_semantic=0,
 frame_tile_invalid_coast=0,frame_tile_invalid_world=0,frame_tile_invalid_anchor=0,frame_tile_invalid_river=0;
 struct Resident {CachedTileGeometry* source=nullptr;
 CachedTileGeometry* resolve(CachedTileGeometry::Handle h){return h.generation?source:nullptr;}}resident_content;
''' + methods + r'''
};
int main(){
 Harness h;CachedGeometryProof p;
 p.appearance_dependencies={{1,7}};p.dependencies={{2,8}};
 p.coast_dependencies={{3,9}};p.world_dependencies={{4,10}};
 assert(h.raster_content_valid(p));auto visits=h.natural.visits;
 for(unsigned i=0;i<100;++i)assert(h.raster_content_valid(p));
 assert(h.natural.visits==visits);
 // Every producer domain and barrier requires the complete proof again.
 for(auto domain:{Domain::appearance,Domain::semantic,Domain::visibility,Domain::coast,Domain::world,Domain::flow,Domain::barrier}){
  h.raster_dependency_revisions.touch(domain,1);assert(h.raster_content_valid(p));assert(h.natural.visits==++visits);
 }
 h.topology_cache.appearance=0;h.raster_dependency_revisions.touch(Domain::appearance,1);
 assert(!h.raster_content_valid(p));h.topology_cache.appearance=7;assert(h.raster_content_valid(p));
 h.world_coast.inputs.value=11;h.raster_dependency_revisions.touch(Domain::world,4);
 assert(!h.raster_content_valid(p));h.world_coast.inputs.value=10;assert(h.raster_content_valid(p));
 h.natural.correct=false;h.raster_dependency_revisions.touch(Domain::flow,4);assert(!h.raster_content_valid(p));
 h.natural.correct=true;assert(h.raster_content_valid(p));
 ++h.topology_cache.scope;assert(!h.raster_content_valid(p));--h.topology_cache.scope;
 ++h.content_revision;assert(!h.raster_content_valid(p));--h.content_revision;
 CachedTileGeometry owner,shared;shared.shared_natural=shared.ground_component=true;
 h.resident_content.source=&shared;owner.natural_content.generation=1;
 owner.appearance_dependencies={{1,7}};owner.dependencies={{2,8}};
 owner.coast_dependencies={{3,9}};owner.world_dependencies={{4,10}};
 c3x_renderer_tile_v1 tile{};assert(h.tile_content_valid(owner,tile));
 auto checks=h.topology_cache.visits,river=h.natural.visits;
 for(unsigned i=0;i<100;++i){++h.topology_cache.epoch;tile.anchor_x+=17;tile.anchor_y-=9;assert(h.tile_content_valid(owner,tile));}
 assert(h.topology_cache.visits==checks && h.natural.visits==river);
 // Shared residency is checked even when the source facts are unchanged.
 h.resident_content.source=nullptr;assert(!h.tile_content_valid(owner,tile));h.resident_content.source=&shared;
 h.topology_cache.present=false;h.raster_dependency_revisions.touch(Domain::semantic,2);assert(!h.tile_content_valid(owner,tile));
 h.topology_cache.present=true;h.raster_dependency_revisions.touch(Domain::semantic,2);assert(h.tile_content_valid(owner,tile));
 // Native/anchored owners cannot borrow the world-only camera shortcut.
 owner.world_ground=owner.world_objects=false;owner.anchor_dependencies={{2,{0,0}}};
 h.topology_cache.observation.occurrence.anchor_x=tile.anchor_x;
 h.topology_cache.observation.occurrence.anchor_y=tile.anchor_y;
 ++h.topology_cache.epoch;assert(h.tile_content_valid(owner,tile));
 ++h.topology_cache.epoch;++tile.anchor_x;assert(!h.tile_content_valid(owner,tile));
 h.topology_cache.observation.occurrence.anchor_x=tile.anchor_x;
 ++h.topology_cache.epoch;assert(h.tile_content_valid(owner,tile));
}
''')


if __name__ == '__main__':
    unittest.main()
