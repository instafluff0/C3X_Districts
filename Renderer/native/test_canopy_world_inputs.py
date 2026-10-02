"""Forest exclusions use immutable permitted city facts for every canopy path."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp

ROOT=Path(__file__).resolve().parents[2]


class CanopyWorldInputTests(unittest.TestCase):
    def test_offscreen_city_exclusions_and_partial_city_changes(self):
        source=(ROOT/'Renderer/native/source_fidelity/geometry.h').read_text()
        start=source.index('        std::vector<BuildingBounds> buildings;')
        end=source.index('        auto emit_forest_instance=',start)
        loop=source[start:end].replace('        std::vector<BuildingBounds> buildings;','')
        run_cpp(r'''
#include "Renderer/native/render_core/captured_scene.h"
#include <cassert>
#include <cmath>
#include <array>
#include <vector>
using namespace c3x_renderer::render_core;
struct BuildingBounds {float x0,y0,x1,y1;};
struct Instance {std::array<float,2> offset{};std::array<float,4> bounds{};};
struct Composition {std::vector<Instance> instances;};
struct Placement {unsigned asset_index=0;float scale=1;};
struct Group {std::vector<Placement> placements;};
struct Vertex {float position[3]{};};
struct Asset {std::vector<Vertex> vertices;};
struct Bundle {std::vector<Asset> assets;};
namespace c3x_renderer {
Group const* find_feature_group(Bundle const&,char const*){return nullptr;}
float stable_random(unsigned){return 0;}
}
int main(){
 CapturedScene scene;auto& topology_cache=scene;
 c3x_renderer_frame_v1 frame{};frame.world_width_tiles=frame.world_height_tiles=64;
 scene.publication_scope(frame,{1,1,1,1},1);
 c3x_renderer_tile_v1 tile{};tile.tile_x=16;tile.tile_y=8;tile.real_terrain_type=7;
 tile.city_id=-1;tile.tile_flags=C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED;
 auto city=tile;city.tile_x=18;city.tile_y=8;city.real_terrain_type=2;city.city_id=17;city.city_size=1;
 city.tile_flags=C3X_RENDERER_TILE_TOPOLOGY_HALO|C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED|C3X_RENDERER_TILE_CITY_BODY_KNOWN;
 bool changed=false;assert(scene.publish(tile,changed)&&scene.publish(city,changed));
 frame.tiles=&tile;frame.tile_count=1;assert(scene.begin(frame));
 assert(scene.update(tile,2,-1,2,CapturedScene::topology(tile)));scene.finish();
 assert(!scene.current(scene.key(18,8))); // the neighboring city is offscreen
 auto ground_observations=scene.compilation_view(true);
 auto observed_coordinate_key=[&](int x,int y){return scene.key(x,y);};
 int nc=12,nr=4;Bundle city_bundle;Composition composition;
 auto selected_city=[&](auto const& record,int,int){
  composition.instances={{{.1f,.2f},{0,0,float(record.city_size)*.25f,.5f}}};
  return &composition;
 };
 std::vector<BuildingBounds> buildings;
 auto collect=[&]()->bool{buildings.clear();
'''+loop+r'''
 return true;};
 assert(collect()&&buildings.size()==1);
 // Neighbor (18,8) maps to (13,5); exclusion follows the permitted city recipe.
 assert(std::abs(buildings[0].x1-13.85f)<.0001f);
 assert(!scene.retained(scene.key(18,8))->authoritative);
 auto old=ground_observations;
 city.city_size=2;assert(scene.publish(city,changed));
 ground_observations=scene.compilation_view(true);assert(collect()&&buildings.size()==1);
 assert(std::abs(buildings[0].x1-14.10f)<.0001f);
 ground_observations=old;assert(collect()&&std::abs(buildings[0].x1-13.85f)<.0001f);
 city.city_id=-1;assert(scene.publish(city,changed));
 ground_observations=scene.compilation_view(true);assert(collect()&&buildings.empty());
}
''')

    def test_every_canopy_path_records_neighbor_appearance(self):
        source=(ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
        start=source.index('            bool const canopy_city_dependencies=')
        end=source.index('            float left =',start)
        proof=source[start:end]
        start=source.index('            if(canopy_city_dependencies)for')
        end=source.index('            // World-space GPU data',start)
        key=source[start:end]
        run_cpp(r'''
#include "Renderer/native/render_core/captured_scene.h"
#include <cassert>
#include <array>
#include <unordered_map>
using namespace c3x_renderer::render_core;
int main(){
 CapturedScene topology_cache;c3x_renderer_frame_v1 frame{};frame.world_width_tiles=frame.world_height_tiles=64;
 topology_cache.publication_scope(frame,{1,1,1,1},1);
 auto coordinate_key=[&](int x,int y){return topology_cache.key(x,y);};
 for(auto state:std::vector<std::array<int,4>>{{7,0,0,1},{8,0,0,1},{5,7,0,1},{5,8,0,1},{6,0,7,1},{10,0,8,1},{2,0,0,0}}){
  c3x_renderer_tile_v1 tile{};tile.tile_x=16;tile.tile_y=8;tile.real_terrain_type=state[0];
  int hill_vegetation=state[1],raised_vegetation=state[2];bool retained_world=true;
  std::unordered_map<std::uint64_t,std::uint64_t> appearance_dependencies;
'''+proof+r'''
  assert(canopy_city_dependencies==bool(state[3]));
  assert(appearance_dependencies.size()==(state[3]?25u:0u));
  if(state[3])assert(appearance_dependencies.count(coordinate_key(18,8)));
  struct Observations {unsigned calls=0;CapturedScene::Observation const* current(std::uint64_t){++calls;return nullptr;}} ground_observations;
  auto tile_content_signature=[](auto const&){return std::uint64_t(7);};
  std::uint64_t natural_key=1469598103934665603ull;
'''+key+r'''
  assert(ground_observations.calls==(state[3]?25u:0u));
 }
}
''')

if __name__=='__main__':unittest.main()
