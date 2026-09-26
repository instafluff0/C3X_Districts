"""Background world preparation priority, lifetime and cancellation contracts."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class WorldPreparationScheduleTests(unittest.TestCase):
    def test_demand_reprioritizes_without_rebuilding_completed_regions(self):
        run_cpp(r'''
#include "Renderer/native/render_core/world_preparation_region.h"
#include <cassert>
#include <set>
using namespace c3x_renderer::render_core;
int main(){
 c3x_renderer_frame_v1 f{};f.world_width_tiles=f.world_height_tiles=64;
 f.tile_width=128;f.tile_height=64;f.target_width=800;f.target_height=600;
 c3x_renderer_tile_v1 tile{};tile.tile_flags=C3X_RENDERER_TILE_RENDER;
 tile.tile_x=tile.tile_y=4;f.tiles=&tile;f.tile_count=1;
 WorldPreparationSchedule q;q.prioritize(f);q.configure(f,1,1,1);
 assert(q.next()==0);auto cancelled=q.next();q.configure(f,1,1,1);
 assert(q.next()==cancelled && !q.completed);q.finish(true);
 tile.tile_x=tile.tile_y=60;q.prioritize(f);q.configure(f,1,1,1);
 assert(q.next()==63 && q.completed==1);std::set<unsigned> seen{0};
 while(!q.empty()){assert(seen.insert(q.next()).second);q.finish(true);q.configure(f,1,1,1);}
 assert(seen.size()==WorldPreparationRegion::count(f) && q.completed==seen.size());
 q.prioritize(f);q.configure(f,1,1,1);assert(q.empty());
 // A local edit re-arms only the cores whose halo uses that tile.
 q.invalidate(f,30,30);auto affected=seen.size()-q.completed;
 assert(affected>0 && affected<seen.size());
 while(!q.empty()){q.finish(true);q.configure(f,1,1,1);}
 assert(q.completed==seen.size());
 // Projection, device or world lifetime changes re-arm the whole schedule.
 q.configure(f,1,1,2);assert(q.completed==0);q.finish(false);assert(q.unavailable==1);
 q.configure(f,1,1,3);assert(q.completed==0 && q.unavailable==0);
 q.finish(true);f.tile_width=160;q.configure(f,1,1,3);assert(q.completed==0);
 q.finish(true);f.target_width=1024;q.configure(f,1,1,3);assert(q.completed==0);
 q.finish(true);q.configure(f,2,1,3);assert(q.completed==0);
 // Wrapped edge occurrences remain part of coverage, with no duplicate region.
 f.world_wrap_x=f.world_wrap_y=1;q.configure(f,3,1,3);seen.clear();
 while(!q.empty()){assert(seen.insert(q.next()).second);q.finish(true);}
 assert(seen.size()==WorldPreparationRegion::count(f));
}
''')


if __name__ == '__main__':
    unittest.main()
