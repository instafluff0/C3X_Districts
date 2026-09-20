"""Durable captured changes are independent of replaceable camera work."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class ScenePublicationTests(unittest.TestCase):
    def test_capture_coalescing_retirement_and_old_view_isolation(self):
        run_cpp(r'''
#include "Renderer/native/render_core/scene_publication.h"
#include <cassert>
using namespace c3x_renderer::render_core;
int main(){
 ScenePublication journal;CapturedScene scene;unsigned topology[]={1,2};
 c3x_renderer_tile_v1 tile{};tile.tile_x=2;tile.tile_y=2;tile.city_id=17;tile.resource_id=3;
 tile.tile_flags=C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_VISIBLE;
 auto original=tile;
 c3x_renderer_frame_v1 f{};f.tiles=&tile;f.tile_count=1;f.world_width_tiles=f.world_height_tiles=100;
 f.world_wrap_x=1;f.world_topology=topology;f.world_topology_count=2;f.world_topology_revision=1;
 f.presentation_time_ticks=125;f.presentation_frequency=1000;f.hour=12;f.season=1;
 c3x_renderer_camera_identity_v1 id{1,1,1,1};
 assert(journal.capture(f,id));auto first=journal.state();
 bool changed=false;assert(journal.apply(scene,changed)&&changed);
 auto key=scene.key(2,2);auto revision=scene.retained(key)->revision;
 assert(scene.begin(f)&&scene.update(tile,2,2,2,100));scene.finish();scene.attach(tile,{2,10});
 assert(scene.retained(key)->compiled.generation==10);auto visible_revision=scene.retained(key)->visibility_revision;
 tile.anchor_x=99;f.presentation_time_ticks=150;
 assert(journal.capture(f,id)&&journal.state()->tiles==first->tiles&&journal.tiles_reused==1);
 assert(journal.apply(scene,changed)&&!changed&&scene.retained(key)->revision==revision);
 tile.anchor_x=256;tile.tile_x-=100;f.presentation_time_ticks=200;tile.tile_flags&=~C3X_RENDERER_TILE_VISIBLE;++id.visibility_epoch;
 assert(journal.capture(f,id));assert(first->topology==journal.state()->topology);
 assert(journal.apply(scene,changed)&&!changed&&scene.retained(key)->revision==revision);
 assert(!(scene.retained(key)->visibility_flags&C3X_RENDERER_TILE_VISIBLE)&&scene.retained(key)->visibility_revision>visible_revision);
 assert(scene.retained(key)->compiled.generation==10);
 // Capture A removes objects, then its camera is superseded by a disjoint B.
 // Both changes must reach the world even though neither camera was rendered.
 tile.city_id=tile.resource_id=-1;topology[0]=9;++f.world_topology_revision;
 assert(journal.capture(f,id));tile.tile_x=4;tile.city_id=20;
 assert(journal.capture(f,id));tile.city_id=999;topology[0]=99;
 assert(journal.apply(scene,changed)&&changed);
 assert(scene.retained(key)->appearance.city_id==-1&&scene.retained(key)->appearance.resource_id==-1);
 assert(!scene.retained(key)->compiled.generation && scene.retained(key)->revision>revision);
 assert(scene.retained(scene.key(4,2))->appearance.city_id==20);
 assert(first->metadata.presentation_time_ticks==125&&first->identity.visibility_epoch==1&&(*first->topology)[0]==1);
 assert((*journal.state()->topology)[0]==9&&journal.state()->metadata.tiles==nullptr&&journal.state()->metadata.world_topology==nullptr);
 // Sampling the previous displayed view cannot republish obsolete objects or
 // bind a compiled representation to the current authoritative revision.
 f.tiles=&original;assert(scene.begin(f)&&scene.update(original,2,2,2,100));scene.finish();
 assert(scene.current(key)->occurrence.city_id==17&&scene.retained(key)->appearance.city_id==-1);
 assert(!scene.appearance_revision(key));scene.attach(original,{9,90});assert(!scene.retained(key)->compiled.generation);
 auto count=scene.size();f.tiles=&tile;tile.tile_flags=C3X_RENDERER_TILE_TOPOLOGY_HALO;
 assert(journal.capture(f,id)&&journal.apply(scene,changed)&&!changed&&scene.size()==count);
 // Viewer/configuration and map replacement retire the previous world scope.
 ++id.viewer_epoch;assert(journal.capture(f,id)&&journal.apply(scene,changed)&&changed&&!scene.retained(key));
 auto config=journal.state()->configuration;journal.reset();tile.tile_flags=C3X_RENDERER_TILE_RENDER;
 assert(journal.capture(f,id)&&journal.state()->configuration>config&&journal.apply(scene,changed)&&changed);
 ++id.map_epoch;f.tile_count=0;assert(journal.capture(f,id)&&journal.apply(scene,changed)&&changed&&!scene.size());
 assert(first->metadata.presentation_time_ticks==125); // old metadata remains immutable
}
''')

    def test_bounded_transaction_and_duplicates(self):
        run_cpp(r'''
#include "Renderer/native/render_core/scene_publication.h"
#include <cassert>
using namespace c3x_renderer::render_core;
int main(){
 c3x_renderer_tile_v1 tiles[2]{};tiles[0].tile_flags=C3X_RENDERER_TILE_RENDER;
 tiles[0].tile_x=2;tiles[0].city_id=1;tiles[1]=tiles[0];tiles[1].tile_x=102;tiles[1].anchor_x=777;
 c3x_renderer_frame_v1 f{};f.world_width_tiles=f.world_height_tiles=100;f.world_wrap_x=1;
 f.tiles=tiles;f.tile_count=2;ScenePublication journal(8192);CapturedScene scene;
 assert(journal.capture(f,{}));auto state=journal.state();auto bytes=journal.bytes();
 for(int i=0;i<100;++i){f.presentation_time_ticks=i;assert(journal.capture(f,{}));assert(journal.bytes()==bytes);}
 assert(journal.tiles_reused==100);
 state=journal.state();tiles[1].city_id=8;assert(!journal.capture(f,{}));assert(journal.state()==state&&journal.bytes()==bytes);
 f.tile_count=8193;assert(!journal.capture(f,{}));assert(journal.state()==state);
 f.tile_count=1;f.world_topology_count=1;f.world_topology=nullptr;assert(!journal.capture(f,{}));
 f.world_topology_count=0;bool changed=false;assert(journal.apply(scene,changed)&&scene.retained(scene.key(2,0))->appearance.city_id==1);
 scene={};assert(journal.capture(f,{}));assert(journal.apply(scene,changed)&&changed);
 assert(scene.retained(scene.key(2,0))->authoritative);
 assert(journal.capture(f,{})&&journal.apply(scene,changed)&&!changed);
 scene={};auto reused=journal.tiles_reused;assert(journal.capture(f,{})&&journal.tiles_reused==reused+1);
 assert(journal.apply(scene,changed)&&changed&&scene.retained(scene.key(2,0))->authoritative);
 // Admission failure never drains or replaces accepted changes.
 ScenePublication tight(128);assert(!tight.capture(f,{})&&!tight.state()&&!tight.ready());
 // Failed worker admission retains accepted values, suppresses output and
 // waits for a later capture rather than retrying continuously.
 struct Exhausted:CapturedScene {bool publish(c3x_renderer_tile_v1 const&,bool&){throw std::bad_alloc();}} exhausted;
 assert(journal.capture(f,{}));state=journal.state();
 assert(!journal.apply(exhausted,changed)&&!journal.ready()&&journal.state()==state);
 tiles[0].tile_x=4;tiles[0].city_id=7;assert(journal.capture(f,{}));CapturedScene recovered;
 assert(journal.apply(recovered,changed)&&recovered.retained(recovered.key(2,0))->appearance.city_id==1);
 assert(recovered.retained(recovered.key(4,0))->appearance.city_id==7);
}
''')


if __name__ == '__main__':
    unittest.main()
