"""Execute prepared-area projection and fresh native capture validity."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp


class AreaTests(unittest.TestCase):
    def test_exact_crops_and_complete_capture_proofs(self):
        source = Path('Renderer/native/c3x_renderer.cpp').read_text()
        publication = 'struct PublishedMapFrame {' + source.split('struct PublishedMapFrame {', 1)[1].split('// Cheap, deliberately provisional', 1)[0]
        run_cpp(r'''
#include <cassert>
#include <vector>
#include <memory>
#include <cstring>
#include <algorithm>
#include "Renderer/native/c3x_renderer_api.h"
#include "Renderer/native/prepared_view_area.h"
''' + publication + r'''
int main(){
    std::vector<c3x_renderer_tile_v1> tiles;
    for(int y=30;y<=70;y+=2)for(int x=30;x<=70;x+=2){
        c3x_renderer_tile_v1 t={};t.tile_x=x;t.tile_y=y;t.anchor_x=(x-48)*32;t.anchor_y=(y-48)*16;
        t.city_id=-1;t.tile_flags=(t.anchor_x>=0 && t.anchor_x<=128 && t.anchor_y>=0 && t.anchor_y<=96)?
            C3X_RENDERER_TILE_RENDER:C3X_RENDERER_TILE_TOPOLOGY_HALO|C3X_RENDERER_TILE_PREFETCH;
        t.terrain_type=t.real_terrain_type=2;t.visibility_mask=t.tile_visibility=1;tiles.push_back(t);
    }
    std::vector<unsigned> world(5000,2);
    c3x_renderer_frame_v1 f={};f.api_version=C3X_RENDERER_API_VERSION;f.struct_size=sizeof(f);
    f.target_width=f.clip_right=128;f.target_height=f.clip_bottom=96;f.tile_width=64;f.tile_height=32;
    f.hour=12;f.presentation_frequency=1000;f.presentation_time_ticks=100;
    f.world_width_tiles=f.world_height_tiles=100;f.world_wrap_x=1;
    f.world_topology=world.data();f.world_topology_count=unsigned(world.size());f.world_topology_revision=1;
    f.tiles=tiles.data();f.tile_count=unsigned(tiles.size());
    c3x_renderer_camera_identity_v1 epochs={1,2,3,4};
    c3x_renderer::PreparedViewArea<PublishedMapFrame> area;
    assert(area.prepare(f,epochs));
    std::vector<unsigned> pixels(std::size_t(area.input.target_width)*area.input.target_height);
    for(unsigned i=0;i<pixels.size();++i)pixels[i]=0xff000000u|i;
    std::vector<unsigned> ownership(tiles.size());
    for(unsigned i=0;i<ownership.size();++i)if(area.tiles[i].tile_flags&C3X_RENDERER_TILE_RENDER)
        ownership[i]=C3X_RENDERER_TILE_CUSTOM_TERRAIN_REPLACED;
    c3x_renderer_output_v1 output={};output.api_version=C3X_RENDERER_API_VERSION;output.struct_size=sizeof(output);
    output.width=area.input.target_width;output.height=area.input.target_height;output.stride_bytes=output.width*4;
    output.bgra_pixels=pixels.data();output.replacement_tile_count=unsigned(tiles.size());output.replacement_tile_flags=ownership.data();
    output.visible_animation_count=1;
    auto dependency=area.key(tiles.front());auto absent=(std::uint64_t(99)<<32)|99;
    assert(area.finish(output,{dependency,absent}));
    // The same validity owner selects immutable GPU storage without allocating
    // a cropped CPU bitmap or extending the donor's ambient clock.
    unsigned released=0;
    {
        c3x_renderer::PreparedViewArea<PublishedMapFrame> gpu_area;
        assert(gpu_area.prepare(f,epochs));
        PublishedMapFrame donor;
        PublishedMapFrame::Resident storage={std::shared_ptr<void>(new int(7),[&](void* p){delete static_cast<int*>(p);++released;}),output.width,output.height};
        auto metadata=output;metadata.bgra_pixels=nullptr;
        assert(donor.capture(metadata,0,0,&gpu_area.input,epochs,&storage));
        assert(gpu_area.finish(metadata,{dependency,absent},&donor));
        PublishedMapFrame selected;
        for(auto delta:std::vector<std::pair<int,int>>{{0,0},{32,0},{-32,0},{0,16},{0,-16}}){
            auto fresh=tiles;for(auto& t:fresh){t.anchor_x-=delta.first;t.anchor_y-=delta.second;}
            auto current=f;current.tiles=fresh.data();current.presentation_time_ticks=190;
            assert(gpu_area.project(current,epochs,selected));
            assert(!selected.output.bgra_pixels && selected.pixels.empty() && selected.resident.texture==storage.texture);
            assert(selected.source_x==gpu_area.pad_x+delta.first && selected.source_y==gpu_area.pad_y+delta.second);
            assert(selected.frame.presentation_time_ticks==100);
        }
        auto original=selected.resident.texture;auto original_x=selected.source_x;
        tiles.front().city_id=4;assert(!gpu_area.project(f,epochs,selected));tiles.front().city_id=-1;
        assert(selected.resident.texture==original && selected.source_x==original_x);
        ++epochs.visibility_epoch;assert(!gpu_area.project(f,epochs,selected));--epochs.visibility_epoch;
        gpu_area.clear();assert(selected.resident.texture && !released);
    }
    assert(released==1);
    PublishedMapFrame result;
    for(auto delta:std::vector<std::pair<int,int>>{{0,0},{32,0},{-32,0},{0,16},{0,-16},{64,32}}){
        auto fresh=tiles;
        for(auto& t:fresh){t.anchor_x-=delta.first;t.anchor_y-=delta.second;
            t.tile_flags=(t.anchor_x>=0 && t.anchor_x<=128 && t.anchor_y>=0 && t.anchor_y<=96)?
                C3X_RENDERER_TILE_RENDER:C3X_RENDERER_TILE_TOPOLOGY_HALO|C3X_RENDERER_TILE_PREFETCH;}
        auto current=f;current.tiles=fresh.data();current.presentation_time_ticks=150;
        assert(area.project(current,epochs,result));
        assert(result.frame.presentation_time_ticks==100 && result.frame.tiles!=fresh.data());
        for(int y=0;y<96;++y)for(int x=0;x<128;++x)
            assert(result.pixels[y*128+x]==pixels[(y+area.pad_y+delta.second)*area.input.target_width+x+area.pad_x+delta.first]);
        for(unsigned i=0;i<fresh.size();++i)assert(bool(result.replacements[i])==bool(fresh[i].tile_flags&C3X_RENDERER_TILE_RENDER));
    }
    assert(area.project(f,epochs,result));auto pointer=result.pixels.data();
    auto reordered=tiles;std::reverse(reordered.begin(),reordered.end());auto reordered_frame=f;reordered_frame.tiles=reordered.data();
    assert(!area.project(reordered_frame,epochs,result) && result.pixels.data()==pointer);
    auto wrapped=tiles;for(auto& tile:wrapped)tile.tile_x+=f.world_width_tiles;auto wrapped_frame=f;wrapped_frame.tiles=wrapped.data();
    assert(!area.project(wrapped_frame,epochs,result) && result.pixels.data()==pointer);
    f.visible_animation_count=20;assert(area.project(f,epochs,result));
    assert(result.pixels.data()==pointer && result.output.visible_animation_count==21 && result.frame.visible_animation_count==20);
    f.clip_left=7;f.clip_top=9;f.clip_right=100;f.clip_bottom=80;f.dirty_flags=17;f.presentation_time_ticks=140;
    assert(area.project(f,epochs,result) && result.pixels.data()==pointer && result.frame.presentation_time_ticks==100);
    assert(result.frame.clip_left==7 && result.frame.clip_top==9 && result.frame.clip_right==100 && result.frame.clip_bottom==80);
    assert(result.output.clip_left==7 && result.output.clip_top==9 && result.output.clip_right==100 && result.output.clip_bottom==80);
    assert(result.frame.dirty_flags==17 && result.frame.world_topology==nullptr && result.frame.world_topology_count==0);
    assert(result.frame.tiles==result.occurrences.data() && result.output.request_continuous_redraw);
    auto sampled=result.frame;auto shifted=f;shifted.tile_width*=2;
    assert(!result.refresh_view(shifted) && !std::memcmp(&sampled,&result.frame,sizeof(sampled)));
    f.clip_left=f.clip_top=0;f.clip_right=128;f.clip_bottom=96;f.dirty_flags=0;f.presentation_time_ticks=100;
    f.visible_animation_count=0;assert(area.project(f,epochs,result));
    // A newer selected viewport never advances the untouched wider donor.
    auto fresh_pixels=pixels;for(auto& pixel:fresh_pixels)pixel^=255;
    auto sample=output;sample.bgra_pixels=fresh_pixels.data();f.presentation_time_ticks=180;
    assert(area.project(f,epochs,area.center,&sample,180));
    assert(area.center.frame.presentation_time_ticks==180 && area.input.presentation_time_ticks==100);
    assert(area.center.pixels.front()==(result.pixels.front()^255));
    assert(area.project(f,epochs,result) && result.frame.presentation_time_ticks==100);
    assert(!area.project(f,epochs,result,&sample,181));
    f.presentation_time_ticks=100;
    auto old=result.pixels;
    auto reject=[&]{assert(!area.project(f,epochs,result));assert(result.pixels==old);};
    tiles.front().city_id=4;reject();tiles.front().city_id=-1; // dependency outside visible region
    auto role=tiles.front().tile_flags;tiles.front().tile_flags=C3X_RENDERER_TILE_TOPOLOGY_HALO;reject();tiles.front().tile_flags=role;
    world[4999]=3;reject();world[4999]=2; // unchanged revision is insufficient
    ++epochs.visibility_epoch;reject();--epochs.visibility_epoch;
    tiles.front().fog_status=2;reject();tiles.front().fog_status=0;
    for(auto bit:{C3X_RENDERER_TILE_VISIBILITY_KNOWN,C3X_RENDERER_TILE_EXPLORED,C3X_RENDERER_TILE_VISIBLE}){
        tiles.front().tile_flags^=bit;reject();tiles.front().tile_flags^=bit;
    }
    f.presentation_time_ticks=351;assert(area.project(f,epochs,result));
    assert(result.frame.presentation_time_ticks==100); // retain the honest old ambient sample, not an old camera
    f.presentation_time_ticks=100;
    f.tile_width=128;reject();f.tile_width=64;
    f.hour=0;reject();f.hour=12;
    tiles[0].tile_x=99;tiles[0].tile_y=99;reject();tiles[0].tile_x=tiles[0].tile_y=30;
    assert(area.project(f,epochs,result));
    // Native-only unit animation changes do not invalidate static map content.
    tiles[0].unit_state=9;tiles[0].unit_direction=3;assert(area.project(f,epochs,result));
    for(auto& t:tiles)t.anchor_x-=256;reject();
    c3x_renderer::PreparedViewArea<PublishedMapFrame> unavailable;
    f.target_width=2240;f.target_height=1192;assert(!unavailable.prepare(f,epochs));
    area.clear();assert(!area.map.output.bgra_pixels && area.bytes()==decltype(area){}.bytes());
}
''')


if __name__ == '__main__':
    unittest.main()
