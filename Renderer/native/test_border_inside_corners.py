"""Borders turn inward without a gap (the user's screenshot, October 17).

Civ III captures four border edges per tile, one per edge neighbour. Where a
territory's border turns inward, the two tiles across a vertex's edges share
the owner and only the tile across the vertex does not, so no captured edge
reaches that vertex: the two bands ended at it and left the corner of the
third tile uncoloured. Renderer64 now derives these inside corners from the
neighbours' captured edges (CapturedScene::inside_corners) and the border
shader continues the band around the vertex (test_territory_borders).
"""
from pathlib import Path
import unittest

from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.test_fresh_shared_submission import method

ROOT = Path(__file__).resolve().parents[2]


class BorderInsideCornerTests(unittest.TestCase):
    def test_inside_corners_follow_ownership_and_reach_the_drawn_tile(self):
        window_tile = method((ROOT / 'Renderer/native/c3x_renderer.cpp').read_text(),
                             '    bool window_tile(int x,int y,c3x_renderer_tile_v1& tile)const{')
        run_cpp(r'''
#include "Renderer/native/render_core/captured_scene.h"
#include <array>
#include <cassert>
#include <cstdio>
using namespace c3x_renderer::render_core;
constexpr int W=24,H=24;
constexpr unsigned known=C3X_RENDERER_TILE_VISIBILITY_KNOWN,explored=C3X_RENDERER_TILE_EXPLORED;
int owner[H][W];
bool inside(int x,int y){return x>=0&&y>=0&&x<W&&y<H;}
int at(int x,int y){return inside(x,y)?owner[y][x]:-1;}
// As Civ III's capture (injected read_custom_renderer_tile): an edge per
// diagonal neighbour of another owner, none at invalid map edges.
c3x_renderer_tile_v1 capture(int x,int y){c3x_renderer_tile_v1 t{};t.tile_x=x;t.tile_y=y;
 t.terrain_type=t.real_terrain_type=2;t.city_id=t.city_owner_id=t.city_size=t.city_culture_group=t.city_era=-1;
 t.resource_id=t.resource_class=t.barbarian_tribe_id=-1;t.tile_flags=known|explored|C3X_RENDERER_TILE_PREFETCH|C3X_RENDERER_TILE_TOPOLOGY_HALO;
 int o=owner[y][x];t.territory_owner_id=o;
 if(o>0){t.territory_color_rgb=o==1?0x1dd8dau:0xd83020u;
  for(int e=0;e<4;++e){int nx=x-1+(e&1)*2,ny=y-1+(e/2)*2;if(inside(nx,ny)&&owner[ny][nx]!=o)t.territory_edge_mask|=1u<<e;}}
 return t;}
// Corner k: the neighbours across edge bits ux=k&1?8:1 and vy=k&2?2:4 and
// the tile across their shared vertex.
unsigned expected(int x,int y){int o=owner[y][x];if(o<=0)return 0;unsigned corners=0;
 int ax[4]={x-1,x+1,x-1,x+1},ay[4]={y-1,y+1,y-1,y+1},cx[4]={x-1,x-1,x+1,x+1},cy[4]={y+1,y+1,y-1,y-1};
 int dx[4]={x-2,x,x,x+2},dy[4]={y,y+2,y-2,y};
 for(int k=0;k<4;++k)if(at(ax[k],ay[k])==o&&at(cx[k],cy[k])==o&&inside(dx[k],dy[k])&&at(dx[k],dy[k])!=o)corners|=16u<<k;
 return corners;}
struct Host {
 CapturedScene topology_cache;mutable std::array<unsigned,4> window_skips{};
''' + window_tile + r'''
};
void publish_all(CapturedScene& scene){bool changed=false;
 for(int y=0;y<H;++y)for(int x=y&1;x<W;x+=2)assert(scene.publish(capture(x,y),changed));}
int main(){
 c3x_renderer_frame_v1 f{};f.world_width_tiles=W;f.world_height_tiles=H;f.tile_width=128;f.tile_height=64;
 // Random clustered territories of two owners, away from the map edge.
 unsigned seed=12345,checked=0,corners=0;
 auto next=[&]{seed=seed*1103515245u+12345u;return (seed>>16)&0x7fff;};
 for(int trial=0;trial<60;++trial){
  for(auto& row:owner)for(auto& o:row)o=0;
  for(int blob=0;blob<4;++blob){int bx=4+int(next()%16),by=4+int(next()%16),r=1+int(next()%3),o=1+int(next()%2);
   for(int y=3;y<H-3;++y)for(int x=3;x<W-3;++x)if(((x+y)&1)==0&&std::abs(x-bx)+std::abs(y-by)<=2*r)owner[y][x]=o;}
  Host host;host.topology_cache.publication_scope(f,{1,1,1,1},1);publish_all(host.topology_cache);
  for(int y=0;y<H;++y)for(int x=y&1;x<W;x+=2){
   auto t=capture(x,y);auto derived=host.topology_cache.inside_corners(x,y,t);
   if(derived!=expected(x,y))std::printf("trial %d tile %d,%d derived=%u expected=%u\n",trial,x,y,derived,expected(x,y));
   assert(derived==expected(x,y));++checked;corners+=derived!=0;}
 }
 assert(corners>100);
 // The drawn tile: a notch in one territory. B=(10,10); D=(8,10) across its
 // q=(0,0) vertex is foreign, its neighbours A=(9,9) and C=(9,11) are owned.
 for(auto& row:owner)for(auto& o:row)o=0;
 for(int y=6;y<=14;++y)for(int x=6;x<=14;++x)if(((x+y)&1)==0)owner[y][x]=1;
 owner[10][8]=0;owner[10][6]=0;
 Host host;host.topology_cache.publication_scope(f,{1,1,1,1},1);publish_all(host.topology_cache);
 c3x_renderer_tile_v1 drawn{};assert(host.window_tile(10,10,drawn));
 assert((drawn.territory_edge_mask&15u)==0 && (drawn.territory_edge_mask&16u));
 auto record=host.topology_cache.retained(host.topology_cache.key(10,10))->revision;
 // D joins the territory: only its neighbours' captures change, yet B's
 // drawn tile loses the corner, so its records are placed again.
 owner[10][8]=1;publish_all(host.topology_cache);
 assert(host.window_tile(10,10,drawn) && drawn.territory_edge_mask==0);
 assert(host.topology_cache.retained(host.topology_cache.key(10,10))->revision==record);
 std::printf("PASS inside corners: tiles=%u with_corners=%u drawn_tile_follows_neighbours=1\n",checked,corners);
}
''')


if __name__ == '__main__':
    unittest.main()
