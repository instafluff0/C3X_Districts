"""Form input coverage fast paths stay exact and stay fast.

Civ III's thread waits for the input-coverage worker once it falls behind
(performance review 4t). These paths keep that worker ahead without changing a
single hit-test answer:
- an aligned keyed full-screen transfer (the per-tick unit/HUD canvas onto the
  screen) builds retain()'s per-tile result directly;
- a still-current single-upload source (sprite sheet, text raster) never forces
  a regional compaction, while replaced uploads keep the memory bound;
- a canvas grid nobody else references is updated in place, never a grid that
  a copied canvas or deferred transfer still shares;
- the worker client stages commands, and a hit query publishes them first.
Each test compares every sampled answer with a direct per-pixel model.
"""
import importlib.util
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


class HitSceneFastPathTests(unittest.TestCase):
    def test_keyed_screen_transfers_are_exact_and_retain_only_drawn_tiles(self):
        run_cpp(r'''
#include "Renderer/native/native_hit_scene.h"
#include <cassert>
#include <cstdio>
#include <utility>
#include <vector>
using namespace c3x_native_hit;
constexpr int W=640,H=448;constexpr unsigned key=0x7c1f;
int main(){Scene scene;Rect all{0,0,W,H};unsigned value=0;
 std::vector<unsigned> map(W*H,opaque_map),units(W*H,key),screen(W*H,0),sprite(32*32);
 for(unsigned n=0;n<sprite.size();++n)sprite[n]=n%5==0?key:0x0400+n;
 scene.create(1,W,H,Format::rgb555,opaque_map);scene.create(2,W,H,Format::rgb555,key);
 scene.create(3,W,H,Format::rgb555);scene.create(4,32,32,Format::rgb555);scene.upload(4,sprite.data(),sprite.size());
 auto fill=[&](std::vector<unsigned>& image,Rect r,unsigned color){
  for(int y=std::max(0,r.top);y<std::min(H,r.bottom);++y)for(int x=std::max(0,r.left);x<std::min(W,r.right);++x)image[y*W+x]=color;};
 std::uint64_t keyed_transfers=0;
 for(int tick=0;tick<400;++tick){
  if(tick==0){scene.submit({Kind::quantize,1,0,all,all});fill(map,all,opaque_map);}
  // Keyed map overlays accumulate history. Two pinned spots, half a
  // compaction cycle apart, keep the screen deeper than a full-screen
  // transfer may defer, as on a long-lived map canvas.
  int moving_x=(tick*72)%(W-32),moving_y=((tick/8)*40)%(H-32);
  for(auto spot:{std::pair<int,int>{8,8},std::pair<int,int>{72,8},std::pair<int,int>{moving_x,moving_y}}){
   if(spot.first==72&&tick<12)continue;
   int ox=spot.first,oy=spot.second;Rect overlay{ox,oy,ox+32,oy+32};
   scene.submit({Kind::color_key,1,4,overlay,all,0,0,key});
   for(int y=0;y<32;++y)for(int x=0;x<32;++x)if(sprite[y*32+x]!=key)map[(oy+y)*W+ox+x]=sprite[y*32+x];}
  // Civ III clears the unit canvas to its key and redraws a few bodies.
  scene.create(2,W,H,Format::rgb555,key);fill(units,all,key);
  for(int k=0;k<3;++k){int ux=(tick*13+k*200)%(W-24),uy=(k*140+tick*5)%(H-24);Rect body{ux,uy,ux+24,uy+24};
   unsigned color=0x0100+unsigned((tick*7+k)%0x3000);scene.submit({Kind::fill,2,0,body,all,0,0,color});fill(units,body,color);}
  scene.submit({Kind::copy,3,1,all,all,0,0});screen=map;
  scene.submit({Kind::native_image,3,2,all,all,0,0,key,0,0,0,W,H});++keyed_transfers;
  for(int p=0;p<W*H;++p){auto v=units[p];if(v==opaque_map)screen[p]=opaque_map;else if(v!=key)screen[p]=v;}
  Rect hud{W-96,H-40,W-8,H-8};scene.submit({Kind::fill,3,0,hud,all,0,0,0x2222});fill(screen,hud,0x2222);
  if(tick%25==0)for(int y=0;y<H;y+=3)for(int x=0;x<W;x+=3){
   assert(scene.pixel(3,x,y,value)&&value==screen[y*W+x]);assert(scene.pixel(1,x,y,value)&&value==map[y*W+x]);}
 }
 // Retained per-tile work follows what was drawn (bodies, overlay, HUD),
 // not the 70 tiles every keyed full-screen transfer covers.
 auto general=scene.general_tiles();
 std::printf("HIT_KEYED transfers=%llu general_tiles=%llu nodes=%zu\n",(unsigned long long)keyed_transfers,(unsigned long long)general,scene.nodes());
 assert(general<keyed_transfers*24);
}
''')

    def test_current_sources_never_compact_but_replaced_uploads_still_do(self):
        run_cpp(r'''
#include "Renderer/native/native_hit_scene.h"
#include <cassert>
#include <cstdio>
#include <vector>
using namespace c3x_native_hit;
constexpr int W=640,H=448,S=128;
int main(){Scene scene;Rect all{0,0,W,H};unsigned value=0;
 std::vector<unsigned> canvas(W*H,0x0042),sheet(S*S);
 auto paint=[&](int seed){for(int n=0;n<S*S;++n)sheet[n]=(n+seed)%3?65536u|unsigned((n*7+seed)%32767):0u;};
 auto draw=[&](int n){int x=(n*96)%(W-S),y=((n/6)*64)%(H-S);Rect area{x,y,x+S,y+S};
  scene.submit({Kind::native_sprite,1,5,area,all,0,0,0});
  for(int yy=0;yy<S;++yy)for(int xx=0;xx<S;++xx){auto p=sheet[yy*S+xx];if(p&65536)canvas[(y+yy)*W+x+xx]=p&65535;}};
 auto exact=[&]{for(int y=0;y<H;y+=2)for(int x=0;x<W;x+=2)assert(scene.pixel(1,x,y,value)&&value==canvas[y*W+x]);};
 scene.create(1,W,H,Format::rgb555,0x0042);scene.create(5,S,S,Format::bgra32);
 // One upload, still current: drawing from it pins no orphaned memory.
 paint(0);scene.upload(5,sheet.data(),sheet.size());
 for(int n=0;n<60;++n)draw(n);
 exact();auto current=scene.compacted_tiles();assert(current==0);
 // Sprite preparation replaces the upload before every draw: those
 // orphaned arrays still count, so tiles compact and memory stays bounded.
 for(int n=0;n<60;++n){paint(n+1);scene.upload(5,sheet.data(),sheet.size());draw(n+60);}
 exact();auto replaced=scene.compacted_tiles();assert(replaced>0&&scene.bytes()<4u*1024u*1024u);
 // Text rasters are uploaded once, drawn, then retired from the cache.
 // A destroyed source no longer counts as current, so the history that
 // still references it compacts instead of pinning every retired raster.
 for(int n=0;n<300;++n){scene.destroy(5);scene.create(5,S,S,Format::bgra32);paint(1000+n);
  scene.upload(5,sheet.data(),sheet.size());draw(n+200);}
 scene.destroy(5);exact();assert(scene.bytes()<8u*1024u*1024u);
 std::printf("HIT_PAYLOAD current_compactions=%llu replaced_compactions=%llu bytes=%zu\n",
  (unsigned long long)current,(unsigned long long)replaced,scene.bytes());
}
''')

    def test_in_place_grids_preserve_shared_snapshots(self):
        run_cpp(r'''
#include "Renderer/native/native_hit_scene.h"
#include <cassert>
#include <vector>
using namespace c3x_native_hit;
constexpr int W=640,H=448;constexpr unsigned key=0x7c1f;
int main(){Scene scene;Rect all{0,0,W,H};unsigned value=0;
 std::vector<unsigned> a(W*H,0x0100),b,c(W*H,0x0200);
 auto fill=[&](std::vector<unsigned>& image,Rect r,unsigned color){for(int y=r.top;y<r.bottom;++y)for(int x=r.left;x<r.right;++x)image[y*W+x]=color;};
 auto same=[&](Id id,std::vector<unsigned> const& image){for(int y=0;y<H;y+=4)for(int x=0;x<W;x+=4)assert(scene.pixel(id,x,y,value)&&value==image[y*W+x]);};
 scene.create(1,W,H,Format::rgb555,0x0100);scene.create(2,W,H,Format::rgb555);scene.create(3,W,H,Format::rgb555,0x0200);
 for(int n=0;n<40;++n){Rect r{(n*37)%(W-20),(n*23)%(H-20),(n*37)%(W-20)+20,(n*23)%(H-20)+20};
  scene.submit({Kind::fill,1,0,r,all,0,0,0x0300+unsigned(n)});fill(a,r,0x0300+unsigned(n));} // unique grid: in place
 // A full copy shares the source grid; later draws on the source must not reach it.
 scene.submit({Kind::copy,2,1,all,all,0,0});b=a;
 for(int n=0;n<40;++n){Rect r{(n*53)%(W-16),(n*31)%(H-16),(n*53)%(W-16)+16,(n*31)%(H-16)+16};
  scene.submit({Kind::fill,1,0,r,all,0,0,0x0500+unsigned(n)});fill(a,r,0x0500+unsigned(n));}
 same(1,a);same(2,b);
 // A deferred full-screen keyed transfer also holds the source grid.
 Rect hole{0,0,64,64};scene.submit({Kind::fill,1,0,hole,all,0,0,key});fill(a,hole,key);
 scene.submit({Kind::native_image,3,1,all,all,0,0,key,0,0,0,W,H});
 for(int p=0;p<W*H;++p)if(a[p]!=key)c[p]=a[p];
 for(int n=0;n<40;++n){Rect r{(n*41)%(W-12),(n*29)%(H-12),(n*41)%(W-12)+12,(n*29)%(H-12)+12};
  scene.submit({Kind::fill,1,0,r,all,0,0,0x0700+unsigned(n)});fill(a,r,0x0700+unsigned(n));}
 same(1,a);same(2,b);same(3,c);
}
''')

    def test_staged_commands_are_visible_to_immediate_queries(self):
        run_cpp(r'''
#include "Renderer/native/gpu_image_worker_client.h"
#include <cassert>
using namespace c3x_gpu_images;
long long next_id=10;
int execute(c3x_renderer_gpu_images_v1 const*,c3x_renderer_gpu_result_v1* out,unsigned*,unsigned count){
 out->image=next_id++;out->pixel_count=count;return C3X_RENDERER_RESULT_OK;}
int main(){
 c3x_renderer_gpu_frame_v1 frame={};frame.struct_size=sizeof(frame);frame.ticket=frame.session=1;
 WorkerClient client(execute,frame,true);
 auto id=client.create(128,64,Format::rgb555);assert(id);
 Rect all{0,0,128,64};Command fill={Kind::fill,id,0,all,all,0,0,0x1234};
 // Draws only stage locally (no helper call happens); the query must
 // still observe them, in order, before answering.
 assert(client.submit(&fill,1));unsigned value=0;
 assert(client.hit_pixel(id,5,5,value)&&value==0x1234);
 for(unsigned k=0;k<1500;++k){Command dot=fill;dot.area={int(k%128),int(k/128%64),int(k%128)+1,int(k/128%64)+1};dot.color=0x2000+k;
  assert(client.submit(&dot,1));}
 assert(client.hit_pixel(id,1499%128,1499/128%64,value)&&value==0x2000+1499);
 assert(client.hit_pixel(id,3,0,value)&&value==0x2000+3&&client.hit_pixel(id,3,10,value)&&value==0x2000+1283);
 assert(client.hit_pixel(id,127,63,value)&&value==0x1234);
}
''')

    def test_capture_checker_bounds_game_thread_backlog_waits(self):
        spec = importlib.util.spec_from_file_location('waits', ROOT / 'Renderer/tools/check_native_call_waits.py')
        waits = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(waits)
        line = ('42\t63.9\t[1]\t[C3X renderer] stage=native-call-waits calls={} execute_ms=1.0 execute_max_ms=0.1 '
                'backlog_waits=3 backlog_ms={} backlog_max_ms=30.0 queries=0 query_ms=0.0 query_max_ms=0.00 window_ms=2004')
        before = waits.windows([line.format(756, 424.9), line.format(72, 937.4), 'unrelated'])
        after = waits.windows([line.format(717, 121.3), line.format(91, 235.9)])
        # A window with few helper calls still counts: it had the longest waits.
        self.assertEqual(waits.check(before), (False, 937.4, 2))
        self.assertEqual(waits.check(after), (True, 235.9, 2))


if __name__ == '__main__':
    unittest.main()
