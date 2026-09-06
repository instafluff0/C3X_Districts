"""Exercise actual combined coordinate kernels, including crop/wrap identity."""
from pathlib import Path
import subprocess
import tempfile
import unittest


class SourceMappingTests(unittest.TestCase):
    def test_volcano_channels_and_hill_crop_identity(self):
        scene=(Path(__file__).resolve().parents[2]/'shared/frozen_scene.cpp').read_text()
        def block(marker):
            start=scene.index(marker);brace=scene.index('{',start);depth=1;end=brace+1
            while depth:
                depth+=(scene[end]=='{')-(scene[end]=='}');end+=1
            return scene[start:end]
        volcano=block('void biq_volcano_sample(')
        mountain=block('void biq_mountain_sample(')
        chain=block('void biq_chain_relief_sample(')
        hill=block('if (lab_v2_direct_hill_source)')
        source=r'''
#include <algorithm>
#include <cmath>
#include <cassert>
#include <vector>
struct BiqWindowTile {int column=0,row=0,source_x=0,source_y=0,real=10;};
struct HeightField {bool second=false;float sample(float u,float v)const{return second?v:u;}};
bool volcano_geometry_enabled=true,lab_v2_volcano_source_mapping=true;
bool lab_v2_direct_hill_source=true;
float lab_v2_relief_scale=1;
float lab_v2_volcano_scale=1;
HeightField hf,bf{true};
HeightField const* promotion_volcano_height_field=&hf;
HeightField const* promotion_volcano_blend_field=&bf;
HeightField const* promotion_mountain_height_fields[5]={&hf,&hf,&hf,&hf,&hf};
HeightField const* promotion_mountain_blend_fields[5]={&bf,&bf,&bf,&bf,&bf};
std::vector<BiqWindowTile> source_tiles;
BiqWindowTile const* biq_tile_at(int x,int y){for(auto const&t:source_tiles)if(t.column==x&&t.row==y)return &t;return nullptr;}
float smoothstep01(float x){x=std::clamp(x,0.f,1.f);return x*x*(3-2*x);}
struct Window {std::vector<BiqWindowTile> tiles;} biq_window;
'''+volcano+mountain+chain+'\nfloat hill(float world_x,float world_y,HeightField const* field){'+hill+'return -1;}'+r'''
int main(){
 BiqWindowTile tile{5,5,70,50,10};
 // The height sampler must use the same local-v orientation/footprint that
 // dormant, active, slope and specular material channels now share.
 for(float scale:{1.f,1.3f})for(float u:{.2f,.4f,.65f,.8f})for(float v:{.2f,.4f,.65f,.8f}){
  lab_v2_relief_scale=scale;lab_v2_volcano_scale=scale;
  float height,blend;biq_volcano_sample(tile,u,v,height,blend);
  assert(std::abs(height-(.5f+(u-.5f)*.62f/scale))<1e-6f);
  assert(std::abs(blend-(.5f+(v-.5f)*.62f/scale))<1e-6f);
 }
 // The same source point moves horizontally and vertically by the same scale.
 // The world skirt crosses the ownership edge, ends before another tile's
 // center, and remains continuous when the receiving tile changes.
 for(int kind:{6,10}){
  source_tiles={{0,0,70,50,kind}};
  float h,b,d,hh,bb,dd;lab_v2_relief_scale=lab_v2_volcano_scale=1;
  biq_chain_relief_sample(.65f,.5f,h,b,d);
  lab_v2_relief_scale=lab_v2_volcano_scale=1.3f;
  biq_chain_relief_sample(.5f+.15f*1.3f,.5f,hh,bb,dd);
  assert(std::abs(h-hh)<1e-6 && std::abs(d*1.3f-dd)<1e-4);
  biq_chain_relief_sample(1.2f,.5f,h,b,d);assert(d>0);
  biq_chain_relief_sample(1.5f,.5f,h,b,d);assert(d==0);
  biq_chain_relief_sample(1.f-1e-6f,.5f,h,b,d);
  biq_chain_relief_sample(1.f+1e-6f,.5f,hh,bb,dd);assert(std::abs(d-dd)<.001f);
 }
 source_tiles.clear();lab_v2_relief_scale=1;
 // Two crops containing one physical point must sample one source coordinate.
 biq_window.tiles={{0,0,64,38,2}};
 float a=hill(7.25f,4.125f,&hf),b=hill(7.25f,4.125f,&bf);
 biq_window.tiles={{0,0,67,39,2}};
 assert(std::abs(a-hill(5.25f,3.125f,&hf))<1e-6f);
 assert(std::abs(b-hill(5.25f,3.125f,&bf))<1e-6f);
 biq_window.tiles={{0,0,164,38,2}};
 assert(std::abs(a-hill(7.25f,4.125f,&hf))<1e-6f);
 assert(std::abs(b-hill(7.25f,4.125f,&bf))<1e-6f);
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'test.cpp').write_text(source)
            subprocess.run(['clang++','-std=c++17',str(p/'test.cpp'),'-o',str(p/'test')],check=True)
            subprocess.run([str(p/'test')],check=True)


if __name__=='__main__':unittest.main()
