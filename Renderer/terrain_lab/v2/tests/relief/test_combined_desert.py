"""Exercise the real combined height kernel with unrelated relief disabled."""
from pathlib import Path
import subprocess
import tempfile
import unittest


class CombinedDesertTests(unittest.TestCase):
    def test_shared_vertices_and_flat_optical_shore(self):
        scene=(Path(__file__).resolve().parents[2]/'shared/frozen_scene.cpp').read_text()

        def function(name):
            start=scene.index('float '+name+'(')
            brace=scene.index('{',start);depth=1;end=brace+1
            while depth:
                depth+=(scene[end]=='{')-(scene[end]=='}');end+=1
            return scene[start:end]

        prelude=r'''
#include <algorithm>
#include <cmath>
#include <cassert>
#include <cstdlib>
struct HeightField {};
struct BiqWindowTile {int column,row,base,real,source_x,source_y;unsigned river_mask=0;};
BiqWindowTile tiles[8][8];
bool dune_scene_enabled=true,biq_scene_enabled=true;
bool volcano_geometry_enabled=false,l13_scene_enabled=false;
int lab_v2_coastal_cliff_join=0;
float lab_v2_relief_scale=1;
namespace labv2 {struct Hooks {void(*shore_sample)(float,float,float*)=nullptr;} hydrology_hooks;}
void rocky_shore(float x,float,float*out){out[0]=2-x;out[1]=.1f;out[2]=1;out[3]=0;}
int lab_v2_continuous_desert=3;
float lab_v2_hill_height_multiplier=1;
float smoothstep01(float x){x=std::clamp(x,0.f,1.f);return x*x*(3-2*x);}
bool is_water_terrain(int x){return x==11;}
BiqWindowTile const* biq_tile_at(int x,int y){return x>=-2&&x<6&&y>=-2&&y<6?&tiles[y+2][x+2]:nullptr;}
BiqWindowTile const* biq_tile_at(float x,float y){return biq_tile_at(int(std::floor(x)),int(std::floor(y)));}
BiqWindowTile const* find_biq_source_tile(int x,int y){return biq_tile_at((x+y)/2,(x-y)/2);}
float biq_signed_shore_distance(BiqWindowTile const&t,float u,float){return std::clamp((t.column+u-2)/.65f,-1.f,1.f);}
float biq_relief_envelope(BiqWindowTile const&,float,float){return 1;}
float biq_hill_support(float,float){return 0;}
float promotion_hill_value(float,float){return 0;}
void biq_chain_relief_sample(float,float,float&a,float&b,float&c,bool){a=b=c=0;}
float biq_mountain_hill_transition_envelope(BiqWindowTile const&,float,float){return 1;}
float biq_river_distance(BiqWindowTile const&,float,float){return 100;}
'''
        body='\n'.join(function(n) for n in ['dune_region_weight','dune_height_value',
            'biq_coastal_relief_envelope','biq_hill_compatibility_envelope',
            'smooth_relief_max','biq_tile_height'])
        assertions=r'''
int main(int argc,char**argv){
 if(argc>1)lab_v2_continuous_desert=std::atoi(argv[1]);
 for(int y=-2;y<6;y++)for(int x=-2;x<6;x++)
  tiles[y+2][x+2]={x,y,x>=2?11:((x==0&&y==1)?2:0),0,x+y,x-y};
 auto h=[](int x,int y,float u,float v){return biq_tile_height(*biq_tile_at(x,y),u,v,nullptr,nullptr);};
 // Every incident tile must submit the same position along shared edges.
 for(int y=0;y<3;y++)for(int x=0;x<3;x++)for(int k=0;k<=32;k++){
  float t=k/32.f;
  assert(std::abs(h(x,y,1,t)-h(x+1,y,0,t))<1e-5);
  assert(std::abs(h(x,y,t,0)-h(x,y+1,t,1))<1e-5);
 }
 // Shore-straddling triangles need an actual flat collar, not just a zero
 // at the contour. Version 2 fails this regression while inland relief stays.
 for(int i=0;i<16;i++)assert(h(1,0,.88f+i*.12f/16,.6f)==2.5f);
 assert(h(1,0,.3f,.6f)>2.6f);
 // The optional source-cliff shoulder also preserves the actual water datum
 // and common-edge vertices while adding support behind the source bodies.
 lab_v2_coastal_cliff_join=4;labv2::hydrology_hooks.shore_sample=rocky_shore;
 for(int i=0;i<12;i++)assert(h(1,0,.965f+i*.035f/12,.6f)==2.5f);
 assert(h(1,0,.70f,.6f)>15.f);
 for(int y=0;y<3;y++)for(int x=0;x<3;x++)for(int k=0;k<=32;k++){
  float t=k/32.f;
  assert(std::abs(h(x,y,1,t)-h(x+1,y,0,t))<1e-5);
  assert(std::abs(h(x,y,t,0)-h(x,y+1,t,1))<1e-5);
 }
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'test.cpp').write_text(prelude+body+assertions)
            subprocess.run(['clang++','-std=c++17',str(p/'test.cpp'),'-o',str(p/'test')],check=True)
            subprocess.run([str(p/'test')],check=True)
            old=subprocess.run([str(p/'test'),'2'],capture_output=True)
            self.assertNotEqual(old.returncode,0,'Prior geometry must fail the flat-shore regression')


if __name__=='__main__':unittest.main()
