"""The flat city-light index build matches the original per-cell lists exactly.

A wide light selection spans up to 262144 cells. The original build kept one
vector per cell and tested every light against every blocker (~190 ms per
rebuild on the busy save). The flat build must produce identical records,
headers and counters, including invalid and unbounded blocker boxes.
"""
import unittest
from Renderer.native.native_cpp_test import run_cpp

# The original LightSpatialIndex::build, kept verbatim as the oracle.
REFERENCE = r'''
struct Reference {
    using Record=std::array<float,4>;
    static constexpr unsigned record_limit=LightSpatialIndex::record_limit;
    static constexpr float cell_size=LightSpatialIndex::cell_size;
    std::vector<Record> records;
    float grid[4]={},info[4]={};
    unsigned cells=0,light_entries=0,blocker_entries=0,max_lights=0,max_blockers=0;
    static double guard(double coordinate,double radius){return LightSpatialIndex::guard(coordinate,radius);}
    bool build(std::vector<Record> const& field,unsigned nl,unsigned nb){
        *this={};
        if(!nl)return true;
        if(field.size()>record_limit)return false;
        for(auto const& record:field)for(float value:record)if(!std::isfinite(value))return false;
        double low[2]={1e30,1e30},high[2]={-1e30,-1e30};
        for(unsigned i=0;i<nl;++i)for(unsigned a=0;a<2;++a){
            auto const& p=field[i*3];double r=std::abs(double(p[3]));
            double pad=guard(p[a],r);
            low[a]=std::min(low[a],double(p[a])-r-pad);
            high[a]=std::max(high[a],double(p[a])+r+pad);
        }
        double origin[2]={std::floor(low[0]/cell_size)*cell_size,std::floor(low[1]/cell_size)*cell_size};
        double dimensions[2]={std::floor((high[0]-origin[0])/cell_size)+1.,std::floor((high[1]-origin[1])/cell_size)+1.};
        if(dimensions[0]<1 || dimensions[1]<1 || dimensions[0]*dimensions[1]>262144 ||
            std::abs(origin[0])>65536 || std::abs(origin[1])>65536)return false;
        unsigned width=unsigned(dimensions[0]),height=unsigned(dimensions[1]);cells=width*height;
        grid[0]=float(origin[0]);grid[1]=float(origin[1]);grid[2]=1.f/cell_size;grid[3]=float(width);
        info[0]=float(height);info[1]=float(field.size());info[2]=float(field.size()+cells);info[3]=1;
        std::vector<std::vector<unsigned>> lights(cells),blockers(nl);
        std::size_t entries=0;
        auto fits=[&](){return field.size()+cells+nl+(entries+3)/4<=record_limit;};
        if(!fits())return false;
        for(unsigned i=0;i<nl;++i){
            auto const&p=field[i*3];double radius=std::abs(double(p[3]));
            int begin[2],end[2];
            for(unsigned a=0;a<2;++a){
                double pad=guard(p[a],radius);
                begin[a]=std::max(0,int(std::floor((double(p[a])-radius-pad-origin[a])/cell_size)));
                end[a]=std::min(int(a?height:width)-1,int(std::floor((double(p[a])+radius+pad-origin[a])/cell_size)));
            }
            for(int y=begin[1];y<=end[1];++y)for(int x=begin[0];x<=end[0];++x){
                lights[unsigned(y)*width+unsigned(x)].push_back(i);++entries;++light_entries;
                if(!fits())return false;
            }
            for(unsigned j=0;j<nb;++j){
                if(int(j)==int(field[i*3+2][3]))continue;
                auto const&lo=field[nl*3+j*2];auto const&hi=field[nl*3+j*2+1];bool intersects=true;
                for(unsigned a=0;a<3;++a){
                    double pad=guard(p[a],radius);
                    if(lo[a]>hi[a])continue;
                    if(double(hi[a])<double(p[a])-radius-pad || double(lo[a])>double(p[a])+radius+pad)intersects=false;
                }
                if(intersects){blockers[i].push_back(j);++entries;++blocker_entries;if(!fits())return false;}
            }
        }
        records.resize(cells+nl+(entries+3)/4);
        std::size_t offset=0;unsigned base=unsigned(field.size()+cells+nl);
        auto store=[&](std::vector<unsigned>const& list,unsigned header){
            records[header]={float(offset),float(list.size()),0,0};
            for(unsigned value:list){records[base-unsigned(field.size())+unsigned(offset/4)][offset%4]=float(value);++offset;}
        };
        for(unsigned c=0;c<cells;++c){max_lights=std::max(max_lights,unsigned(lights[c].size()));store(lights[c],c);}
        for(unsigned i=0;i<nl;++i){max_blockers=std::max(max_blockers,unsigned(blockers[i].size()));store(blockers[i],cells+i);}
        return true;
    }
};
'''


class FlatLightIndexTests(unittest.TestCase):
    def test_flat_build_matches_original_lists(self):
        run_cpp(r'''
#include "Renderer/native/city_fidelity/light_spatial_index.h"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
using namespace c3x_renderer::city_fidelity;
''' + REFERENCE + r'''
using Record=LightSpatialIndex::Record;
int main(){
 std::mt19937 random(7);double flat_ms=0,reference_ms=0;unsigned built=0;
 for(int trial=0;trial<40;++trial){
  bool wide=trial%4==0;float span=wide?60.f:6.f;
  std::uniform_real_distribution<float> at(-span,span),size(.01f,.6f),radius(.1f,wide?2.5f:.8f),unit(0.f,1.f);
  unsigned nl=wide?2400:unsigned(1+trial*7),nb=wide?3000:unsigned(trial*5);
  std::vector<Record> field(nl*3+nb*2);
  for(unsigned i=0;i<nl;++i){field[i*3]={at(random),at(random),unit(random),radius(random)};
   field[i*3+1]={1,1,1,1};field[i*3+2]={0,0,-1,float(nb?random()%nb:0)};}
  for(unsigned j=0;j<nb;++j){float x=at(random),y=at(random),z=unit(random),w=size(random),h=size(random);
   Record lo={x,y,z,0},hi={x+w,y+h,z+size(random),0};
   if(j%17==3)std::swap(lo[0],hi[0]);          // invalid x never prunes
   if(j%23==5)std::swap(lo[2],hi[2]);          // invalid z
   if(j%29==7){lo[1]=-1e30f;hi[1]=1e30f;}      // unbounded y
   if(j%31==9){lo[0]=4e6f;hi[0]=5e6f;}         // far outside the grid
   field[nl*3+j*2]=lo;field[nl*3+j*2+1]=hi;}
  LightSpatialIndex flat;Reference reference;
  auto t0=std::chrono::steady_clock::now();bool a=flat.build(field,nl,nb);
  auto t1=std::chrono::steady_clock::now();bool b=reference.build(field,nl,nb);
  auto t2=std::chrono::steady_clock::now();
  flat_ms+=std::chrono::duration<double,std::milli>(t1-t0).count();
  reference_ms+=std::chrono::duration<double,std::milli>(t2-t1).count();
  assert(a==b);if(!a)continue;++built;
  assert(flat.records==reference.records);
  assert(!std::memcmp(flat.grid,reference.grid,sizeof(flat.grid)) && !std::memcmp(flat.info,reference.info,sizeof(flat.info)));
  assert(flat.cells==reference.cells && flat.light_entries==reference.light_entries && flat.blocker_entries==reference.blocker_entries);
  assert(flat.max_lights==reference.max_lights && flat.max_blockers==reference.max_blockers);
 }
 assert(built>30);
 std::printf("PASS flat light index: built=%u identical_records=1 flat_ms=%.1f reference_ms=%.1f\n",built,flat_ms,reference_ms);
}
''')


if __name__ == '__main__':
    unittest.main()
