#pragma once
#include "runtime.h"
#include <array>
#include <limits>

namespace c3x_renderer { namespace city_fidelity {
// Receiver coordinates are (world.x, -world.y, world.z*source_z_metric).
// The index is independent of projection and applies to reflected receivers too.
// Float index values are exact: the entire field is bounded below 2^22 records.
struct LightSpatialIndex {
    using Record=std::array<float,4>;
    static constexpr unsigned record_limit=4u*1024u*1024u;
    static constexpr float cell_size=.25f;
    std::vector<Record> records;
    float grid[4]={},info[4]={};
    unsigned cells=0,light_entries=0,blocker_entries=0,max_lights=0,max_blockers=0;
    // Pad in world units for float cell classification, sphere subtraction and
    // the existing slab test's 1e-6 safe-ray perturbation. This only adds work.
    static double guard(double coordinate,double radius){
        return 1e-5+32.*std::numeric_limits<float>::epsilon()*(1.+std::abs(coordinate)+radius);
    }
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
        // Outsize or poorly representable grids take the complete scan path.
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
            // Stable increasing light order, including adjacent boundary cells.
            for(int y=begin[1];y<=end[1];++y)for(int x=begin[0];x<=end[0];++x){
                lights[unsigned(y)*width+unsigned(x)].push_back(i);++entries;++light_entries;
                if(!fits())return false;
            }
            for(unsigned j=0;j<nb;++j){
                if(int(j)==int(field[i*3+2][3]))continue;
                auto const&lo=field[nl*3+j*2];auto const&hi=field[nl*3+j*2+1];bool intersects=true;
                for(unsigned a=0;a<3;++a){
                    double pad=guard(p[a],radius);
                    // Invalid boxes use the original shader test, never pruning.
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
        // Headers hold scalar offsets from the packed index base.
        for(unsigned c=0;c<cells;++c){max_lights=std::max(max_lights,unsigned(lights[c].size()));store(lights[c],c);}
        for(unsigned i=0;i<nl;++i){max_blockers=std::max(max_blockers,unsigned(blockers[i].size()));store(blockers[i],cells+i);}
        return true;
    }
};
} }
