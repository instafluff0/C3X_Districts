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
        // Flat per-cell counts and offsets: a wide light selection spans up to
        // 262144 cells, and one vector per cell cost ~190 ms per rebuild.
        std::vector<std::array<int,4>> spans(nl);
        std::vector<unsigned> cell_count(cells,0);
        std::size_t entries=0;
        auto fits=[&](){return field.size()+cells+nl+(entries+3)/4<=record_limit;};
        if(!fits())return false;
        for(unsigned i=0;i<nl;++i){
            auto const&p=field[i*3];double radius=std::abs(double(p[3]));
            auto& span=spans[i];
            for(unsigned a=0;a<2;++a){
                double pad=guard(p[a],radius);
                span[a]=std::max(0,int(std::floor((double(p[a])-radius-pad-origin[a])/cell_size)));
                span[2+a]=std::min(int(a?height:width)-1,int(std::floor((double(p[a])+radius+pad-origin[a])/cell_size)));
            }
            if(span[2]>=span[0] && span[3]>=span[1]){
                entries+=std::size_t(span[2]-span[0]+1)*std::size_t(span[3]-span[1]+1);
                if(!fits())return false;
            }
            for(int y=span[1];y<=span[3];++y)for(int x=span[0];x<=span[2];++x)++cell_count[unsigned(y)*width+unsigned(x)];
        }
        light_entries=unsigned(entries);
        // Blockers are binned by the same cells; a blocker whose x/y range
        // meets a light's padded range shares at least one of its cells. Each
        // candidate still takes the exact test below, in increasing order.
        std::vector<unsigned> blocker_start(cells+1,0),blocker_cells,everywhere;
        std::vector<std::array<int,4>> blocker_spans(nb);
        for(unsigned j=0;j<nb;++j){
            auto const&lo=field[nl*3+j*2];auto const&hi=field[nl*3+j*2+1];auto& span=blocker_spans[j];
            if(lo[0]>hi[0] || lo[1]>hi[1]){span={0,0,-1,-1};everywhere.push_back(j);continue;}
            // Clamp in double first: blocker boxes are not bounded by the grid.
            auto cell=[&](float value,unsigned axis,unsigned limit){
                return int(std::max(-1.,std::min(double(limit),std::floor((double(value)-origin[axis])/cell_size))));};
            span={std::max(0,cell(lo[0],0,width)),std::max(0,cell(lo[1],1,height)),
                std::min(int(width)-1,cell(hi[0],0,width)),std::min(int(height)-1,cell(hi[1],1,height))};
            for(int y=span[1];y<=span[3];++y)for(int x=span[0];x<=span[2];++x)++blocker_start[unsigned(y)*width+unsigned(x)+1];
        }
        for(unsigned c=0;c<cells;++c)blocker_start[c+1]+=blocker_start[c];
        blocker_cells.resize(blocker_start[cells]);
        {std::vector<unsigned> cursor(blocker_start.begin(),blocker_start.end()-1);
         for(unsigned j=0;j<nb;++j){auto const& span=blocker_spans[j];
            for(int y=span[1];y<=span[3];++y)for(int x=span[0];x<=span[2];++x)blocker_cells[cursor[unsigned(y)*width+unsigned(x)]++]=j;}}
        std::vector<std::vector<unsigned>> blockers(nl);std::vector<unsigned> candidates,seen(nb,0);
        for(unsigned i=0;i<nl;++i){
            auto const&p=field[i*3];double radius=std::abs(double(p[3]));auto const& span=spans[i];
            // A blocker appears in every cell it spans; stamp it once per light.
            candidates.assign(everywhere.begin(),everywhere.end());
            for(int y=span[1];y<=span[3];++y)for(int x=span[0];x<=span[2];++x){auto c=unsigned(y)*width+unsigned(x);
                for(auto at=blocker_start[c];at<blocker_start[c+1];++at){auto j=blocker_cells[at];
                    if(seen[j]!=i+1){seen[j]=i+1;candidates.push_back(j);}}}
            std::sort(candidates.begin(),candidates.end());
            for(unsigned j:candidates){
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
        auto store=[&](unsigned value){records[base-unsigned(field.size())+unsigned(offset/4)][offset%4]=float(value);++offset;};
        // Headers hold scalar offsets from the packed index base. Cell lists
        // keep stable increasing light order.
        {std::vector<std::size_t> cell_offset(cells);
         for(unsigned c=0;c<cells;++c){max_lights=std::max(max_lights,cell_count[c]);records[c]={float(offset),float(cell_count[c]),0,0};cell_offset[c]=offset;offset+=cell_count[c];}
         auto end=offset;
         for(unsigned i=0;i<nl;++i){auto const& span=spans[i];
            for(int y=span[1];y<=span[3];++y)for(int x=span[0];x<=span[2];++x){auto& at=cell_offset[unsigned(y)*width+unsigned(x)];offset=at++;store(i);}}
         offset=end;}
        for(unsigned i=0;i<nl;++i){max_blockers=std::max(max_blockers,unsigned(blockers[i].size()));
            records[cells+i]={float(offset),float(blockers[i].size()),0,0};for(unsigned value:blockers[i])store(value);}
        return true;
    }
};
} }
