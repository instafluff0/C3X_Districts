#include "city_fidelity/light_spatial_index.h"
#include <cassert>
#include <random>
#include <cstdio>
using namespace c3x_renderer::city_fidelity;
using Record=LightSpatialIndex::Record;
std::vector<unsigned> list(LightSpatialIndex const&index,unsigned nl,unsigned header){
    auto h=index.records[header];std::vector<unsigned> result;
    for(unsigned e=0;e<unsigned(h[1]);++e){unsigned scalar=unsigned(h[0])+e;
        result.push_back(unsigned(index.records[index.cells+nl+scalar/4][scalar%4]));}
    assert(std::is_sorted(result.begin(),result.end()));return result;
}
bool blocks(Record const&p,Record const&r,Record const&low,Record const&high){
    float near_t=-1e30f,far_t=1e30f;
    for(unsigned a=0;a<3;++a){float ray=r[a]-p[a];float safe=std::max(std::abs(ray),1e-6f)*(ray<0?-1.f:1.f);
        float lo=(low[a]-p[a])/safe,hi=(high[a]-p[a])/safe;
        near_t=std::max(near_t,std::min(lo,hi));far_t=std::min(far_t,std::max(lo,hi));}
    return far_t>=std::max(near_t,.001f) && near_t<.995f;
}
Record illumination(std::vector<Record>const&field,unsigned nl,unsigned nb,Record const&r,Record const&n,
                    LightSpatialIndex const*index){
    std::vector<unsigned> selected;Record sum={};
    if(index){
        int x=int(std::floor((r[0]-index->grid[0])*index->grid[2]));
        int y=int(std::floor((r[1]-index->grid[1])*index->grid[2]));
        if(x<0 || y<0 || x>=int(index->grid[3]) || y>=int(index->info[0]))return sum;
        selected=list(*index,nl,unsigned(y)*unsigned(index->grid[3])+unsigned(x));
    }else for(unsigned i=0;i<nl;++i)selected.push_back(i);
    for(unsigned i:selected){
        auto p=field[3*i];auto color=field[3*i+1];auto direction=field[3*i+2];float d2=0;Record to={};
        for(unsigned a=0;a<3;++a){to[a]=p[a]-r[a];d2+=to[a]*to[a];}
        if(d2>=p[3]*p[3])continue;
        float face=0,diffuse=0;
        for(unsigned a=0;a<3;++a){to[a]/=std::sqrt(std::max(d2,1e-8f));face-=direction[a]*to[a];diffuse+=n[a]*to[a];}
        face=std::clamp(face,0.f,1.f);diffuse=std::clamp(diffuse,0.f,1.f);if(face*diffuse<=0)continue;
        auto candidates=std::vector<unsigned>{};if(index)candidates=list(*index,nl,index->cells+i);
        else for(unsigned j=0;j<nb;++j)candidates.push_back(j);
        bool blocked=false;for(unsigned j:candidates){if(int(j)==int(direction[3]))continue;
            if(blocks(p,r,field[nl*3+j*2],field[nl*3+j*2+1])){blocked=true;break;}}
        if(blocked)continue;
        float nd=d2/(p[3]*p[3]);float attenuation=std::pow(1-nd,2)/(1+8*nd);
        for(unsigned a=0;a<3;++a)sum[a]+=color[a]*color[3]*attenuation*face*diffuse;
    }
    return sum;
}
int main(){
    std::mt19937 rng(821523);std::uniform_real_distribution<float> u(-1,1);
    unsigned nl=120,nb=70;std::vector<Record> field(nl*3+nb*2);
    for(unsigned i=0;i<nl;++i){float cluster=i%3==0?-12.f:i%3==1?0.f:14.f;
        field[i*3]={cluster+u(rng),cluster+u(rng),u(rng)*.3f,.01f+std::abs(u(rng))*1.4f};
        field[i*3+1]={.7f,.4f,.2f,.8f};field[i*3+2]={u(rng),u(rng),u(rng),float(i%nb)};
    }
    for(unsigned j=0;j<nb;++j){float cluster=j%3==0?-12.f:j%3==1?0.f:14.f;
        Record p={cluster+u(rng),cluster+u(rng),u(rng)*.3f,0};
        for(unsigned a=0;a<3;++a){field[nl*3+j*2][a]=p[a]-.1f;field[nl*3+j*2+1][a]=p[a]+.1f;}
    }
    // Exact cell/range boundaries, owner exclusion, and cross-city boxes.
    field[0]={-.25f,-.5f,0,.5f};field[2]={1,0,0,0};
    field[nl*3]={-.26f,-.51f,-.01f,0};field[nl*3+1]={-.24f,-.49f,.01f,0};
    field[nl*3+2]={-.5f,-.7f,-.1f,0};field[nl*3+3]={-.45f,-.3f,.1f,0};
    LightSpatialIndex index;assert(index.build(field,nl,nb));
    unsigned probes=0;
    auto check=[&](Record r){
        Record normal={u(rng),u(rng),u(rng),0};
        assert(illumination(field,nl,nb,r,normal,nullptr)==illumination(field,nl,nb,r,normal,&index));++probes;
        // Every original successful ray in every light's sphere must be present,
        // independent of diffuse/face rejection and early blocker termination.
        for(unsigned i=0;i<nl;++i){auto p=field[i*3];float d2=0;
            for(unsigned a=0;a<3;++a)d2+=(r[a]-p[a])*(r[a]-p[a]);if(d2>=p[3]*p[3])continue;
            auto candidates=list(index,nl,index.cells+i);
            for(unsigned j=0;j<nb;++j)if(int(j)!=int(field[i*3+2][3]) && blocks(p,r,field[nl*3+j*2],field[nl*3+j*2+1]))
                assert(std::binary_search(candidates.begin(),candidates.end(),j));
        }
    };
    for(unsigned k=0;k<18000;++k){auto p=field[(k%nl)*3];check({p[0]+u(rng)*1.5f,p[1]+u(rng)*1.5f,p[2]+u(rng)*1.5f,0});}
    for(unsigned i=0;i<nl;++i)for(unsigned a=0;a<3;++a)for(float sign:{-1.f,1.f}){
        auto r=field[i*3];r[a]+=sign*r[3];check(r);r[a]=std::nextafter(r[a],field[i*3][a]);check(r);}
    for(float x:{-.5f,-.25f,0.f,.25f,.5f})for(float y:{-.5f,-.25f,0.f,.25f,.5f})check({x,y,0,0});
    assert(index.build({},0,0));assert(index.records.empty());
    auto extreme=field;extreme[0][3]=10000;assert(!index.build(extreme,nl,nb));
    extreme=field;extreme[0][0]=std::numeric_limits<float>::infinity();assert(!index.build(extreme,nl,nb));
    // A different scene at the same storage address builds different lists.
    assert(index.build(field,nl,nb));auto old=index.records;field[0][0]+=5;
    assert(index.build(field,nl,nb));assert(old!=index.records);
    std::printf("PASS conservative city light index: %u exact illumination/ray probes, boundaries, negative/wrapped occurrences, owner/cross-city blockers, empty/failure/mutation\n",probes);
}
