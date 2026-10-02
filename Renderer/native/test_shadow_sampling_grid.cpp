#include "render_core/shadow_sampling_grid.h"
#include "render_core/frame_working_set.h"
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <random>
#include <vector>

using Grid=c3x_renderer::render_core::ShadowSamplingGrid;
using Pair=std::array<int,2>;
std::array<float,12> light={.8f,.6f,0,0,-.36f,.48f,.8f,0,.48f,-.64f,.6f,0};
std::uint64_t checks=0;
void require(bool value){++checks;assert(value);}
float legacy_span(float minimum,float maximum){auto origin=std::floor((minimum-4)/2)*2;
    return std::ceil((maximum+4-origin)/2)*2;}

// The host field is an independent analytic blocker, not a triangle/GPU oracle.
// The per-page path stores the exact global cell's value in any physical slot.
// Fine checkerboard cutouts and a sloping receiver make phase/PCF errors visible.
float blocker(int x,int y) {
    unsigned pattern=unsigned(x)*73856093u^unsigned(y)*19349663u;
    if((pattern&7u)==0)return -1e6f;
    return float(int(pattern%29u)-14)*.002f + float(x)*.0000005f;
}
struct Field {
    Grid grid;
    std::map<Pair,float> cells;
    explicit Field(Grid value):grid(value){}
    float load(int x,int y) {
        auto px=int(std::floor(double(x)/Grid::page_texels));
        auto py=int(std::floor(double(y)/Grid::page_texels));
        require(grid.slot(px,py)>=0);
        Pair physical={grid.slot(px,py), (y-py*int(Grid::page_texels))*int(Grid::page_texels)+x-px*int(Grid::page_texels)};
        auto found=cells.find(physical);
        if(found==cells.end())found=cells.emplace(physical,blocker(x,y)).first;
        return found->second;
    }
};
unsigned independent_visibility(double u,double v,float z,Grid const& grid) {
    double x=u*grid.inverse_pitch(0),y=v*grid.inverse_pitch(1);
    int center_x=int(std::floor(x)),center_y=int(std::floor(y));unsigned lit=0;
    for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){
        int sx=center_x+dx,sy=center_y+dy;
        double receiver=double(z)+.0002*(sx+.5-x)-.0003*(sy+.5-y);
        lit+=double(blocker(sx,sy))<=receiver+std::max(grid.pitch(0),grid.pitch(1))*.35;
    }
    return lit;
}
unsigned stored_visibility(double u,double v,float z,Field& field) {
    double x=u*field.grid.inverse_pitch(0),y=v*field.grid.inverse_pitch(1);
    int center_x=int(std::floor(x)),center_y=int(std::floor(y));unsigned lit=0;
    for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){
        int sx=center_x+dx,sy=center_y+dy;
        double receiver=double(z)+.0002*(sx+.5-x)-.0003*(sy+.5-y);
        lit+=double(field.load(sx,sy))<=receiver+std::max(field.grid.pitch(0),field.grid.pitch(1))*.35;
    }
    return lit;
}
int main(){
    std::mt19937 random(291103);std::uniform_real_distribution<float> positions(-1024,1024),spans(.01f,160.f);
    for(unsigned sample=0;sample<20000;++sample){
        float bounds[4]={positions(random),positions(random),0,0};
        bounds[2]=bounds[0]+spans(random);bounds[3]=bounds[1]+spans(random);
        Grid grid;require(grid.configure(bounds,light));require(grid.pages()<=25);
        for(unsigned axis=0;axis<2;++axis){
            require(grid.quality_span[axis]<=legacy_span(bounds[axis],bounds[axis+2]));
            // Independently enumerate the extreme normal-displaced PCF cells.
            // The float density cap may trim unused padding below four units.
            double projected_normal=0;
            for(unsigned component=0;component<3;++component)
                projected_normal+=double(light[axis*4+component])*light[axis*4+component];
            projected_normal=std::sqrt(projected_normal);
            double displacement=projected_normal*std::max(6./1024.,1.5*double(std::max(grid.pitch(0),grid.pitch(1))));
            int first_cell=int(std::floor((double(bounds[axis])-displacement)*grid.inverse_pitch(axis)))-1;
            int last_cell=int(std::floor((double(bounds[axis+2])+displacement)*grid.inverse_pitch(axis)))+1;
            int first_page=int(std::floor(double(first_cell)/1024));
            int last_page=int(std::floor(double(last_cell)/1024));
            require(first_page>=grid.low[axis]);
            require(last_page<grid.low[axis]+int(grid.count[axis]));
        }
        for(unsigned slot=0;slot<grid.pages();++slot){auto page=grid.page(slot);require(grid.slot(page[0],page[1])==int(slot));}
    }
    std::cout<<"quality_density_and_25_page_bound=pass cases=20000\n";
    float rounding_edge[4]={0,0,24.f+std::ldexp(1.f,-19),24.f+std::ldexp(1.f,-19)};Grid edge_grid;
    require(edge_grid.configure(rounding_edge,light));
    require(edge_grid.quality_span[0]==32.f && edge_grid.quality_span[1]==32.f);
    require(edge_grid.pages()<=25);
    std::cout<<"legacy_float_boundary_density_cap=pass\n";
    float original[4]={-17.5f,-7,18.5f,15};Grid resident;require(resident.configure(original,light));
    require(resident.count[0]==5 && resident.count[1]==5);
    float smaller[4]={-16.5f,-6,17.5f,14};Grid smaller_grid;require(smaller_grid.configure(smaller,light));
    require(!resident.same_sampling(smaller_grid));require(!resident.covers(smaller_grid));
    auto old_coverage=resident.coverage();require(smaller[0]>old_coverage[0]+1 && smaller[2]<old_coverage[0]+old_coverage[2]-1);
    std::cout<<"containing_coverage_cannot_authorize_changed_current_quality=pass\n";
    // Configure the identical current view after several different prior grids.
    // History must affect neither sampling identity nor requested coverage.
    for(unsigned history=0;history<64;++history){
        float previous[4]={float(history)-32,-20,float(history)+42,17};
        Grid carried;require(carried.configure(previous,light));
        require(carried.configure(smaller,light));
        require(carried.quality_span==smaller_grid.quality_span);
        require(carried.low==smaller_grid.low && carried.count==smaller_grid.count);
        for(unsigned slot=0;slot<carried.pages();++slot)
            require(carried.page_box(slot)==smaller_grid.page_box(slot));
    }
    std::cout<<"current_quality_and_coverage_are_history_independent=pass histories=64\n";
    Field warm(resident);unsigned retained=0,rebuilds=0,queries=0;
    for(unsigned sweep=0;sweep<6;++sweep)for(int tick=-24;tick<=24;++tick){
        float du=float(tick)*.125f,dv=float(tick%7)*.125f;
        float moved[4]={original[0]+du,original[1]+dv,original[2]+du,original[3]+dv};
        Grid cold_grid;require(cold_grid.configure(moved,light));
        require(cold_grid.same_sampling(resident));
        if(!warm.grid.covers(cold_grid)){warm=Field(cold_grid);++rebuilds;}else ++retained;
        Field cold(cold_grid);
        for(int row=0;row<19;++row)for(int column=0;column<29;++column){
            double u=double(moved[0])+(column+.37)*(double(moved[2])-moved[0])/29;
            double v=double(moved[1])+(row+.63)*(double(moved[3])-moved[1])/19;
            float z=float((column%5)-2)*.003f;
            auto expected=independent_visibility(u,v,z,cold_grid);
            require(stored_visibility(u,v,z,cold)==expected);
            require(stored_visibility(u,v,z,warm)==expected);++queries;
        }
    }
    require(retained>0 && rebuilds>0);
    std::cout<<"warm_cold_global_pcf_continuity=pass samples="<<queries<<" reused="<<retained<<" rebuilds="<<rebuilds<<"\n";
    // Page raster settings are independent of array layer and neighboring box.
    float translated[4]={original[0]+12,original[1],original[2]+12,original[3]};Grid next;
    require(next.configure(translated,light));require(next.same_sampling(resident));
    unsigned overlap=0,different_slots=0;
    for(unsigned slot=0;slot<resident.pages();++slot){auto p=resident.page(slot);auto other=next.slot(p[0],p[1]);
        if(other<0)continue;++overlap;different_slots+=int(slot)!=other;
        auto a=resident.page_box(slot),b=next.page_box(unsigned(other));require(a==b);
        std::array<float,14> settings_a{},settings_b{};
        for(unsigned axis=0;axis<2;++axis)for(unsigned component=0;component<3;++component){
            settings_a[axis*4+component]=light[axis*4+component]*6/a[axis+2];
            settings_b[axis*4+component]=light[axis*4+component]*6/b[axis+2];
        }
        settings_a[12]=settings_b[12]=float(p[0]);settings_a[13]=settings_b[13]=float(p[1]);
        require(std::memcmp(settings_a.data(),settings_b.data(),sizeof(settings_a))==0);
    }
    require(overlap>0 && different_slots>0);
    std::cout<<"canonical_caster_projection_slot_independence=pass shared_pages="<<overlap<<"\n";
    // Shader floor division must work on the negative side of every page seam.
    for(int page=-4;page<=4;++page)for(int delta=-2;delta<=2;++delta){
        int cell=page*1024+delta,p=int(std::floor(double(cell)/1024));int local=cell-p*1024;
        require(local>=0 && local<1024 && p*1024+local==cell);
    }
    std::cout<<"negative_coordinate_and_cross_page_pcf=pass\n";
    // A chart change is part of the caller's sampling key, never a cache hint.
    auto fold=[](double x,double y,double cx,double cy,double w,double h){
        if(w>0){double s=std::floor((x+y-cx+w*.5)/w)*w*.5;x-=s;y-=s;}
        if(h>0){double s=std::floor((x-y-cy+h*.5)/h)*h*.5;x-=s;y+=s;}
        return std::array<double,2>{x,y};
    };
    for(int n=-3;n<=3;++n){auto reference=fold(9,7,16,4,32,24);
        auto x=fold(9+n*16,7+n*16,16,4,32,24);require(x==reference);
        auto y=fold(9+n*12,7-n*12,16,4,32,24);require(y==reference);
    }
    std::cout<<"periodic_occurrence_chart_same_key_continuity=pass\n";
    float invalid[4]={0,0,std::numeric_limits<float>::infinity(),1};Grid rejected;require(!rejected.configure(invalid,light));
    float too_large[4]={0,0,100000,1};require(!rejected.configure(too_large,light));
    float far[4]={100000,0,100001,1};require(!rejected.configure(far,light));
    float tiny[4]={0,0,.5f,.5f};auto exaggerated=light;exaggerated[0]=10000;
    require(!rejected.configure(tiny,exaggerated));
    require(Grid::texture_bytes==100u*1024u*1024u);
    require(!rejected.covers(resident) && !resident.same_sampling(rejected));
    std::cout<<"unsupported_extent_and_guard_fail_closed=pass\n";
    using Budget=c3x_renderer::render_core::FrameWorkingSet;
    static_assert(Grid::texture_bytes==25u*1024u*1024u*4u,"bounded R32 page storage");
    // The old field remains owned when its SRV is temporarily replaced. A
    // current attachment charge must include both fields before cache admission.
    constexpr std::size_t old_field=32u*1024u*1024u*4u,constant_buffer=80;
    constexpr std::size_t shadow_charge=Grid::texture_bytes+old_field+constant_buffer;
    auto attachments=Budget::cache_limit-400u*Budget::mib;
    auto before=Budget::caches(attachments,false,false);
    auto charged=Budget::caches(attachments+shadow_charge,false,false);
    require(before.regions+before.units==400u*Budget::mib);
    require(charged.regions+charged.units==400u*Budget::mib-shadow_charge);
    require(charged.regions+charged.units+attachments+shadow_charge<=Budget::cache_limit);
    std::cout<<"fixed_field_and_pinned_production_field_budget=pass charge_bytes="<<shadow_charge<<"\n";
    std::cout<<"SHADOW_GRID_HOST_PASS checks="<<checks<<" fixed_texture_bytes="<<Grid::texture_bytes<<"\n";
}
