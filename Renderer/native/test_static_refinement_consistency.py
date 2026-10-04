"""Execute production refinement across a world edit between raster bands."""
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.test_fresh_shared_submission import method


class StaticRefinementConsistencyTests(unittest.TestCase):
    def test_failed_local_repair_preserves_front_and_success_commits_only_dirty_pixels(self):
        source = (ROOT / 'Renderer/sandbox/fresh_pipeline.h').read_text()
        repair = method(source, '    bool repair_front(')
        start = repair.index('        long long area=0;')
        end = repair.index('\n#else', start)
        run_cpp(r'''
#include "Renderer/sandbox/static_raster_state.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdio>
#include <vector>
using namespace c3x_renderer::render_core;
struct D3D11_RECT {int left,top,right,bottom;};
struct Target {
    unsigned width=64,height=48;
    std::vector<int> colors=std::vector<int>(width*height,11),depths=std::vector<int>(width*height,97);
    std::vector<int>* samples=&colors;std::vector<int>* depth_samples=&depths;
    void reset(){};std::size_t bytes()const{return 0;}
};
using StaticState=StaticRasterState<Target>;using StaticRect=StaticState::Rect;
struct ViewportShaderSettings {};
struct Harness {
    StaticRasterStates<Target> static_rasters;
    struct Inputs {bool complete=true;void clear(){complete=true;}};
    std::array<Inputs,4> raster_inputs;
    struct Context {void OMSetRenderTargets(int,void*,void*){}} context;
    struct Renderer {Context* context;} renderer{&context};
    struct Work {void draw(int){}} work;
    struct Restore {
        bool draw(Context*,Target& target,std::vector<int>* color,std::vector<int>* depth,
                int dx,int dy,std::vector<D3D11_RECT> const&,void*,unsigned,unsigned,
                bool,bool clear,int,D3D11_RECT const* clip,int scale=2){
            assert(dx==0 && dy==0 && clip && (clear || (color&&depth&&scale==1)));
            for(int y=clip->top;y<clip->bottom;++y)for(int x=clip->left;x<clip->right;++x){
                auto at=y*target.width+x;
                target.colors[at]=clear?0:(*color)[at];target.depths[at]=clear?999:(*depth)[at];
            }
            return true;
        }
    } static_restore;
    unsigned region_width_px=64,region_height_px=48,scene_samples=1;
    unsigned writes=0,fail_at=0,dependency_rebuilds=0,partial_repairs=0;
    std::uint64_t partial_repair_pixels=0;bool refine_worked=false;
    bool ensure_linear_target(Target&,unsigned,unsigned,unsigned,bool){return true;}
    bool write_slot(unsigned,StaticState& slot,ViewportShaderSettings const&,StaticRect r,float,bool){
        ++writes;++slot.revision;
        // Simulate an early terrain layer succeeding and a later city/mesh
        // layer refusing the draw. Its partial pixels must never be displayed.
        for(int y=r.top;y<r.bottom;++y)for(int x=r.left;x<r.right;++x){
            slot.region.colors[y*64+x]=22;slot.region.depths[y*64+x]=49;
        }
        return writes!=fail_at;
    }
    bool raster_dependencies(Inputs&,ViewportShaderSettings const&,D3D11_RECT,bool){++dependency_rebuilds;return true;}
    bool repair(unsigned index,std::vector<StaticRect> const& dirty){
        auto& slot=static_rasters.states[index];auto& inputs=raster_inputs[index];
        auto const& c=slot.covered;ViewportShaderSettings view,screen;
        D3D11_RECT covered{c.left,c.top,c.right,c.bottom};
''' + repair[start:end] + r'''
    }
};
int main(){
    unsigned cases=0;
    for(unsigned index=0;index<4;++index)for(unsigned fail:{2u,1u,0u}){
        Harness h;auto& front=h.static_rasters.states[index];front.covered={0,0,64,48};
        front.valid=true;front.camera_x=91;front.camera_y=-77;front.projection=2.f;front.depth_translation=-4096;
        auto& back=h.static_rasters.states[index^1u];back.valid=back.refining=true;back.covered=front.covered;
        auto original_color=front.region.colors,original_depth=front.region.depths;auto revision=front.revision;
        h.fail_at=fail;std::vector<StaticRect> dirty{{4,4,12,12},{22,26,36,35}};
        bool ok=h.repair(index,dirty);
        assert(ok==(fail==0));
        if(fail){
            assert(front.region.colors==original_color&&front.region.depths==original_depth);
            assert(front.revision==revision&&h.dependency_rebuilds==0&&h.partial_repairs==0);
        }else{
            for(int y=0;y<48;++y)for(int x=0;x<64;++x){
                bool changed=false;for(auto r:dirty)changed|=x>=r.left&&x<r.right&&y>=r.top&&y<r.bottom;
                assert(front.region.colors[y*64+x]==(changed?22:11));
                assert(front.region.depths[y*64+x]==(changed?49:97));
            }
            assert(front.revision>revision&&h.dependency_rebuilds==1&&h.partial_repairs==1);
            assert(h.partial_repair_pixels==190);
        }
        assert(!back.valid&&!back.refining&&back.covered.empty());
        assert(back.camera_x==front.camera_x&&back.camera_y==front.camera_y&&back.projection==front.projection&&back.depth_translation==front.depth_translation);
        ++cases;
    }
    std::printf("PASS local repair transaction: cases=%u all_slots=1 late_draw_failure=1 exact_color_depth_damage=1\n",cases);
}
''')

    def test_changed_covered_pixels_restart_before_promotion(self):
        source = (ROOT / 'Renderer/sandbox/fresh_pipeline.h').read_text()
        refine = method(source, '        auto refine=[&](float projection,double& spend)->int{') + ';'
        run_cpp(r'''
#include "Renderer/sandbox/static_raster_state.h"
#include "Renderer/native/render_core/raster_contributors.h"
#include <cassert>
#include <cstdio>
#include <vector>
using namespace c3x_renderer::render_core;
struct Target {std::vector<int> pixels;void reset(){};std::size_t bytes()const{return 0;}};
using StaticState=StaticRasterState<Target>;
using StaticRect=StaticState::Rect;
using States=StaticRasterStates<Target>;
struct D3D11_RECT {int left,top,right,bottom;};
struct ViewportShaderSettings {float projection=0;};
struct Proof {unsigned row=0;int version=0;};
using Inputs=RasterContributors<Proof>;
struct Shift {bool reusable=true;};
struct Harness {
    States static_rasters;std::array<Inputs,4> raster_inputs;
    std::array<int,4> world{{1,2,3,4}};
    std::array<bool,4> present{{true,true,true,true}};
    std::array<std::uint64_t,4> visibility{{1,1,1,1}};
    unsigned refine_restarts=0,refine_slices=0,cache_full_draws=0,refine_promotions=0,resets=0,checks=0;
    bool refine_worked=false;float projection_zoom=1;
    struct ZoomScope {
        Harness& owner;float prior;
        ZoomScope(Harness& h,float p):owner(h),prior(h.projection_zoom){h.projection_zoom=p;}
        ~ZoomScope(){owner.projection_zoom=prior;}
    };
    ViewportShaderSettings slot_settings(StaticState const& slot,ViewportShaderSettings const&)const{return {slot.projection};}
    bool raster_dependencies(Inputs& inputs,ViewportShaderSettings view,D3D11_RECT rect,bool append){
        ++checks;assert(!append && view.projection==projection_zoom);
        if(!inputs.valid([&](Proof const& p){return world[p.row]==p.version;},
                [&](std::uint64_t row){return visibility[row];}))return false;
        inputs.begin_membership();bool valid=true;
        for(int row=rect.top;row<rect.bottom;++row)if(present[row]){
            Inputs::Key key{};key[0]=row+1;
            valid=inputs.visit_membership(key)&&valid;
        }
        return valid&&inputs.exact_membership();
    }
    bool reset_slot(unsigned index,float projection,ViewportShaderSettings const&){
        ++resets;auto& slot=static_rasters.states[index];
        slot.covered={};slot.valid=slot.stale=false;slot.refining=true;slot.projection=projection;
        slot.region.pixels.assign(4,0);raster_inputs[index].clear();return true;
    }
    bool extend_coverage(unsigned index,StaticState& slot,ViewportShaderSettings const&,
            StaticRect needed,double& budget,int,float,bool){
        auto& inputs=raster_inputs[index];int row=slot.covered.bottom;
        for(;row<needed.bottom&&budget!=0;++row){
            slot.region.pixels[row]=present[row]?world[row]:0;
            if(present[row]){Inputs::Key key{};key[0]=row+1;
                inputs.add(key,std::make_shared<Proof>(Proof{unsigned(row),world[row]}),row,visibility[row]);}
            if(budget>0)--budget;
        }
        slot.covered={0,0,1,row};return true;
    }
    int advance(unsigned lane,float projection,double spend){
        ViewportShaderSettings settings{};StaticRasterKey key{};
        float zoom=projection;unsigned front_index=static_rasters.front_slot[lane];
        Shift shift{};bool shifted=false;
        auto visible_rect=[](StaticState const&,Shift&){return StaticRect{0,0,1,4};};
''' + refine + r'''
        return refine(projection,spend);
    }
};
int main(){
    unsigned scenarios=0;
    for(unsigned lane:{0u,1u})for(unsigned change=0;change<5;++change){
        Harness h;float zoom=lane?2.f:1.f;
        // Keep a coherent displayed image while its replacement is incomplete.
        auto& old=h.static_rasters.front(lane);old.valid=old.stale=true;
        old.region.pixels={91,92,93,94};
        if(change==2)h.present[0]=false; // reveal enters an already drawn band
        assert(h.advance(lane,zoom,2)==0 && h.resets==1);
        assert(h.static_rasters.front(lane).region.pixels==std::vector<int>({91,92,93,94}));
        if(change==0)h.world[0]=10;             // geometry replacement
        if(change==1)h.present[0]=false;        // contributor removal
        if(change==2)h.present[0]=true;         // fog reveal / contributor addition
        if(change==3)++h.visibility[0];        // same mesh, different visibility
        if(change==4)h.world[3]=40;             // only a not-yet-drawn band changes
        auto result=h.advance(lane,zoom,2);
        if(change<4){
            assert(result==0 && h.resets==2 && h.refine_promotions==0);
            assert(h.static_rasters.front(lane).region.pixels==std::vector<int>({91,92,93,94}));
            assert(h.advance(lane,zoom,2)==1);
        }else assert(result==1 && h.resets==1); // no unnecessary restart
        std::vector<int> expected;
        for(unsigned row=0;row<4;++row)expected.push_back(h.present[row]?h.world[row]:0);
        assert(h.static_rasters.front(lane).region.pixels==expected);
        assert(h.refine_promotions==1 && h.checks>0 && h.projection_zoom==1.f);
        ++scenarios;
    }
    std::printf("PASS static refinement consistency: scenarios=%u lanes=2 stale_front_preserved=1 unaffected_bands_continue=1\n",scenarios);
}
''')


if __name__ == '__main__':
    unittest.main()
