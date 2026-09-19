#include "Renderer/native/render_core/scene_provenance.h"
#include <cassert>
struct Source {int epoch;};
struct Rect {int left,top,right,bottom;};
int test_scene_provenance(){
    using Proof=c3x_renderer::render_core::SceneProvenance<Source,Rect>;
    auto old=std::make_shared<Source>(Source{1}),next=std::make_shared<Source>(Source{2});
    Proof map;map.parts.push_back({{0,0,100,100},old,20,30});
    Proof screen;screen.copy(map,{10,15,60,65},-10,-15);
    auto p=screen.select({11,16,59,64});assert(p.source==old&&p.x==10&&p.y==15);
    assert(!screen.select({0,0,20,20}).source);
    // A popup invalidates only its own pixels; restoring its saved before-image
    // restores the exact old map proof, even after another map is published.
    Proof save;save.copy(screen,{0,0,20,20},20,25);screen.erase({20,25,40,45});
    assert(!screen.select({20,25,40,45}).source);
    assert(screen.select({10,15,20,25}).source==old);
    map.parts={{{0,0,100,100},next,40,50}};
    screen.copy(save,{20,25,40,45},-20,-25);
    p=screen.select({20,25,40,45});assert(p.source==old&&p.x==10&&p.y==15);
    // A restored rectangle and its surrounding fragments form one proof.
    p=screen.select({10,15,60,65});assert(p.source==old&&p.x==10&&p.y==15);
    Proof mixed; mixed.parts={{{0,0,10,10},old,0,0},{{10,0,20,10},next,0,0}};
    assert(!mixed.select({0,0,20,10}).source);
    mixed.parts[1].source=old;mixed.parts[1].x=1;assert(!mixed.select({0,0,20,10}).source);
    mixed.parts[1].x=0;assert(mixed.select({0,0,20,10}).source==old);
    // Cross-position self-copy uses the before-image and shifts its map anchor.
    screen.copy(screen,{60,15,70,25},-50,0);p=screen.select({60,15,70,25});
    assert(p.source==old&&p.x==-40&&p.y==15);
    screen.erase({60,15,70,25});assert(!screen.select({60,15,70,25}).source);
    // Metadata is bounded; fragmentation can lose eligibility, never invent it.
    Proof fragmented;fragmented.parts.push_back({{0,0,1024,1024},old,0,0});
    for(int y=0;y<64;++y)for(int x=0;x<64;++x)fragmented.erase({x*16,y*16,x*16+1,y*16+1});
    assert(fragmented.parts.size()<=256&&!fragmented.select({0,0,1024,1024}).source);
    return 0;
}
