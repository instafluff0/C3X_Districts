// Mountain shape contract: neighbouring tiles agree on shared edges and the
// mesh halo, ranges join across edges and corners, and the result does not
// depend on piece order. A synthetic ridged stamp stands in for pack fields.
#include "mountain_shape.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <map>
using namespace c3x_renderer::fidelity;

static NaturalData synthetic(){
    NaturalData natural;natural.fields.resize(2);
    for(auto& f:natural.fields){f.width=f.height=256;f.pixels.resize(256*256);}
    for(unsigned y=0;y<256;y++)for(unsigned x=0;x<256;x++){
        float u=(x+.5f)/256-.5f,v=(y+.5f)/256-.5f,r=std::sqrt(u*u+v*v),a=std::atan2(v,u);
        // Five gullied spurs around a high summit, like an authored stamp.
        float ridge=1-.35f*(1-std::cos(5*a+r*9))*.5f*std::min(1.f,r/.1f);
        natural.fields[0].pixels[y*256+x]=std::uint8_t(255*std::clamp((1-r/.34f)*ridge,0.f,1.f));
        natural.fields[1].pixels[y*256+x]=std::uint8_t(255*std::clamp((.38f-r)/.08f,0.f,1.f));
    }
    for(unsigned i=0;i<5;i++){natural.macro[i][0]=0;natural.macro[i][1]=1;natural.macro_axis[i]=.3f*float(i);}
    // A crater cone with three radial lava channels.
    for(auto* f:{&natural.volcano_height,&natural.volcano_blend,&natural.volcano_channel}){f->width=f->height=256;f->pixels.resize(256*256);}
    for(unsigned y=0;y<256;y++)for(unsigned x=0;x<256;x++){
        float u=(x+.5f)/256-.5f,v=(y+.5f)/256-.5f,r=std::sqrt(u*u+v*v),a=std::atan2(v,u);
        float cone=std::clamp(1-r/.30f,0.f,1.f)-.25f*std::clamp(1-r/.04f,0.f,1.f);
        natural.volcano_height.pixels[y*256+x]=std::uint8_t(255*cone);
        natural.volcano_blend.pixels[y*256+x]=std::uint8_t(255*std::clamp((.34f-r)/.06f,0.f,1.f));
        natural.volcano_channel.pixels[y*256+x]=r>.05f && r<.26f && std::abs(std::sin(1.5f*a))<.06f?255:0;
    }
    natural.volcano_channel.maximum=1;natural.volcano_ready=true;natural.measure_volcano_trend();
    return natural;
}

struct World {
    struct Cell {int real=2;unsigned flags=0;};
    std::map<std::pair<int,int>,Cell> cells; // terrain, MountainFlag bits
    mutable unsigned reads=0;
    Tile tile(int c,int r)const{++reads;auto it=cells.find({c,r});
        return Tile{c+r,c-r,c,r,it!=cells.end()?it->second.real:2};}
    unsigned snow(int c,int r)const{auto it=cells.find({c,r});return it!=cells.end()?it->second.flags:0u;}
    MountainShape shape(NaturalData const& natural,int c,int r)const{
        return MountainShape(natural,c,r,[&](int a,int b){return tile(a,b);},[&](int a,int b){return snow(a,b);});
    }
};

int main(){
    auto natural=synthetic();
    constexpr float step=1/64.f;
    // 1. Shared edges and the one-step mesh halo agree exactly between tiles.
    std::srand(7);
    std::size_t compared=0,raised=0;
    for(unsigned trial=0;trial<6;trial++){
        World world;
        for(int r=0;r<12;r++)for(int c=0;c<12;c++)
            world.cells[{c,r}]={std::rand()%100<(trial<3?40:70)?(std::rand()%5==0?10:6):2,
                unsigned(std::rand()%8)};
        for(int r=2;r<10;r++)for(int c=2;c<10;c++){
            auto owner=world.shape(natural,c,r);
            for(int dr=-1;dr<=1;dr++)for(int dc=-1;dc<=1;dc++){
                if(!dr && !dc)continue;
                auto other=world.shape(natural,c+dc,r+dr);
                // Every halo point of the owner that also lies in the other's halo.
                for(int j=-1;j<=65;j++)for(int i=-1;i<=65;i++){
                    if(i>0 && i<64 && j>0 && j<64)continue;
                    float x=float(c)+float(i)*step,y=float(r)+1-float(j)*step;
                    if(x<float(c+dc)-step || x>float(c+dc+1)+step || y<float(r+dr)-step || y>float(r+dr+1)+step)continue;
                    auto a=owner.sample(natural,x,y),b=other.sample(natural,x,y);
                    if(a.displacement!=b.displacement || a.snow!=b.snow){
                        std::printf("EDGE_MISMATCH tile %d,%d vs %d,%d at %.4f,%.4f: %.5f %.5f\n",c,r,c+dc,r+dr,x,y,a.displacement,b.displacement);
                        return 3;
                    }
                    ++compared;raised+=a.displacement>0;
                }
            }
        }
    }
    assert(raised>compared/4);
    // 2. A tile with no mountain in its 3x3 window reads only that window.
    World sparse;sparse.cells[{0,0}]={6,0u};
    sparse.reads=0;auto far=sparse.shape(natural,2,0);assert(!far.count && sparse.reads==9);
    sparse.reads=0;auto near=sparse.shape(natural,1,0);assert(near.count==1 && sparse.reads<=25);
    // 3. Corner neighbours join through their saddle; mains alone leave a gap.
    World pair;pair.cells[{0,0}]={6,0u};pair.cells[{1,1}]={6,0u};
    auto joined=pair.shape(natural,0,0);
    unsigned mains=0;for(unsigned i=0;i<joined.count;i++)mains+=std::fmod(joined.pieces[i].center_x,1.f)==.5f && std::fmod(joined.pieces[i].center_y,1.f)==.5f;
    assert(mains==2 && joined.count==3);
    float corner=joined.sample(natural,1.f,1.f).displacement;
    auto apart=joined;apart.count=mains;
    float gap=apart.sample(natural,1.f,1.f).displacement;
    if(corner<.25f*MountainShape::height || corner<=gap){std::printf("CORNER_GAP %.2f %.2f\n",corner,gap);return 4;}
    // 4. Piece order does not change the surface.
    auto reversed=joined;std::reverse(reversed.pieces.begin(),reversed.pieces.begin()+reversed.count);
    for(float y=-.2f;y<2.2f;y+=.05f)for(float x=-.2f;x<2.2f;x+=.05f)
        assert(joined.sample(natural,x,y).displacement==reversed.sample(natural,x,y).displacement);
    // 5. A straight range turns its stamps along the range.
    World line;for(int c=0;c<5;c++)line.cells[{c,0}]={6,0u};
    auto middle=line.shape(natural,2,0);
    assert(std::abs(std::abs(middle.pieces[1].range_cos)-1)<1e-5f);
    World diagonal;for(int i=0;i<5;i++)diagonal.cells[{i,i}]={6,1u};
    auto slope=diagonal.shape(natural,2,2);
    assert(std::abs(std::abs(slope.pieces[1].range_cos)-.70710678f)<1e-4f);
    assert(slope.sample(natural,2.5f,2.5f).snow==1.f);
    // 6. Nothing reaches beyond its bound: an isolated stamp ends within 1.06.
    World single;single.cells[{0,0}]={6,0u};
    auto lone=single.shape(natural,0,0);
    for(float a=0;a<6.3f;a+=.05f)assert(lone.sample(natural,.5f+1.07f*std::cos(a),.5f+1.07f*std::sin(a)).displacement==0);
    // 7. A volcano: its stamp owns the material at its centre, carries its
    // Civ III activity, marks its authored channels, stays within 1.04 tiles
    // and joins a neighbouring mountain through a saddle.
    World volcanic;volcanic.cells[{0,0}]={10,volcano_active|volcano_erupting};
    auto cone=volcanic.shape(natural,0,0);
    assert(cone.count==1 && cone.pieces[0].channel_field && cone.pieces[0].activity==2);
    auto centre=cone.sample(natural,.5f,.5f);
    assert(centre.volcano>.99f && centre.activity==2 && centre.displacement>.5f*MountainShape::volcano_height);
    float channelled=0;
    for(float a=0;a<6.3f;a+=.01f)channelled=std::max(channelled,cone.sample(natural,.5f+.35f*std::cos(a),.5f+.35f*std::sin(a)).channel);
    assert(channelled>.5f);
    for(float a=0;a<6.3f;a+=.05f)assert(cone.sample(natural,.5f+1.05f*std::cos(a),.5f+1.05f*std::sin(a)).displacement==0);
    // It is a cone, not a flat-topped mound: a rim about a mountain's height
    // around a shallower crater, concave flanks easing out to the foot.
    auto ring=[&](float radius){float sum=0;int n=0;
        for(float a=0;a<6.28f;a+=.02f,n++)sum+=cone.sample(natural,.5f+radius*std::cos(a),.5f+radius*std::sin(a)).displacement;
        return sum/float(n);};
    float span=MountainShape::volcano_span,rim=ring(MountainShape::volcano_rim*span);
    assert(rim>.9f*MountainShape::volcano_height && rim<1.15f*MountainShape::volcano_height);
    assert(centre.displacement<rim-.1f*MountainShape::volcano_height);
    float half=ring(.5f*(MountainShape::volcano_rim+MountainShape::volcano_foot)*span);
    assert(half>.25f*rim && half<.5f*rim);
    assert(ring(.95f*MountainShape::volcano_foot*span)<.03f*rim);
    // The user's bound: a volcano rises little beyond its tile on screen. At
    // .47 px per unit (1x), 92 units put the summit 11 px above the diamond's
    // top corner; it also stays within its tile's neighbourhood.
    float summit=0;
    for(float y=-1;y<=2;y+=.01f)for(float x=-1;x<=2;x+=.01f)summit=std::max(summit,cone.sample(natural,x,y).displacement);
    assert(summit<92.f && MountainShape::volcano_span*MountainShape::volcano_foot<.7f);
    World flank;flank.cells[{0,0}]={10,0u};flank.cells[{1,0}]={6,0u};
    auto joined_volcano=flank.shape(natural,0,0);
    assert(joined_volcano.count==3 && joined_volcano.sample(natural,1.f,.5f).displacement>.15f*MountainShape::height);
    // 8. A wrapped occurrence (same raw coordinates) builds the same volcano.
    struct Wrapped {
        Tile tile(int c,int r)const{int x=((c+r)%64+64)%64;return Tile{x,c-r,c,r,(x==4 && c-r==0)?10:2};}
    } wrapped;
    auto near_cone=MountainShape(natural,2,2,[&](int c,int r){return wrapped.tile(c,r);},[](int,int){return 0u;});
    auto far_cone=MountainShape(natural,34,34,[&](int c,int r){return wrapped.tile(c,r);},[](int,int){return 0u;});
    for(float dy=-.6f;dy<.6f;dy+=.1f)for(float dx=-.6f;dx<.6f;dx+=.1f){
        auto a=near_cone.sample(natural,2.5f+dx,2.5f+dy),b=far_cone.sample(natural,34.5f+dx,34.5f+dy);
        assert(std::abs(a.displacement-b.displacement)<1e-3f && std::abs(a.volcano_u-b.volcano_u)<1e-4f && std::abs(a.volcano-b.volcano)<1e-4f);
    }
    std::printf("PASS mountain shape: %zu shared halo samples agree, corner join %.1f over %.1f\n",compared,corner,gap);
}
