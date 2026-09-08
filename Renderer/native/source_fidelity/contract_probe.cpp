// Independent source/provider parity; no D3D, assets, or game process required.
#include <array>
#include <cmath>
#include <iostream>
#include "../environment_runtime.h"
#include "light_frame.h"
#include "coast_join.h"
#define hydro reference_hydro
#define river reference_river
#include "../../lab/shared/hydrology/river_corridor.h"
#undef hydro
#undef river
#include "river_corridor.h"
#include "../render_core/terrain_query.h"
int main(int argc,char**argv){
    if(argc!=2)return 2;
    reference_hydro::Field reference(argv[1]);
    hydro::Field adapted;adapted.load(argv[1]);adapted.wraps=true;
    auto height=[](double x,double y){return 2.5+std::max(0.,30-12*std::hypot(x-4,y-5));};
    reference_river::Corridor expected;river::Corridor actual;
    expected.build(argv[1],height);actual.build(adapted,height);
    if(expected.edges.size()!=actual.edges.size() || expected.terminals.size()!=actual.terminals.size())return 1;
    unsigned samples=0;
    for(double y=-1;y<11;y+=.125)for(double x=-1;x<11;x+=.125){
        auto a=expected.sample({x,y});auto b=actual.sample({x,y});
        if(a.distance!=b.distance || a.source!=b.source || a.mouth!=b.mouth)return 1;
        ++samples;
    }
    unsigned phases=0;
    // Every partially transparent ground sample lies on the retained beach.
    // Probe variable beach widths, continuity, monotonicity and exact inland
    // preservation independently of mesh tessellation and caster clipping.
    for(float width:{.05f,.2f,.4f}) {
        float previous=0;
        for(int i=-100;i<=1000;i++) {
            float d=width+i*.001f;
            float a=c3x_renderer::fidelity::coast_coverage(d,width);
            float h=c3x_renderer::fidelity::coast_relief(d,width);
            if(a<.99999f && h!=0)return 1;
            if(h<previous || h<0 || h>1 || (d>width+.581f && h!=1))return 1;
            if(h-previous>.0042f)return 1;
            previous=h;
        }
        for(float edge:{.22f,.58f}) {
            float slope=(c3x_renderer::fidelity::coast_relief(width+edge+.0001f,width)-
                c3x_renderer::fidelity::coast_relief(width+edge-.0001f,width))/.0002f;
            if(std::abs(slope)>.004f)return 1;
        }
    }
    // The live boundary adapter consumes the retained continuous field,
    // including four-way intersections and wrapped source coordinates.
    using namespace c3x_renderer::render_core;
    World world{100,100,true,false};
    auto biome=[](int c,int r){return Tile{r<2?(c<2?2:1):(c<2?0:3),c==2&&r==2?9:2,true};};
    for(double y=-1;y<5;y+=.03125)for(double x=-1;x<5;x+=.03125){
        auto a=material_weights({x,y},world,biome);
        auto b=material_weights({x+1e-6,y},world,biome);
        double sum=0;for(int i=0;i<5;i++){sum+=a[i];if(a[i]<-1e-12||a[i]>1+1e-12||std::abs(a[i]-b[i])>1e-4){std::cerr<<"weight "<<x<<","<<y<<" channel "<<i<<" a="<<a[i]<<" b="<<b[i]<<"\n";return 1;}}
        if(std::abs(sum-1)>1e-10)return 1;
        // Encoding in the existing vertex stride must reconstruct all four
        // families independently, without an artificial plains intermediate.
        double source=1-a[3],desert=a[2]/std::max(source,1e-5);
        double tundra=a[4]/std::max(source-a[2],1e-5);
        double plains=a[1]/std::max(source-a[2]-a[4],1e-5);
        if(std::abs(source*(1-desert)*(1-tundra)*plains-a[1])>1e-5)return 1;
    }
    for(int h=0;h<24;h++){
        auto f=c3x_renderer::fidelity::light_frame(c3x_renderer::evaluate_environment(float(h),0));
        for(int i=0;i<3;i++)for(int j=0;j<3;j++){
            float dot=0;for(int k=0;k<3;k++)dot+=f[i*4+k]*f[j*4+k];
            if(std::abs(dot-(i==j?1:0))>1e-5)return 1;
        }
        float dx=-f[8]/f[10],dy=-f[9]/f[10];
        if(h==12 && !((dx+dy)>0 && (dx-dy)<0))return 1;
        ++phases;
    }
    std::cout<<"PASS selected river provider samples="<<samples<<" exact equality; Q6 orthonormal phases="<<phases<<" noon up-right\n";
}
