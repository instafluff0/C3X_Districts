// Shared statement body: retain the native x86 calling context and float rounding.
// Included by the native tile compiler and the portable typed adapter in mesh.h.
    if(tile.real_terrain_type==5){
        Hill hill=composed_hill(owner);std::uint32_t state=hill.seed;
        constexpr unsigned cells[]={0,0,0,0,1,1,1,2,2,2};
        for(unsigned ordinal=0;ordinal<10;ordinal++){
            float keep=random01(state),angle=random01(state)*6.283185307f;
            float radius=std::sqrt(random01(state))*.22f,phase=random01(state)*6.283185307f,scale=.90f+.20f*random01(state);
            if(keep>hill.rockiness)continue;
            float cu=.5f+std::cos(phase)*radius,cv=.5f+std::sin(phase)*radius;
            auto point=[&](unsigned x,unsigned y){
                float du=(x/8.f-.5f)*.42f*scale,dv=(y/8.f-.5f)*.37f*scale;
                float u=cu+std::cos(angle)*du-std::sin(angle)*dv,v=cv+std::sin(angle)*du+std::cos(angle)*dv;
                float support=0;float h=height_natural(float(nc)+u,float(nr)+1-v,&support)+.45f;
                auto out=project_natural(float(nc)+u,float(nr)+1-v,h);
                out.u=(float(cells[ordinal])+.0012f+x/8.f*(1-2*.0012f))*.25f;
                out.v=(.0012f+y/8.f*(1-2*.0012f))*.25f;
                out.material_grass=std::max(0.f,(h-2.5f)/112);out.material_plains=2;out.material_desert=angle;
                auto shore=shore_sample_at(float(nc)+u,float(nr)+1-v);
                out.base_terrain=-10+coast_coverage(float(shore.distance),float(shore.beach_width));
                return out;
            };
            for(unsigned y=0;y<8;y++)for(unsigned x=0;x<8;x++){
                auto a=point(x,y),b=point(x+1,y),c=point(x+1,y+1),d=point(x,y+1);
                triangle(natural_vertices[1],a,b,c);triangle(natural_vertices[1],a,c,d);
            }
        }
    }
    if(tile.real_terrain_type==6){
        unsigned variant=mountain_seed(owner)%5u;
        auto const&hf=natural.fields[natural.macro[variant][0]];
        auto const&bf=natural.fields[natural.macro[variant][1]];
        std::vector<Vertex> grid(128*128);
        for(unsigned y=0;y<128;y++){
            if(cancelled())return false;
            for(unsigned x=0;x<128;x++){
                float u=x/127.f,v=y/127.f,h=hf.sample(u,v),du=1/127.f;
                auto out=project_natural(float(nc)+.5f+(u-.5f)*1.85f,float(nr)+.5f+(.5f-v)*1.55f,2.5f+h*165);
                float n[]={-(hf.sample(u+du,v)-hf.sample(u-du,v))*(165/112.f)/(2*du*1.85f),
                    (hf.sample(u,v+du)-hf.sample(u,v-du))*(165/112.f)/(2*du*1.55f),1};normalize3(n);
                out.normal_x=n[0];out.normal_y=n[1];out.normal_z=n[2];out.u=u;out.v=v;
                out.material_grass=h;out.material_plains=2;out.material_desert=bf.sample(u,v);out.base_terrain=42;
                grid[y*128+x]=out;
            }
        }
        for(unsigned y=0;y<127;y++)for(unsigned x=0;x<127;x++){
            auto&a=grid[y*128+x];auto&b=grid[y*128+x+1];auto&c=grid[(y+1)*128+x+1];auto&d=grid[(y+1)*128+x];
            if(std::max({a.material_desert,b.material_desert,c.material_desert,d.material_desert})<.015f)continue;
            triangle(natural_vertices[2],a,b,c);triangle(natural_vertices[2],a,c,d);
        }
    }
