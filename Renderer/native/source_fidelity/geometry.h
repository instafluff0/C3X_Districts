// Included inside the existing tile compiler: observes the same authoritative
// dependencies and emits into its bounded immutable world-geometry cache.
if(fidelity_profile) {
    using namespace c3x_renderer::fidelity;
    int nc=(tile.tile_x+tile.tile_y)/2,nr=(tile.tile_x-tile.tile_y)/2;
    auto lookup_natural=[&](int c,int r){
        auto value=world_lookup(c,r);
        return Tile{canonical_component(c+r,frame.world_width_tiles,frame.world_wrap_x),
            canonical_component(c-r,frame.world_height_tiles,frame.world_wrap_y),c,r,value.real};
    };
    Tile owner=lookup_natural(nc,nr);
    auto height_natural=[&](float x,float y,float*support=nullptr){
        auto shore=shore_sample_at(x,y);
        float h=std::max(natural.height(x,y,lookup_natural,support),2.5f+pickup_height_at(x,y));
        return 2.5f+(h-2.5f)*coast_relief(float(shore.distance),float(shore.beach_width));
    };
    auto project_natural=[&](float x,float y,float h){
        Vertex out={};float dx=x-float(nc),dy=y-float(nr);
        float base=(dx-dy+1)*half_h;
        out.x=(dx+dy)*half_w;
        out.y=base-(h-2.5f)*relief_projection_scale;
        out.z=base+(h-2.5f)*.0016f*float(frame.target_height);
        out.world_x=x;out.world_y=y;out.world_z=h/112;out.world_valid=1;
        out.normal_z=1;out.u=x;out.v=y;
        return out;
    };
    auto triangle=[](std::vector<Vertex>&out,Vertex const&a,Vertex const&b,Vertex const&c){out.push_back(a);out.push_back(b);out.push_back(c);};
    auto surface=[&](float u,float v,float lift=0.f){
        float x=float(nc)+u,y=float(nr)+1-v,support=0;
        float h=height_natural(x,y,&support);
        Vertex out=project_natural(x,y,h+lift);
        constexpr float e=.006f;
        float n[]={-(height_natural(x+e,y)-height_natural(x-e,y))/(2*e*128),
            -(height_natural(x,y+e)-height_natural(x,y-e))/(2*e*128),1};normalize3(n);
        out.normal_x=n[0];out.normal_y=n[1];out.normal_z=n[2];
        out.material_grass=std::max(0.f,(h-2.5f)/112);out.material_plains=1;out.material_desert=support;
        auto shore=shore_sample_at(x,y);
        float coverage=coast_coverage(float(shore.distance),float(shore.beach_width));
        auto weights=material_weights_for(x,y);
        // Preserve the native marsh interior underneath; fade the selected
        // four source families continuously across its surrounding cells.
        float source_weight=std::clamp(1-weights[3],0.f,1.f);
        out.base_terrain=-10+coverage*source_weight;
        for(auto&weight:weights)weight/=std::max(source_weight,.00001f);
        out.material_marsh=weights[4];
        out.authored_relief_height=weights[1];out.authored_relief_blend=weights[2];
        return out;
    };
    if(ground<11 || shore_sample_at(float(nc)+.5f,float(nr)+.5f).distance>-.8f){
        // Same 16x16 r13 grid, evaluate shared corners only once.
        std::array<Vertex,17*17> grid;
        for(unsigned y=0;y<=16;y++){if(cancelled())return false;for(unsigned x=0;x<=16;x++)grid[y*17+x]=surface(x/16.f,y/16.f);}
        for(unsigned y=0;y<16;y++)for(unsigned x=0;x<16;x++){
            auto&a=grid[y*17+x];auto&b=grid[y*17+x+1];auto&c=grid[(y+1)*17+x+1];auto&d=grid[(y+1)*17+x];
            if(std::max({a.base_terrain,b.base_terrain,c.base_terrain,d.base_terrain})<=-9.999f)continue;
            triangle(natural_vertices[0],a,b,c);triangle(natural_vertices[0],a,c,d);
        }
    }
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
    #include "../city_fidelity/geometry.h"
    if(tile.real_terrain_type==7){
        // Exact current production building meshes/placement, used only as
        // exclusions. City appearance and its geometry path remain unchanged.
        struct BuildingBounds{float x0,y0,x1,y1;};std::vector<BuildingBounds> buildings;
        constexpr char const*eras[]={"ancient","medieval","industrial","modern"};
        constexpr unsigned counts[]={4,7,11};constexpr float radii[]={.25f,.33f,.41f},scales[]={.92f,1,1.08f};
        for(int r=nr-2;r<=nr+2;r++)for(int c=nc-2;c<=nc+2;c++){
            auto it=tile_by_coordinate.find(observed_coordinate_key(c+r,c-r));if(it==tile_by_coordinate.end())continue;
            auto const&city=*it->second;if(city.city_id<0)continue;
            if(auto composition=selected_city(city,c,r)){
                for(auto const&i:composition->instances)buildings.push_back({
                    float(c)+.5f+i.offset[0]+i.bounds[0],float(r)+.5f-i.offset[1]-i.bounds[3],
                    float(c)+.5f+i.offset[0]+i.bounds[2],float(r)+.5f-i.offset[1]-i.bounds[1]});
                continue;
            }
            unsigned size=unsigned(std::clamp(city.city_size,0,2)),culture=unsigned(std::max(0,city.city_culture_group));
            auto*g=c3x_renderer::find_feature_group(city_bundle,eras[std::clamp(city.city_era,0,3)]);
            if(!g || g->placements.empty())continue;
            for(unsigned slot=0;slot<counts[size];slot++){
                auto const&p=g->placements[(culture+city.variant_seed+slot)%g->placements.size()];
                if(p.asset_index>=city_bundle.assets.size())return false;
                float angle=float(slot)*2.39996322973f+c3x_renderer::stable_random(city.variant_seed*53u+culture*19u)*.72f;
                float radius=slot==0?0:radii[size]*std::sqrt(float(slot)/float(counts[size]-1));
                float scale=p.scale*scales[size]*(slot==0 && (city.city_flags&C3X_RENDERER_CITY_CAPITAL)?1.30f:1);
                float cx=float(c)+.5f+std::cos(angle)*radius,cy=float(r)+.5f-std::sin(angle)*radius*.78f;
                float co=std::cos(angle+.55f),si=std::sin(angle+.55f);BuildingBounds b={1e9f,1e9f,-1e9f,-1e9f};
                for(auto const&v:city_bundle.assets[p.asset_index].vertices){
                    float x=cx+(v.position[0]*co-v.position[1]*si)*scale,y=cy-(v.position[0]*si+v.position[1]*co)*scale;
                    b.x0=std::min(b.x0,x);b.x1=std::max(b.x1,x);b.y0=std::min(b.y0,y);b.y1=std::max(b.y1,y);}
                buildings.push_back(b);
            }
        }
        std::uint32_t seed=std::uint32_t(owner.source_x*0x193u)^std::uint32_t(owner.source_y*0x217u);
        unsigned density=31+c3x_renderer::stable_hash(seed^0xa53du)%11u;
        for(unsigned i=0;i<density;i++){
            if(cancelled())return false;
            unsigned selected=c3x_renderer::stable_hash(seed+i*31u)%180;Recipe const*recipe=nullptr;
            for(auto const&r:natural.recipes){if(selected<r.count){recipe=&r;break;}selected-=r.count;}
            if(!recipe)return false;
            float ring=std::sqrt((float(i)+.5f)/float(density));
            float angle=2.39996323f*float(i)+c3x_renderer::stable_random(seed^0x71b3u)*6.283185307f;
            float u=.5f+std::cos(angle)*ring*.43f,v=.5f+std::sin(angle)*ring*.43f;
            float scale=recipe->scale*(1+recipe->variation*(c3x_renderer::stable_random(seed+i*71u+23u)*2-1))*.46f;
            float yaw=c3x_renderer::stable_random(seed+i*97u+47u)*6.283185307f;
            float co=std::cos(yaw),si=std::sin(yaw);auto const&body=natural.bodies[recipe->object];
            auto const&mat=natural.materials[body.material];
            // Clip flags are authoritative source metadata. Test the body hull,
            // not just its center, against captured building and water geometry.
            float x0=1e9f,y0=1e9f,x1=-1e9f,y1=-1e9f;
            for(auto const&p:body.vertices){float x=float(nc)+u+(p.position[0]*co-p.position[1]*si)*scale;
                float y=float(nr)+1-v-(p.position[0]*si+p.position[1]*co)*scale;
                x0=std::min(x0,x);x1=std::max(x1,x);y0=std::min(y0,y);y1=std::max(y1,y);}
            bool clipped=false;
            for(auto const&b:buildings)if(x0<b.x1 && x1>b.x0 && y0<b.y1 && y1>b.y0)clipped=true;
            // Conservative footprint bounds against the authoritative fields:
            // a center-only or sparse point test can miss a channel between
            // samples. Signed distance minus a containing radius cannot.
            float cx=(x0+x1)*.5f,cy=(y0+y1)*.5f,rx=(x1-x0)*.5f,ry=(y1-y0)*.5f;
            float radius=std::hypot(rx,ry);
            if(shore_sample_at(cx,cy).distance<radius)clipped=true;
            float screen_radius=std::hypot((rx+ry)*64,(rx+ry)*32);
            if(natural.river_sample({cx,cy}).distance<9+std::max(screen_radius,radius*64))clipped=true;
            if(clipped)continue;
            float ground_h=height_natural(float(nc)+u,float(nr)+1-v);
            // Uniform source XYZ scale precedes the documented source-to-world
            // basis conversion; normals use its inverse transpose.
            constexpr float z_basis=150.f/(.82f*64.f);
            for(auto const&p:body.vertices){
                float x=float(nc)+u+(p.position[0]*co-p.position[1]*si)*scale;
                float y=float(nr)+1-v-(p.position[0]*si+p.position[1]*co)*scale;
                auto out=project_natural(x,y,ground_h+p.position[2]*scale*z_basis*112);
                float n[]={p.normal[0]*co-p.normal[1]*si,-(p.normal[0]*si+p.normal[1]*co),p.normal[2]/z_basis};normalize3(n);
                out.normal_x=n[0];out.normal_y=n[1];out.normal_z=n[2];out.u=p.uv[0];out.v=p.uv[1];
                out.material_grass=1;out.material_plains=mat.repeat?2.f:0.f;out.material_desert=mat.channels[3]!=0xffffffffu?1.f:0.f;
                out.material_marsh=mat.channels[4]!=0xffffffffu?1.f:0.f;out.authored_relief_height=float(mat.tint);
                out.authored_relief_blend=mat.channels[6]!=0xffffffffu?1.f:0.f;
                out.base_terrain=mat.repeat?41.f:40.f;
                natural_vertices[3+recipe->object].push_back(out);
            }
        }
    }
}
