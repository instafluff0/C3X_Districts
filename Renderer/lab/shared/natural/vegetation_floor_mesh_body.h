// Shared vegetation-floor decal body. The four decoded source entries are
// terrain decals, not vegetation meshes. Jungle's ArtDef supplies eight of
// each decal; forest's zero-count decal pool is selected by each recipe's
// ShowDecal flag. Exact engine scatter remains unavailable, so this generic
// reconstruction preserves those associations, descriptor footprints,
// ArtDef scale/variation, stable world seeds and feature clipping.
    // Raised forest/jungle variants use tree meshes only. Their rock surfaces
    // must not inherit the lowland vegetation-floor decals.
    int floor_hill_canopy=c3x_renderer::native_hill_vegetation(
        tile.real_terrain_type==5?5:0,
        [&](int dx,int dy){return lookup_natural(nc+(dx+dy)/2,nr+(dx-dy)/2).real;},
        c3x_renderer::native_hill_seed(owner.source_x,owner.source_y));
    bool floor_hill_forest=floor_hill_canopy==7;
    int vegetation_kind=floor_hill_canopy?floor_hill_canopy:tile.real_terrain_type;
    if(vegetation_kind==7 || vegetation_kind==8){
        struct FloorDecal {float x0,y0,x1,y1,scale,variation;};
        constexpr FloorDecal forest_floor[]={
            {-1.25433612f,-1.25433612f,1.25433612f,1.25433612f,.35f,.15f},
            {-1.22909896f,-1.22909896f,1.22909896f,1.22909896f,.70f,.10f}};
        constexpr FloorDecal jungle_floor[]={
            {-1.10028418f,-1.10028418f,1.10028418f,1.10028418f,.60f,.25f},
            {-1.26879136f,-1.26879136f,1.26879136f,1.26879136f,.60f,.25f}};
        auto const*decals=vegetation_kind==7?forest_floor:jungle_floor;
        std::uint32_t floor_seed=std::uint32_t(owner.source_x*0x193u)^std::uint32_t(owner.source_y*0x217u)^0x6d2bu;
        float edge=vegetation_kind==7?.01f:.03f;
        auto coverage_at=[&](float x,float y){
            float coverage=0;
            for(int r=int(std::floor(y))-1;r<=int(std::floor(y))+1;r++)for(int c=int(std::floor(x))-1;c<=int(std::floor(x))+1;c++){
                if(lookup_natural(c,r).real!=vegetation_kind &&
                    !(floor_hill_canopy && c==nc && r==nr))continue;
                float dx=std::max({float(c)-x,0.f,x-float(c+1)});
                float dy=std::max({float(r)-y,0.f,y-float(r+1)});
                float distance=std::hypot(dx,dy);
                coverage=std::max(coverage,1-smooth01(distance/std::max(.001f,edge)));
            }
            return coverage;
        };
        std::uint32_t feature_seed=floor_seed^0x6d2bu;
        unsigned forest_density=floor_hill_forest?16u:31+c3x_renderer::stable_hash(feature_seed^0xa53du)%11u;
        unsigned jungle_density=floor_hill_canopy==8?16u:31u+c3x_renderer::stable_hash(feature_seed^0xa53du)%11u;
        unsigned instance_count=vegetation_kind==7?forest_density:16u;
        for(unsigned instance=0;instance<instance_count;instance++){
            if(cancelled())return false;
            unsigned variant;
            float center_x,center_y;
            if(vegetation_kind==7){
                // Match the corresponding forest-body selection and center.
                unsigned selected=c3x_renderer::stable_hash(feature_seed+instance*31u)%180;
                Recipe const*recipe=nullptr;
                for(unsigned j=0;j<25;j++){auto const&r=natural.recipes[j];if(selected<r.count){recipe=&r;break;}selected-=r.count;}
                if(!recipe)return false;
                if((recipe->flags&2u)==0)continue;
                // ShowDecal marks eligibility, not a requirement that every
                // overlapping tree receive a floor patch. Retain a stable
                // half of eligible placements to avoid a continuous mat
                // without reducing the authored patch footprint.
                if((c3x_renderer::stable_hash(feature_seed+instance*131u+83u)&1u)!=0)continue;
                float ring=std::sqrt((float(instance)+.5f)/float(forest_density));
                float angle=2.39996323f*float(instance)+c3x_renderer::stable_random(feature_seed^0x71b3u)*6.283185307f;
                center_x=float(nc)+.5f+std::cos(angle)*ring*.43f;
                center_y=float(nr)+.5f-std::sin(angle)*ring*.43f;
                variant=c3x_renderer::stable_hash(feature_seed+instance*43u+17u)&1u;
            }else{
                // The ArtDef has two eight-count decals. Spread all sixteen
                // over actual jungle body centers using its forest-style
                // spiral, rather than an independent regular grid.
                variant=instance&1u;
                unsigned slot=(c3x_renderer::stable_hash(feature_seed)%jungle_density+
                    instance*jungle_density/16u)%jungle_density;
                float ring=std::sqrt((float(slot)+.5f)/float(jungle_density));
                float angle=2.39996323f*float(slot)+c3x_renderer::stable_random(feature_seed^0x71b3u)*6.283185307f;
                center_x=float(nc)+.5f+std::cos(angle)*ring*.43f;
                center_y=float(nr)+.5f-std::sin(angle)*ring*.43f;
                // Reproduce the corresponding body's water exclusion with its
                // actual rotated source hull, so a clipped palm cannot leave
                // an orphaned floor mark along a river or coast.
                unsigned selected=c3x_renderer::stable_hash(feature_seed+slot*31u)%121u;
                Recipe const*recipe=nullptr;
                for(unsigned j=25;j<35;j++){auto const&r=natural.recipes[j];if(selected<r.count){recipe=&r;break;}selected-=r.count;}
                if(!recipe)return false;
                float tree_scale=recipe->scale*(1+recipe->variation*(c3x_renderer::stable_random(feature_seed+slot*71u+23u)*2-1))*(floor_hill_canopy==8?.34f:.40f);
                float tree_yaw=c3x_renderer::stable_random(feature_seed+slot*97u+47u)*6.283185307f;
                float tree_co=std::cos(tree_yaw),tree_si=std::sin(tree_yaw);
                float x0=1e9f,y0=1e9f,x1=-1e9f,y1=-1e9f;
                for(auto const&p:natural.bodies[recipe->object].vertices){
                    float x=center_x+(p.position[0]*tree_co-p.position[1]*tree_si)*tree_scale;
                    float y=center_y-(p.position[0]*tree_si+p.position[1]*tree_co)*tree_scale;
                    x0=std::min(x0,x);x1=std::max(x1,x);y0=std::min(y0,y);y1=std::max(y1,y);
                }
                float rx=(x1-x0)*.5f,ry=(y1-y0)*.5f;
                float radius=std::hypot(rx,ry);
                if(shore_sample_at((x0+x1)*.5f,(y0+y1)*.5f).distance<radius ||
                   river_at((x0+x1)*.5f,(y0+y1)*.5f)<9+std::max(std::hypot((rx+ry)*64,(rx+ry)*32),radius*64))continue;
            }
            auto const&decal=decals[variant];
            // The vegetation bodies are reduced to the Civ III scene basis by
            // these same factors. Apply that conversion to their source decal
            // footprints so the floor treatment stays beneath the foliage.
            float scene_scale=.28f;
            float scale=decal.scale*scene_scale*(1+decal.variation*(c3x_renderer::stable_random(floor_seed+variant*211u+instance*37u)*2-1));
            float yaw=c3x_renderer::stable_random(floor_seed+variant*307u+instance*53u)*6.283185307f;
            float co=std::cos(yaw),si=std::sin(yaw);
            {
                // The patch's opaque core stays off kept-clear routes, resources
                // and sites, river channels and the shore; its soft rim may meet them.
                float core=.6f*(std::abs(co)+std::abs(si))*std::max(decal.x1-decal.x0,decal.y1-decal.y0)*.5f*scale;
                if(clearing.covers(center_x-core,center_y-core,center_x+core,center_y+core) ||
                   river_at(center_x,center_y)<9+core*64 || shore_sample_at(center_x,center_y).distance<core)continue;
            }
            auto point=[&](unsigned x,unsigned y){
                float sx=decal.x0+(decal.x1-decal.x0)*x/8.f;
                float sy=decal.y0+(decal.y1-decal.y0)*y/8.f;
                float world_x=center_x+(sx*co-sy*si)*scale;
                float world_y=center_y+(sx*si+sy*co)*scale;
                float h=height_natural(world_x,world_y)+.18f;
                auto out=project_natural(world_x,world_y,h);
                out.u=x/8.f;out.v=y/8.f;
                out.material_plains=vegetation_kind==7?3.f:4.f;
                out.base_terrain=-10+coverage_at(world_x,world_y);
                return out;
            };
            for(unsigned y=0;y<8;y++)for(unsigned x=0;x<8;x++){
                auto a=point(x,y),b=point(x+1,y),c=point(x+1,y+1),d=point(x,y+1);
                if(std::max({a.base_terrain,b.base_terrain,c.base_terrain,d.base_terrain})<=-9.999f)continue;
                triangle(natural_vertices[1],a,b,c);triangle(natural_vertices[1],a,c,d);
            }
        }
    }
