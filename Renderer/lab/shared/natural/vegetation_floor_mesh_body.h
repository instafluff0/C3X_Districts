// Shared vegetation-floor decal body. The four decoded source entries are
// terrain decals, not vegetation meshes. Jungle's ArtDef supplies eight of
// each decal; forest's zero-count decal pool is selected by each recipe's
// ShowDecal flag. Exact engine scatter remains unavailable, so this generic
// reconstruction preserves those associations, descriptor footprints,
// ArtDef scale/variation, stable world seeds and feature clipping.
    if(tile.real_terrain_type==7 || tile.real_terrain_type==8){
        struct FloorDecal {float x0,y0,x1,y1,scale,variation;};
        constexpr FloorDecal forest_floor[]={
            {-1.25433612f,-1.25433612f,1.25433612f,1.25433612f,.35f,.15f},
            {-1.22909896f,-1.22909896f,1.22909896f,1.22909896f,.70f,.10f}};
        constexpr FloorDecal jungle_floor[]={
            {-1.10028418f,-1.10028418f,1.10028418f,1.10028418f,.60f,.25f},
            {-1.26879136f,-1.26879136f,1.26879136f,1.26879136f,.60f,.25f}};
        auto const*decals=tile.real_terrain_type==7?forest_floor:jungle_floor;
        std::uint32_t floor_seed=std::uint32_t(owner.source_x*0x193u)^std::uint32_t(owner.source_y*0x217u)^0x6d2bu;
        float edge=tile.real_terrain_type==7?.01f:.03f;
        auto coverage_at=[&](float x,float y){
            float coverage=0;
            for(int r=int(std::floor(y))-1;r<=int(std::floor(y))+1;r++)for(int c=int(std::floor(x))-1;c<=int(std::floor(x))+1;c++){
                if(lookup_natural(c,r).real!=tile.real_terrain_type)continue;
                float dx=std::max({float(c)-x,0.f,x-float(c+1)});
                float dy=std::max({float(r)-y,0.f,y-float(r+1)});
                float distance=std::hypot(dx,dy);
                coverage=std::max(coverage,1-smooth01(distance/std::max(.001f,edge)));
            }
            return coverage;
        };
        std::uint32_t feature_seed=floor_seed^0x6d2bu;
        unsigned forest_density=31+c3x_renderer::stable_hash(feature_seed^0xa53du)%11u;
        unsigned instance_count=tile.real_terrain_type==7?forest_density:8u;
        for(unsigned instance=0;instance<instance_count;instance++){
            if(cancelled())return false;
            unsigned variant;
            float center_x,center_y;
            if(tile.real_terrain_type==7){
                // Match the corresponding forest-body selection and center.
                unsigned selected=c3x_renderer::stable_hash(feature_seed+instance*31u)%180;
                Recipe const*recipe=nullptr;
                for(auto const&r:natural.recipes){if(selected<r.count){recipe=&r;break;}selected-=r.count;}
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
                // Use eight representatives from the ArtDef's two eight-count
                // entries and anchor them to real 7x7 vegetation slots rather
                // than an independent decal scatter.
                variant=instance&1u;
                unsigned slot=(c3x_renderer::stable_hash(feature_seed)%49u+instance*6u)%49u;
                unsigned column=slot%7u,row=slot/7u;
                float jitter_u=c3x_renderer::stable_random(feature_seed+slot*103u+59u)-.5f;
                float jitter_v=c3x_renderer::stable_random(feature_seed+slot*107u+61u)-.5f;
                float u=.07f+.86f*(float(column)+.5f+jitter_u*.68f)/7.f;
                float v=.07f+.86f*(float(row)+.5f+jitter_v*.68f)/7.f;
                center_x=float(nc)+u;
                center_y=float(nr)+1-v;
            }
            auto const&decal=decals[variant];
            // The vegetation bodies are reduced to the Civ III scene basis by
            // these same factors. Apply that conversion to their source decal
            // footprints so the floor treatment stays beneath the foliage.
            float scene_scale=.28f;
            float scale=decal.scale*scene_scale*(1+decal.variation*(c3x_renderer::stable_random(floor_seed+variant*211u+instance*37u)*2-1));
            float yaw=c3x_renderer::stable_random(floor_seed+variant*307u+instance*53u)*6.283185307f;
            float co=std::cos(yaw),si=std::sin(yaw);
            auto point=[&](unsigned x,unsigned y){
                float sx=decal.x0+(decal.x1-decal.x0)*x/8.f;
                float sy=decal.y0+(decal.y1-decal.y0)*y/8.f;
                float world_x=center_x+(sx*co-sy*si)*scale;
                float world_y=center_y+(sx*si+sy*co)*scale;
                float h=height_natural(world_x,world_y)+.18f;
                auto out=project_natural(world_x,world_y,h);
                out.u=x/8.f;out.v=y/8.f;
                out.material_plains=tile.real_terrain_type==7?3.f:4.f;
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
