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
    {
        struct MountainPiece {unsigned height_field,blend_field;float center_x,center_y,long_span,cross_span,height_scale;bool connected,range_y;};
        std::vector<MountainPiece> pieces;
        for(int dr=-1;dr<=1;dr++)for(int dc=-1;dc<=1;dc++){
            int pc=nc+dc,pr=nr+dr;Tile piece_owner=lookup_natural(pc,pr);
            if(piece_owner.real!=6)continue;
            bool west=lookup_natural(pc-1,pr).real==6,east=lookup_natural(pc+1,pr).real==6;
            bool north=lookup_natural(pc,pr-1).real==6,south=lookup_natural(pc,pr+1).real==6;
            unsigned along_x=unsigned(west)+unsigned(east),along_y=unsigned(north)+unsigned(south);
            bool connected=along_x+along_y>0,turn=connected&&along_x==along_y;
            unsigned variant=mountain_seed(piece_owner)%5u;
            pieces.push_back({natural.macro[variant][0],natural.macro[variant][1],
                float(pc)+.5f+.09f*(int(east)-int(west)),
                float(pr)+.5f+.09f*(int(south)-int(north)),
                connected?(turn?2.08f:2.46f):1.85f,
                connected?(turn?1.82f:1.34f):1.55f,
                connected?142.f:165.f,connected,along_y>along_x});
        }
        struct MountainSample {float displacement=0,dominant=0,height=0,blend=0,u=0,v=0;};
        auto mountain_at=[&](float world_x,float world_y){
            MountainSample result;
            for(auto const&piece:pieces){
                float source_x=piece.range_y?(world_y-piece.center_y)/piece.long_span:
                    (world_x-piece.center_x)/piece.long_span;
                float source_y=piece.range_y?(world_x-piece.center_x)/piece.cross_span:
                    (world_y-piece.center_y)/piece.cross_span;
                float u=.5f+source_x,v=.5f-source_y;
                if(u<0||u>1||v<0||v>1)continue;
                float h=natural.fields[piece.height_field].sample(u,v);
                float blend=natural.fields[piece.blend_field].sample(u,v);
                float shaped=piece.connected?std::pow(std::max(0.f,h),.80f):h;
                // Keep the outer authored footprint flat and terrain-colored,
                // then raise the mountain within its actual rocky body. The
                // former early ramp made low-slope ground into a circular berm
                // whose normals and contact shadow remained visible as a lip.
                float displacement=shaped*piece.height_scale*smooth01((blend-.28f)/.34f);
                if(displacement>result.dominant){
                    result.dominant=displacement;result.height=h;result.u=u;result.v=v;
                }
                if(displacement>0){
                    if(result.displacement<=0)result.displacement=displacement;
                    else{
                        float high=std::max(result.displacement,displacement);
                        float ridge=std::max(0.f,10.f-std::abs(result.displacement-displacement));
                        result.displacement=high+ridge*ridge/40.f;
                    }
                }
                result.blend=std::max(result.blend,blend);
            }
            return result;
        };
        if(!pieces.empty()){
            constexpr unsigned count=65,span=67;constexpr float step=1/64.f;
            // A coarser haloed distance field is aligned in world space and is
            // ample for the broad 16-pixel valley shoulder. Reconstructing it
            // for the fine mesh avoids thousands of river-page queries while
            // keeping shared-edge heights and normals identical.
            constexpr unsigned river_count=19;
            std::array<float,river_count*river_count> river_scales;
            for(unsigned y=0;y<river_count;y++)for(unsigned x=0;x<river_count;x++){
                float px=float(nc)+(float(x)-1)/16.f;
                float py=float(nr)+1-(float(y)-1)/16.f;
                river_scales[y*river_count+x]=smooth01((float(river_at(px,py))-6.f)/16.f);
            }
            auto mountain_river_scale=[&](float world_x,float world_y){
                float gx=std::clamp((world_x-float(nc))*16+1,0.f,17.9999f);
                float gy=std::clamp((float(nr)+1-world_y)*16+1,0.f,17.9999f);
                unsigned x=unsigned(std::floor(gx)),y=unsigned(std::floor(gy));
                float tx=gx-x,ty=gy-y;
                float a=river_scales[y*river_count+x]*(1-tx)+river_scales[y*river_count+x+1]*tx;
                float b=river_scales[(y+1)*river_count+x]*(1-tx)+river_scales[(y+1)*river_count+x+1]*tx;
                return a*(1-ty)+b*ty;
            };
            // A haloed scalar grid supplies joined height and normals with one
            // authoritative height/shore/river query per point. Shared world
            // coordinates still produce identical patch-edge samples.
            std::vector<float> surface_height(span*span);
            for(unsigned y=0;y<span;y++){
                if(cancelled())return false;
                for(unsigned x=0;x<span;x++){
                    float world_x=float(nc)+(int(x)-1)*step;
                    float world_y=float(nr)+1-(int(y)-1)*step;
                    auto sample=mountain_at(world_x,world_y);
                    auto shore=shore_sample_at(world_x,world_y);
                    float scale=coast_relief(float(shore.distance),float(shore.beach_width))*
                        mountain_river_scale(world_x,world_y);
                    surface_height[y*span+x]=height_natural(world_x,world_y,nullptr)+
                        sample.displacement*scale;
                }
            }
            std::vector<Vertex> grid(count*count);
            for(unsigned y=0;y<count;y++){
                if(cancelled())return false;
                for(unsigned x=0;x<count;x++){
                    float world_x=float(nc)+x*step,world_y=float(nr)+1-y*step;
                    auto sample=mountain_at(world_x,world_y);
                    auto shore=shore_sample_at(world_x,world_y);
                    float coast_scale=coast_relief(float(shore.distance),float(shore.beach_width));
                    float river_scale=mountain_river_scale(world_x,world_y);
                    sample.blend*=coast_scale*river_scale;
                    unsigned at=(y+1)*span+x+1;
                    float elevation=surface_height[at];
                    auto out=project_natural(world_x,world_y,elevation);
                    float mountain_height=sample.displacement*coast_scale*river_scale;
                    out.world_valid=1+std::max(0.f,(elevation-mountain_height-2.5f)/112);
                    float n[]={-(surface_height[at+1]-surface_height[at-1])/(2*step*112),
                        (surface_height[at+span]-surface_height[at-span])/(2*step*112),1};normalize3(n);
                    out.normal_x=n[0];out.normal_y=n[1];out.normal_z=n[2];out.u=sample.u;out.v=sample.v;
                    out.material_grass=sample.height;out.material_plains=2;out.material_desert=sample.blend;
                    auto weights=material_weights_for(world_x,world_y);
                    // Match the terrain provider's normalized source-family
                    // weights. Native marsh remains underneath at its boundary;
                    // the packed 42..43 range preserves the mountain caster tag
                    // while carrying the same coast coverage as the ground.
                    float source_weight=std::clamp(1-weights[3],0.f,1.f);
                    float desert_weight=std::clamp(weights[2]/std::max(source_weight,.00001f),0.f,1.f);
                    float coverage=coast_coverage(float(shore.distance),float(shore.beach_width))*(1-desert_weight)+
                        desert_coast_coverage(float(shore.distance))*desert_weight;
                    for(auto&weight:weights)weight/=std::max(source_weight,.00001f);
                    out.material_marsh=weights[4];
                    out.authored_relief_height=weights[1];
                    out.authored_relief_blend=weights[2];
                    out.base_terrain=42+coverage*source_weight;
                    grid[y*count+x]=out;
                }
            }
            for(unsigned y=0;y+1<count;y++)for(unsigned x=0;x+1<count;x++){
                auto&a=grid[y*count+x];auto&b=grid[y*count+x+1];auto&c=grid[(y+1)*count+x+1];auto&d=grid[(y+1)*count+x];
                triangle(natural_vertices[2],a,b,c);triangle(natural_vertices[2],a,c,d);
            }
        }
    }
