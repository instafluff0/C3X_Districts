// Shared statement body: retain the native x86 calling context and float rounding.
// Included by the native tile compiler and the portable typed adapter in mesh.h.
    {
        // One shared shape for the mesh, route surfaces and resource seating.
        MountainShape const mountain_shape(natural,nc,nr,lookup_natural,mountain_flags);
        auto mountain_at=[&](float world_x,float world_y){return mountain_shape.sample(natural,world_x,world_y);};
        if(mountain_shape.count){
            unsigned const count=patch_detail.mountain+1,span=count+2;
            float const step=1/float(patch_detail.mountain);
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
            std::vector<float> surface_height(span*span),surface_hill_support(span*span);
            // The vertex grid reuses these exact halo-grid points' samples.
            std::vector<MountainSample> mountain_samples(span*span);
            for(unsigned y=0;y<span;y++){
                if(cancelled())return false;
                for(unsigned x=0;x<span;x++){
                    float world_x=float(nc)+(int(x)-1)*step;
                    float world_y=float(nr)+1-(int(y)-1)*step;
                    auto sample=mountain_samples[y*span+x]=mountain_at(world_x,world_y);
                    auto shore=shore_sample_at(world_x,world_y);
                    float scale=coast_relief(float(shore.distance),float(shore.beach_width))*
                        mountain_river_scale(world_x,world_y)*hidden_taper_at(world_x,world_y);
                    float support=0;
                    surface_height[y*span+x]=height_natural(world_x,world_y,&support)+
                        sample.displacement*scale;
                    surface_hill_support[y*span+x]=support;
                }
            }
            std::vector<Vertex> grid(count*count);
            for(unsigned y=0;y<count;y++){
                if(cancelled())return false;
                for(unsigned x=0;x<count;x++){
                    float world_x=float(nc)+x*step,world_y=float(nr)+1-y*step;
                    auto const& sample=mountain_samples[(y+1)*span+x+1];
                    auto shore=shore_sample_at(world_x,world_y);
                    float coast_scale=coast_relief(float(shore.distance),float(shore.beach_width));
                    float river_scale=mountain_river_scale(world_x,world_y);
                    unsigned at=(y+1)*span+x+1;
                    float elevation=surface_height[at];
                    auto out=project_natural(world_x,world_y,elevation);
                    float mountain_height=sample.displacement*coast_scale*river_scale*hidden_taper_at(world_x,world_y);
                    out.world_valid=1+std::max(0.f,(elevation-mountain_height-2.5f)/112);
                    float n[]={-(surface_height[at+1]-surface_height[at-1])/(2*step*112),
                        (surface_height[at+span]-surface_height[at-span])/(2*step*112),1};normalize3(n);
                    float flat_blend=smooth01(mountain_height/1.f);
                    // The flat fringe replaces ordinary terrain. At its outer
                    // handoff use that provider's exact finite-difference
                    // normal; blend into the joined mountain normal on rise.
                    if(mountain_height<1.f){
                        constexpr float e=.006f;
                        float ground_n[]={-(height_natural(world_x+e,world_y,nullptr)-
                            height_natural(world_x-e,world_y,nullptr))/(2*e*128),
                            -(height_natural(world_x,world_y+e,nullptr)-
                            height_natural(world_x,world_y-e,nullptr))/(2*e*128),1};normalize3(ground_n);
                        for(unsigned axis=0;axis<3;axis++)n[axis]=ground_n[axis]*(1-flat_blend)+n[axis]*flat_blend;
                        normalize3(n);
                    }
                    out.normal_x=n[0];out.normal_y=n[1];out.normal_z=n[2];out.u=sample.u;out.v=sample.v;
                    out.material_grass=std::max(0.f,(elevation-2.5f)/112)*(1-flat_blend)+sample.height*flat_blend;
                    // 2..2.5 carries the dominant stamp's Civ III snow cap.
                    out.material_plains=2+.5f*sample.snow*flat_blend;
                    // Volcano material: texture offset from the stamp centre,
                    // coverage, and 2 x activity + authored lava channel.
                    out.relief_owner_u=sample.volcano_u-.5f;out.relief_owner_v=sample.volcano_v-.5f;
                    out.relief_owner_coverage=sample.volcano;
                    out.relief_owner_state=2.f*float(sample.activity)+sample.channel;
                    // Mountain stone follows mountain rise in the shader;
                    // this channel exclusively marks hill-owned ground.
                    out.material_desert=surface_hill_support[at]*hill_material(world_x-float(nc),
                        float(nr)+1-world_y);
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
            append_surface_grid(natural_vertices[2],grid,count-1,false,mountain_indices,&patch_layouts.get(count-1));
        }
    }
