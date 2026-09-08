// Shared statement body: deterministic source-decal composition on the joined
// production ground. The normalized pack supplies biome-neutral placement
// records; the renderer owns only stable selection and C3X world placement.
    (void)height_natural;
    if(owner.real>=0 && owner.real<=2) {
        unsigned biome=unsigned(2-owner.real); // grass, plains, desert
        unsigned total=0;
        for(auto const&candidate:natural.surface_recipes)
            if(candidate.biome==biome)total+=candidate.weight;
        if(total) {
            std::uint32_t state=std::uint32_t(owner.source_x*0x193u)^
                std::uint32_t(owner.source_y*0x217u)^0x51f3a9u;
            // Exact source triangles make the land families readable without
            // stamping a rectangular copy of the complete atlas. Desert dunes
            // remain regional; grass and plains carry a denser source-clutter
            // layer like their substantially larger authored count sets.
            bool sparse_desert=biome==2 && (random_u32(state)&1u)!=0;
            unsigned density=sparse_desert?0u:(biome==2?3u:5u);
            auto owner_shore=shore_sample_at(float(nc)+.5f,float(nr)+.5f);
            float owner_coverage=biome==2
                ? desert_coast_coverage(float(owner_shore.distance))
                : coast_coverage(float(owner_shore.distance),float(owner_shore.beach_width));
            for(unsigned ordinal=0;ordinal<density;ordinal++) {
                if(cancelled())return false;
                unsigned selected=random_u32(state)%total;SurfaceRecipe const*recipe=nullptr;
                for(auto const&candidate:natural.surface_recipes)if(candidate.biome==biome) {
                    if(selected<candidate.weight){recipe=&candidate;break;}
                    selected-=candidate.weight;
                }
                if(!recipe)return false;
                float center_x=float(nc)+.08f+.84f*random01(state);
                float center_y=float(nr)+.08f+.84f*random01(state);
                float angle=random01(state)*6.283185307f;
                float scale=recipe->scale*.32f*(1+recipe->variation*(random01(state)*2-1));
                float co=std::cos(angle),si=std::sin(angle);
                for(unsigned triangle_index=0;triangle_index<recipe->vertex_count;triangle_index+=3) {
                    if(cancelled())return false;
                    std::array<Vertex,3> projected;
                    for(unsigned corner=0;corner<3;corner++) {
                        auto const&source=natural.surface_vertices[recipe->first+triangle_index+corner];
                        float du=source.x*scale,dv=source.y*scale;
                        float world_x=center_x+co*du-si*dv;
                        float world_y=center_y+si*du+co*dv;
                        auto out=project_natural(world_x,world_y,2.68f);
                        out.normal_x=out.normal_y=0;out.normal_z=1;
                        out.u=source.u;out.v=source.v;
                        auto weights=material_weights_for(world_x,world_y);
                        float source_weight=std::clamp(1-weights[3],0.f,1.f);
                        out.material_grass=0;out.material_plains=5+float(biome);
                        out.material_desert=weights[biome];out.material_marsh=0;
                        out.authored_relief_height=0;out.authored_relief_blend=0;
                        out.base_terrain=-10+owner_coverage*source_weight;
                        out.world_valid=1+coast_ramp((float(owner_shore.distance)-float(owner_shore.beach_width)-.10f)/.90f);
                        projected[corner]=out;
                    }
                    if(std::max({projected[0].material_desert,projected[1].material_desert,
                                 projected[2].material_desert})<.015f)continue;
                    triangle(natural_vertices[1],projected[0],projected[1],projected[2]);
                }
            }
        }
    }
