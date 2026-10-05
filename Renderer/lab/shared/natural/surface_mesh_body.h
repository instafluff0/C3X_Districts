// Shared statement body: deterministic source-decal composition on the joined
// production ground. The normalized pack supplies biome-neutral placement
// records; the renderer owns only stable selection and C3X world placement.
    if((owner.real>=0 && owner.real<=2) || owner.real==4) {
        unsigned biome=owner.real==4?3u:unsigned(2-owner.real); // grass, plains, desert, floodplain
        unsigned total=0;
        for(auto const&candidate:natural.surface_recipes)
            if(candidate.biome==biome)total+=candidate.weight;
        if(total) {
            std::uint32_t state=std::uint32_t(owner.source_x*0x193u)^
                std::uint32_t(owner.source_y*0x217u)^0x51f3a9u;
            // Exact source triangles make the land families readable without
            // stamping a rectangular copy of the complete atlas. Keep the
            // authored recipe weights, but vary the visible patch count by the
            // source tile coordinates as well as position, rotation and scale.
            // Desert dunes retain their sparse regional distribution.
            bool sparse_desert=biome==2 && (random_u32(state)&1u)!=0;
            // Grass source recipes are disconnected authored triangles whose
            // straight mesh edges remain visible even when their color alpha
            // is faded. The continuous grass material supplies this detail.
            unsigned density=biome==0 || sparse_desert?0u:
                (biome==2?3u:biome==3?4u+random_u32(state)%3u:14u+random_u32(state)%9u);
            for(unsigned ordinal=0;ordinal<density;ordinal++) {
                if(cancelled())return false;
                unsigned selected=random_u32(state)%total;SurfaceRecipe const*recipe=nullptr;
                for(auto const&candidate:natural.surface_recipes)if(candidate.biome==biome) {
                    if(selected<candidate.weight){recipe=&candidate;break;}
                    selected-=candidate.weight;
                }
                if(!recipe)return false;
                // Fine grass/plains patches may straddle owner boundaries.
                // The former inset left a systematic empty strip once their
                // scale was reduced. Desert keeps its established placement.
                float center_x=float(nc)+(biome==2?.08f+.84f*random01(state):random01(state));
                float center_y=float(nr)+(biome==2?.08f+.84f*random01(state):random01(state));
                float angle=random01(state)*6.283185307f;
                // Grass and plains need fine grain. Preserve the established
                // desert dune footprint at biome boundaries.
                float scale=recipe->scale*(biome==2?.32f:biome==3?.50f:.18f)*
                    (1+recipe->variation*(random01(state)*2-1));
                float co=std::cos(angle),si=std::sin(angle);
                float patch_normal[]={0,0,1};
                if(biome!=2) {
                    constexpr float normal_step=.006f;
                    patch_normal[0]=-(height_natural(center_x+normal_step,center_y,nullptr)-
                        height_natural(center_x-normal_step,center_y,nullptr))/(2*normal_step*128);
                    patch_normal[1]=-(height_natural(center_x,center_y+normal_step,nullptr)-
                        height_natural(center_x,center_y-normal_step,nullptr))/(2*normal_step*128);
                    normalize3(patch_normal);
                }
                for(unsigned triangle_index=0;triangle_index<recipe->vertex_count;triangle_index+=3) {
                    if(cancelled())return false;
                    std::array<Vertex,3> projected;
                    for(unsigned corner=0;corner<3;corner++) {
                        auto const&source=natural.surface_vertices[recipe->first+triangle_index+corner];
                        float du=source.x*scale,dv=source.y*scale;
                        float world_x=center_x+co*du-si*dv;
                        float world_y=center_y+si*du+co*dv;
                        auto out=project_natural(world_x,world_y,biome==2?2.68f:
                            height_natural(world_x,world_y,nullptr)+.18f);
                        out.normal_x=patch_normal[0];
                        out.normal_y=patch_normal[1];
                        out.normal_z=patch_normal[2];
                        out.u=source.u;out.v=source.v;
                        auto weights=material_weights_for(world_x,world_y);
                        float source_weight=std::clamp(1-weights[3],0.f,1.f);
                        float decal_weight=weights[biome==3?1:biome];
                        if(biome==3) {
                            float gx=world_x-.5f,gy=world_y-.5f;
                            int x0=int(std::floor(gx)),y0=int(std::floor(gy));
                            float tx=std::clamp((gx-float(x0)-.2f)/.6f,0.f,1.f);
                            float ty=std::clamp((gy-float(y0)-.2f)/.6f,0.f,1.f);
                            tx=tx*tx*(3-2*tx);ty=ty*ty*(3-2*ty);
                            auto flood=[&](int x,int y){return lookup_natural(x,y).real==4?1.f:0.f;};
                            float flood_upper=flood(x0,y0)*(1-tx)+flood(x0+1,y0)*tx;
                            float flood_lower=flood(x0,y0+1)*(1-tx)+flood(x0+1,y0+1)*tx;
                            decal_weight*=flood_upper*(1-ty)+flood_lower*ty;
                        }
                        out.material_grass=0;out.material_plains=5+float(biome);
                        out.material_desert=decal_weight;out.material_marsh=0;
                        out.authored_relief_height=0;out.authored_relief_blend=0;
                        // A decal can cross the coast even when its tile center
                        // is inland. Match the ground at each actual vertex;
                        // center coverage left fully opaque patch edges on sand.
                        auto shore=shore_sample_at(world_x,world_y);
                        float coverage=biome==2
                            ? desert_coast_coverage(float(shore.distance))
                            : coast_coverage(float(shore.distance),float(shore.beach_width));
                        out.base_terrain=-10+coverage*source_weight;
                        out.world_valid=1+coast_ramp((float(shore.distance)-float(shore.beach_width)-.10f)/.90f);
                        projected[corner]=out;
                    }
                    if(std::max({projected[0].material_desert,projected[1].material_desert,
                                 projected[2].material_desert})<.015f)continue;
                    triangle(natural_vertices[1],projected[0],projected[1],projected[2]);
                }
            }
        }
    }
