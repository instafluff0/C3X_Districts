// Shared statement body: retain the native x86 calling context and float rounding.
// Included by the native tile compiler and the portable typed adapter in mesh.h.
        std::uint32_t seed=std::uint32_t(owner.source_x*0x193u)^std::uint32_t(owner.source_y*0x217u);
        auto const&set=natural.forest_set(clearing.variety);
        // A resource tile keeps a thinner stand around it, as the source
        // resource clutter sets replace the full forest with a few trees.
        bool opening=clearing.open_radius>0 && !raised_canopy && !hill_forest;
        unsigned density=raised_canopy?4u:hill_forest?16u:opening?9u+hash(seed^0x5c1du)%4u:31+hash(seed^0xa53du)%11u;
        for(unsigned i=0;i<density;i++){
            if(cancelled())return false;
            unsigned selected=hash(seed+i*31u)%set.weight;Recipe const*recipe=nullptr;
            for(unsigned j=set.first;j<set.end;j++){auto const&r=natural.recipes[j];if(selected<r.count){recipe=&r;break;}selected-=r.count;}
            if(!recipe)return false;
            float ring=std::sqrt((float(i)+.5f)/float(density));
            float angle=2.39996323f*float(i)+random(seed^0x71b3u)*6.283185307f;
            float u=.5f+std::cos(angle)*ring*.43f,v=.5f+std::sin(angle)*ring*.43f;
            if(raised_canopy){
                constexpr float foot[][2]={{.06f,.90f},{.34f,.97f},{.69f,.97f},{.95f,.79f}};
                u=foot[i][0];v=foot[i][1];
            }
            float scale=recipe->scale*(1+recipe->variation*(random(seed+i*71u+23u)*2-1))*(raised_canopy?.27f:hill_forest?.34f:.46f);
            float yaw=random(seed+i*97u+47u)*6.283185307f;
            float co=std::cos(yaw),si=std::sin(yaw);auto const&body=natural.bodies[recipe->object];
            auto const&mat=natural.materials[body.material];
            if(opening){
                float reach=0;for(auto const&p:body.vertices)reach=std::max(reach,std::hypot(p.position[0],p.position[1])*scale);
                if(!clearing.open_place(nc,nr,angle,ring,reach,u,v))continue;
            }
            // Clip flags are authoritative source metadata. Test the body hull,
            // not just its center, against captured building and water geometry.
            float x0=1e9f,y0=1e9f,x1=-1e9f,y1=-1e9f,z1=0;
            for(auto const&p:body.vertices){float x=float(nc)+u+(p.position[0]*co-p.position[1]*si)*scale;
                float y=float(nr)+1-v-(p.position[0]*si+p.position[1]*co)*scale;
                x0=std::min(x0,x);x1=std::max(x1,x);y0=std::min(y0,y);y1=std::max(y1,y);z1=std::max(z1,p.position[2]);}
            bool clipped=clearing.blocks(x0,y0,x1,y1,z1*scale);
            for(auto const&b:buildings)if(x0<b.x1 && x1>b.x0 && y0<b.y1 && y1>b.y0)clipped=true;
            // Conservative footprint bounds against the authoritative fields:
            // a center-only or sparse point test can miss a channel between
            // samples. Signed distance minus a containing radius cannot.
            float cx=(x0+x1)*.5f,cy=(y0+y1)*.5f,rx=(x1-x0)*.5f,ry=(y1-y0)*.5f;
            float radius=std::hypot(rx,ry);
            if(shore_sample_at(cx,cy).distance<radius)clipped=true;
            float screen_radius=std::hypot((rx+ry)*64,(rx+ry)*32);
            if(river_at(cx,cy)<9+std::max(screen_radius,radius*64))clipped=true;
            if(clipped)continue;
            // Uniform source XYZ scale precedes the documented source-to-world
            // basis conversion; normals use its inverse transpose.
            constexpr float z_basis=150.f/(.82f*64.f);
            float ground_h=hill_forest
                ? 2.5f+c3x_renderer::native_hill_plant_ground(body,float(nc)+u,float(nr)+1-v,
                    co,si,scale,z_basis*112.f,height_natural)
                : height_natural(float(nc)+u,float(nr)+1-v);
            if(emit_forest_instance(recipe->object,nc,nr,u,v,co,si,scale,ground_h))continue;
            for(auto const&p:body.vertices){
                float x=float(nc)+u+(p.position[0]*co-p.position[1]*si)*scale;
                float y=float(nr)+1-v-(p.position[0]*si+p.position[1]*co)*scale;
                auto out=project_natural(x,y,ground_h+p.position[2]*scale*z_basis*112);
                float n[]={p.normal[0]*co-p.normal[1]*si,-(p.normal[0]*si+p.normal[1]*co),p.normal[2]/z_basis};normalize3(n);
                out.normal_x=n[0];out.normal_y=n[1];out.normal_z=n[2];out.u=p.uv[0];out.v=p.uv[1];
                out.material_grass=1;out.material_plains=(mat.repeat?2.f:0.f)+
                    (mat.channels[1]!=0xffffffffu && mat.channels[2]!=0xffffffffu?1.f:0.f);
                out.material_desert=mat.channels[3]!=0xffffffffu?1.f:0.f;
                out.material_marsh=mat.channels[4]!=0xffffffffu?1.f:0.f;out.authored_relief_height=float(mat.tint);
                out.authored_relief_blend=mat.channels[6]!=0xffffffffu?1.f:0.f;
                out.base_terrain=mat.repeat?41.f:40.f;
                natural_vertices[3+recipe->object].push_back(out);
            }
        }
