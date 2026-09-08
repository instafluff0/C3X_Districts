// Shared statement body: retain the native x86 calling context and float rounding.
// Included by the native tile compiler and the portable typed adapter in mesh.h.
        std::uint32_t seed=std::uint32_t(owner.source_x*0x193u)^std::uint32_t(owner.source_y*0x217u);
        unsigned density=31+hash(seed^0xa53du)%11u;
        for(unsigned i=0;i<density;i++){
            if(cancelled())return false;
            unsigned selected=hash(seed+i*31u)%180;Recipe const*recipe=nullptr;
            for(auto const&r:natural.recipes){if(selected<r.count){recipe=&r;break;}selected-=r.count;}
            if(!recipe)return false;
            float ring=std::sqrt((float(i)+.5f)/float(density));
            float angle=2.39996323f*float(i)+random(seed^0x71b3u)*6.283185307f;
            float u=.5f+std::cos(angle)*ring*.43f,v=.5f+std::sin(angle)*ring*.43f;
            float scale=recipe->scale*(1+recipe->variation*(random(seed+i*71u+23u)*2-1))*.46f;
            float yaw=random(seed+i*97u+47u)*6.283185307f;
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
            if(river_at(cx,cy)<9+std::max(screen_radius,radius*64))clipped=true;
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
