// Included inside the existing tile compiler: observes the same authoritative
// dependencies and emits into its bounded immutable world-geometry cache.
if(fidelity_profile) {
    begin_natural_phase();
    using namespace c3x_renderer::fidelity;
    int nc=(tile.tile_x+tile.tile_y)/2,nr=(tile.tile_x-tile.tile_y)/2;
    auto lookup_natural=[&](int c,int r){
        return queries.natural_tile(c,r);
    };
    Tile owner=lookup_natural(nc,nr);
    auto height_natural=[&](float x,float y,float*support=nullptr){
        return natural_height_at(x,y,support);
    };
    // The page includes the authoritative halo needed by this complete tile
    // and its finite-difference collar. Bind it once: mountain tessellation
    // performs thousands of distance samples and must not re-enter the LRU for
    // every vertex and normal tap.
    auto const river_field=natural.bind_river_page(float(nc)+.5,float(nr)+.5);
    auto river_at=[&](float x,float y){return river_field.sample({x,y}).distance;};
    GroundProjection project_natural{nc,nr,half_w,half_h,relief_projection_scale,float(frame.target_height)};
    auto triangle=[](std::vector<Vertex>&out,Vertex const&a,Vertex const&b,Vertex const&c){out.push_back(a);out.push_back(b);out.push_back(c);};
    auto surface=[&](float u,float v){
        return ground_surface(project_natural,u,v,height_natural,shore_sample_at,material_weights_for);
    };
    if(!cpu_terrain_enabled) {
        #include "terrain_mesh_body.h"
    }
    #include "../city_fidelity/geometry.h"
    record_natural_phase(4);
    if(tile.real_terrain_type==7){
        // Exact current production building meshes/placement, used only as
        // exclusions. City appearance and its geometry path remain unchanged.
        std::vector<BuildingBounds> buildings;
        constexpr char const*eras[]={"ancient","medieval","industrial","modern"};
        constexpr unsigned counts[]={4,7,11};constexpr float radii[]={.25f,.33f,.41f},scales[]={.92f,1,1.08f};
        for(int r=nr-2;r<=nr+2;r++)for(int c=nc-2;c<=nc+2;c++){
            auto it=topology_cache.current(observed_coordinate_key(c+r,c-r));if(!it)continue;
            auto const&city=it->occurrence;if(city.city_id<0)continue;
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
        auto emit_forest_instance=[&](unsigned body,int c,int r,float u,float v,float co,float si,float scale,float ground_h){
            if(!tree_instances_enabled)return false;
            MeshInstance instance;float data[]={float(c),float(r),u,v,co,si,scale,ground_h};std::copy(data,data+8,instance.place);
            forest_instances[body].push_back(instance);
            auto& bounds=forest_bounds[body];
            if(forest_instances[body].size()==1)for(unsigned axis=0;axis<3;++axis){bounds.low[axis]=1e9f;bounds.high[axis]=-1e9f;}
            constexpr float z_basis=150.f/(.82f*64.f);
            for(auto const&p:natural.bodies[body].vertices){
                float x=float(c)+u+(p.position[0]*co-p.position[1]*si)*scale;
                float y=float(r)+1-v-(p.position[0]*si+p.position[1]*co)*scale;
                float h=ground_h+p.position[2]*scale*z_basis*112;
                float world[]={x,y,h/112};for(unsigned axis=0;axis<3;++axis){bounds.low[axis]=std::min(bounds.low[axis],world[axis]);bounds.high[axis]=std::max(bounds.high[axis],world[axis]);}
                forest_projected[body].include(x,y,h/112);
            }
            return true;
        };
        auto&hash=c3x_renderer::stable_hash;
        auto&random=c3x_renderer::stable_random;
        #include "../../lab/shared/natural/forest_mesh_body.h"
    }
    record_natural_phase(5);
    if(cpu_terrain_enabled) {
        // Join only after independent city/forest assembly has used this read lease.
        auto input=terrain_compile_input(tile,frame,ground,skip_flat_shore,separate_natural_relief,index_natural_grids,retain_height_samples,world_objects);
        auto prepared=prepared_world?std::move(prepared_world->terrain):terrain_preparation.take(input.key,true);
        if(prepared && !terrain_result_valid(*prepared))prepared.reset();
        if(!prepared){
            prepared=compile_terrain(input,foreground_terrain_scratch,cancelled,false);
            if(prepared && !attach_terrain_vertex_buffer(*prepared))prepared.reset();
        }
        if(!prepared)return false;
        world_dependencies.insert(prepared->world.begin(),prepared->world.end());
        coast_dependencies.insert(prepared->coast.begin(),prepared->coast.end());
        river_dependencies.insert(prepared->rivers.begin(),prepared->rivers.end());
        prepared_terrain=std::move(prepared);
        record_natural_phase(3);
    }
}
