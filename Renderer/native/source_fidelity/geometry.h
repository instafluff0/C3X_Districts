// Included inside the existing tile compiler: observes the same authoritative
// dependencies and emits into its bounded immutable world-geometry cache.
if(fidelity_profile) {
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
    auto const river_field=natural.retain_river_page(float(nc)+.5,float(nr)+.5);
    auto river_at=[&](float x,float y){return river_field->sample({x,y}).distance;};
    GroundProjection project_natural{nc,nr,half_w,half_h,relief_projection_scale,float(frame.target_height)};
    auto triangle=[](std::vector<Vertex>&out,Vertex const&a,Vertex const&b,Vertex const&c){out.push_back(a);out.push_back(b);out.push_back(c);};
    auto surface=[&](float u,float v){
        return ground_surface(project_natural,u,v,height_natural,shore_sample_at,material_weights_for);
    };
    // A mountain-neighborhood patch is emitted once as a continuous
    // terrain-relief surface below. Do not leave the ordinary ground mesh
    // underneath it: two coincident surfaces can never share depth, normals
    // and shadow reception exactly and were the source of the visible lips.
    bool unified_mountain_surface=false;
    for(int dr=-1;dr<=1;dr++)for(int dc=-1;dc<=1;dc++)
        unified_mountain_surface|=lookup_natural(nc+dc,nr+dr).real==6;
    if(!unified_mountain_surface &&
       (ground<11 || shore_sample_at(float(nc)+.5f,float(nr)+.5f).distance>-.8f)){
        auto coastal=shore_sample_at(float(nc)+.5f,float(nr)+.5f);
        unsigned divisions=coastal.rocky>.55 && std::abs(coastal.distance)<1.25 ? 48 : 16;
        if(!emit_ground_grid(natural_vertices[0],surface,cancelled,divisions,
                             index_natural_grids?&natural_grid_indices[0]:nullptr))return false;
    }
    auto*mountain_indices=index_natural_grids?&natural_grid_indices[1]:nullptr;
    #include "../../lab/shared/natural/surface_mesh_body.h"
    #include "../../lab/shared/natural/relief_mesh_body.h"
    #include "../../lab/shared/natural/vegetation_floor_mesh_body.h"
    #include "../city_fidelity/geometry.h"
    if(tile.real_terrain_type==7){
        // Exact current production building meshes/placement, used only as
        // exclusions. City appearance and its geometry path remain unchanged.
        std::vector<BuildingBounds> buildings;
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
        auto&hash=c3x_renderer::stable_hash;
        auto&random=c3x_renderer::stable_random;
        #include "../../lab/shared/natural/forest_mesh_body.h"
    }
}
