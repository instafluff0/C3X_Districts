// Shared by the foreground and CPU preparation compiler; no GPU ownership.
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
        unsigned divisions=coastal.rocky>.55 && std::abs(coastal.distance)<1.25 ? patch_detail.rocky_ground : 16;
        if(!emit_ground_grid(natural_vertices[0],surface,cancelled,divisions,
                             index_natural_grids?&natural_grid_indices[0]:nullptr,&patch_layouts.get(divisions)))return false;
    }
    auto*mountain_indices=index_natural_grids?&natural_grid_indices[1]:nullptr;

    record_natural_phase(0);
    #include "../../lab/shared/natural/surface_mesh_body.h"
    record_natural_phase(1);

    #include "../../lab/shared/natural/relief_mesh_body.h"
    // Carry local volcano ownership on both replacement surface families.
    // Lookup uses the authoritative dependency observer, including wrapped tiles.
    // Geometry and inherited relief normals remain unchanged.
    std::vector<std::array<float,2>> volcano_centers;
    for(int dr=-1;dr<=1;dr++)for(int dc=-1;dc<=1;dc++)
        if(lookup_natural(nc+dc,nr+dr).real==10)
            volcano_centers.push_back({float(nc+dc)+.5f,float(nr+dr)+.5f});
    if(!volcano_centers.empty())for(unsigned layer:{0u,2u})
        for(auto&v:natural_vertices[layer]) {
            float nearest=1e9f;
            for(auto const&center:volcano_centers) {
                float dx=v.world_x-center[0],dy=v.world_y-center[1];
                float distance=dx*dx+dy*dy;
                if(distance<nearest) {
                    nearest=distance;
                    v.relief_owner_u=dx;v.relief_owner_v=dy;
                    v.relief_owner_coverage=1;v.relief_owner_state=0;
                }
            }
        }

    record_natural_phase(2);
    #include "../../lab/shared/natural/vegetation_floor_mesh_body.h"
    record_natural_phase(3);

