// Shared by the foreground and CPU preparation compiler; no GPU ownership.
    // A mountain-neighborhood patch is emitted once as a continuous
    // terrain-relief surface below. Do not leave the ordinary ground mesh
    // underneath it: two coincident surfaces can never share depth, normals
    // and shadow reception exactly and were the source of the visible lips.
    bool unified_mountain_surface=false,hill_surface=false;
    for(int dr=-1;dr<=1;dr++)for(int dc=-1;dc<=1;dc++){
        int real=lookup_natural(nc+dc,nr+dr).real;
        unified_mountain_surface|=real==6 || real==10;
        hill_surface|=real==5;
    }
    if(!unified_mountain_surface &&
       (ground<11 || shore_sample_at(float(nc)+.5f,float(nr)+.5f).distance>-.8f)){
        auto coastal=shore_sample_at(float(nc)+.5f,float(nr)+.5f);
        bool detailed=hill_surface || (coastal.rocky>.55 && std::abs(coastal.distance)<1.25) ||
            (river_terrain_near && project_natural.half_width>=48);
        unsigned divisions=detailed?patch_detail.mountain:std::min(16u,patch_detail.mountain);
        if(!emit_ground_grid(natural_vertices[0],surface,cancelled,divisions,
                             index_natural_grids?&natural_grid_indices[0]:nullptr,&patch_layouts.get(divisions),false,
                             patch_detail.mountain))return false;
        // Territory edges cross the wet portion of land-classified shore
        // tiles too. Keep the complete ground mesh for that pass; the terrain
        // and shadow shaders retain their existing optical coast clipping.
    }
    auto*mountain_indices=index_natural_grids?&natural_grid_indices[1]:nullptr;

    record_natural_phase(0);
    #include "../../lab/shared/natural/surface_mesh_body.h"
    record_natural_phase(1);

    #include "../../lab/shared/natural/relief_mesh_body.h"
    if(tile.real_terrain_type==5){
        unsigned layer=unified_mountain_surface?2u:0u;
        auto indices=index_natural_grids?&natural_grid_indices[unified_mountain_surface?1:0]:nullptr;
        if(!emit_hill_decal_surface(owner,nc,nr,natural_vertices[layer],indices,natural_vertices[1]))return false;
    }
    // Volcano ownership travels with the relief grid (relief_mesh_body.h).

    record_natural_phase(2);
    #include "../../lab/shared/natural/vegetation_floor_mesh_body.h"
    record_natural_phase(3);
