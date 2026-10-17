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

    // Raised ground facing an unexplored tile in front of it (screen-down
    // edges x=nc+1 and y=nr) stands above that tile's flat fog diamond, and
    // nothing is drawn there: the flat underlay showed between the edge and
    // the fog as a dull band (October 17). The surface continues down from
    // this edge's exact vertices, with their materials, to the datum a
    // quarter tile beyond it, so the fog fades over ground at any elevation.
    // Revealing the tile recompiles this one without it.
    for(int side=0;side<2;++side){
        if(!queries.tile(side?nc:nc+1,side?nr-1:nr).unexplored)continue;
        float const edge=side?float(nr):float(nc+1);
        auto along=[&](Vertex const& v){return side?v.world_x:v.world_y;};
        for(unsigned layer:{0u,2u}){
            std::vector<Vertex> rim;
            for(auto const& v:natural_vertices[layer])
                if(std::fabs((side?v.world_y:v.world_x)-edge)<1e-4f)rim.push_back(v);
            std::sort(rim.begin(),rim.end(),[&](Vertex const& a,Vertex const& b){return along(a)<along(b);});
            rim.erase(std::unique(rim.begin(),rim.end(),[&](Vertex const& a,Vertex const& b){
                return std::fabs(along(a)-along(b))<1e-4f;}),rim.end());
            auto datum=[&](Vertex v){
                auto low=project_natural(v.world_x+(side?0.f:.25f),v.world_y-(side?.25f:0.f),2.5f);
                v.x=low.x;v.y=low.y;v.z=low.z;v.world_x=low.world_x;v.world_y=low.world_y;v.world_z=low.world_z;
                return v;};
            auto& out=natural_vertices[layer];
            for(std::size_t i=0;i+1<rim.size();++i){
                if(rim[i].world_z*112<=2.51f && rim[i+1].world_z*112<=2.51f)continue;
                Vertex quad[]={rim[i],rim[i+1],datum(rim[i+1]),datum(rim[i])};
                if(index_natural_grids){
                    auto first=unsigned(out.size());
                    out.insert(out.end(),std::begin(quad),std::end(quad));
                    for(unsigned k:{0u,1u,2u,0u,2u,3u})natural_grid_indices[layer==0?0:1].push_back(first+k);
                }else for(unsigned k:{0u,1u,2u,0u,2u,3u})out.push_back(quad[k]);
            }
        }
    }
    // Native fog blends half a shroud width into an unexplored tile, showing
    // its rim (VisibilityCoverage). Flat ground at the datum with that rim's
    // own materials and no relief or features reaches a quarter tile into
    // every unexplored edge and corner neighbour.
    for(int dr=-1;dr<=1;++dr)for(int dc=-1;dc<=1;++dc){
        if((!dc && !dr) || !queries.tile(nc+dc,nr+dr).unexplored)continue;
        auto flat=[](float,float,float* support){if(support)*support=0;return 2.5f;};
        float u0=dc>0?1.f:dc<0?-.25f:0.f,v0=dr<0?1.f:dr>0?-.25f:0.f;
        float du=dc?.25f:1.f,dv=dr?.25f:1.f;unsigned nu=dc?1u:8u,nv=dr?1u:8u;
        std::vector<Vertex> rim;
        for(unsigned j=0;j<=nv;++j)for(unsigned i=0;i<=nu;++i){
            auto v=ground_surface(project_natural,u0+du*float(i)/float(nu),v0+dv*float(j)/float(nv),
                flat,shore_sample_at,material_weights_for);
            v.river_distance=river_terrain_near?float(river_at(v.world_x,v.world_y)):1000.f;
            rim.push_back(v);
        }
        auto& out=natural_vertices[0];
        for(unsigned j=0;j<nv;++j)for(unsigned i=0;i<nu;++i){
            unsigned a=j*(nu+1)+i,b=a+1,c=b+nu+1,d=a+nu+1;
            // Open water: the ground compiler's rim carries it.
            if(std::max({rim[a].base_terrain,rim[b].base_terrain,rim[c].base_terrain,rim[d].base_terrain})<=-9.999f)continue;
            Vertex quad[]={rim[a],rim[b],rim[c],rim[d]};
            if(index_natural_grids){
                auto first=unsigned(out.size());
                out.insert(out.end(),std::begin(quad),std::end(quad));
                for(unsigned k:{0u,1u,2u,0u,2u,3u})natural_grid_indices[0].push_back(first+k);
            }else for(unsigned k:{0u,1u,2u,0u,2u,3u})out.push_back(quad[k]);
        }
    }
