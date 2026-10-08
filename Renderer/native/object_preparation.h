#pragma once
#include "rigid_object_instance.h"
#include "city_fidelity/compiler.h"
#include "source_fidelity/terrain_compiler.h"
namespace c3x_renderer { namespace objects {
// Inputs own capture scalars. Packs, observations and world/coast are immutable
// until the frame lease joins; all mutable query state belongs to this lane.
struct PreparationInput {
    Projection projection;
    int ground=0;
    std::int64_t world_revision=0;
    bool river_ready=false,skip_flat_shore=true,separate_relief=true,retain_height=true;
    bool route_ready=false,routes_enabled=true,mine_ready=false,farm_ready=false,city_ready=false,composition_ready=false;
    bool shared_rigid=false;
    // A farm's resource parts as tile-local (u0,v0,u1,v1) boxes it keeps open.
    std::vector<std::array<float,4>> farm_resource;
};
struct PreparedPart {
    render_core::PreparedMesh mesh;
    unsigned vertex_offset=0,index_offset=0,material=0;
    bool environment=false,terrain_conforming=false,effect=false;
    std::array<float,4> atlas{};
    std::shared_ptr<city_fidelity::Lighting> lighting;
};
struct PreparedObjects {
    struct Draw {unsigned layer=0,first=0,count=0,rigid=~0u;};
    std::array<PreparedPart,layer_count> layers;
    std::vector<PreparedPart> city;
    std::vector<PreparedRigid> rigid;
    std::vector<Draw> draws;
    unsigned composition=~0u,instances=0,routes=0;
    std::shared_ptr<void> buffer;
    std::size_t gpu_bytes=0;
    std::unordered_map<std::uint64_t,std::uint64_t> topology,coast;
    std::unordered_map<std::size_t,std::uint32_t> world;
    fidelity::NaturalWorld::CellProof rivers;
    std::size_t proof_bytes=0;
    std::size_t bytes()const {
        std::size_t total=sizeof(*this)+city.capacity()*sizeof(PreparedPart)+rigid.capacity()*sizeof(PreparedRigid)+draws.capacity()*sizeof(Draw)+gpu_bytes+proof_bytes;
        for(auto const& part:layers)total+=part.mesh.bytes();
        for(auto const& part:city)total+=part.mesh.bytes();
        if(!city.empty())total+=sizeof(city_fidelity::Lighting)+city.front().lighting->lights.capacity()*sizeof(city_fidelity::Light)+
            city.front().lighting->blockers.capacity()*sizeof(city_fidelity::Lighting::Box);
        return total+(world.size()+coast.size()+topology.size())*64+
            (world.bucket_count()+coast.bucket_count()+topology.bucket_count())*sizeof(void*);
    }
};
using Preparation=render_core::ContentPreparation<unsigned,PreparationInput,PreparedObjects>;
// Destruction order joins the producer before scratch or source leases die.
struct PreparationLease {
    fidelity::TerrainCompileScratch scratch;
    Preparation queue;
};
template<class TerrainAssets,class Observations,class Stop>
std::unique_ptr<PreparedObjects> prepare(PreparationInput const& input,Assets const& assets,
        city_fidelity::Library const& library,fidelity::NaturalData const& natural,
        TerrainAssets const& terrain_assets,render_core::WorldCoast const& world_coast,
        Observations const& observations,fidelity::TerrainCompileScratch& scratch,Stop stop,bool bounded=true) {
    using namespace fidelity;
    if(stop())return {};
    auto result=std::make_unique<PreparedObjects>();
    auto const& capture=input.projection.tile;
    // Volcanoes pass too: an active one carries a plume.
    bool const volcano=capture.real_terrain_type==10 && input.city_ready;
    if(!capture.road_mask && !capture.railroad_mask && capture.city_id<0 && !volcano &&
        !(capture.improvement_flags&(C3X_RENDERER_IMPROVEMENT_MINE|C3X_RENDERER_IMPROVEMENT_IRRIGATION|
            C3X_RENDERER_IMPROVEMENT_GOODY_HUT|C3X_RENDERER_IMPROVEMENT_BARBARIAN_CAMP|
            C3X_RENDERER_IMPROVEMENT_POLLUTION|C3X_RENDERER_IMPROVEMENT_CRATER|C3X_RENDERER_IMPROVEMENT_RUINS)))return result;
    scratch.bind(natural,world_coast.world(),input.world_revision);
    NaturalWorld::CellInputs river_inputs;
    NaturalWorld::DependencyScope scope(scratch.rivers,&river_inputs);
    scratch.pickup.clear();scratch.heights.clear();
    auto observe_world=[&](std::size_t index,std::uint32_t value){result->world.emplace(index,value);};
    auto observe_coast=[&](std::uint64_t index,std::uint64_t revision){result->coast.emplace(index,revision);};
    SurfaceQueries queries(world_coast,scratch.shores,input.projection.tile.tile_x,input.projection.tile.tile_y,observe_world,observe_coast,input.skip_flat_shore,&scratch.rivers);
    auto world_lookup=[&](int c,int r){return queries.tile(c,r);};
    auto lookup_natural=[&](int c,int r){return queries.natural_tile(c,r);};
    auto shore_sample_at=[&](float x,float y){return queries.shore(x,y);};
    auto material_weights_for=[&](float x,float y){return queries.weights(x,y);};
    auto relief_sample=[&](int kind,unsigned variant,int channel,float u,float v){return relief_source(terrain_assets,true,kind,variant,channel,u,v);};
    auto river=[&](int c,int r,float u,float v){
        auto const& world=world_coast.world();auto i=world.index(c,r);auto value=world.at(i);
        if(i!=std::size_t(-1))observe_world(i,value);
        if(value==0xffffffffu || !(value>>16&170u) || !input.river_ready)return 1000.f;
        return float(scratch.rivers.river_sample({float(c)+u,float(r)+1-v}).distance);
    };
    auto dune=[](float,float){return 0.f;};
    auto activity=[&](int c,int r){auto const& world=world_coast.world();auto i=world.index(c,r);auto value=world.at(i);
        if(i!=std::size_t(-1))observe_world(i,value);return value!=0xffffffffu && ((value>>24)&1u)!=0?1.f:0.f;};
    int nc=(input.projection.tile.tile_x+input.projection.tile.tile_y)/2,nr=(input.projection.tile.tile_x-input.projection.tile.tile_y)/2;
    std::size_t height_queries=0;
    ReliefSurface pickup_surface(world_coast.world().dimensions(),nc,nr,
        shore_sample_at(queries.center_u,queries.center_v).distance,world_lookup,relief_sample,shore_sample_at,river,dune,activity,
        scratch.pickup,height_queries,input.separate_relief);
    auto pickup_height=[&](float x,float y){return pickup_surface.height(x,y);};
    auto height_natural=[&](float x,float y,float* support=nullptr){
        auto compute=[&]{std::array<float,2> value{};value[0]=queries.height(natural,pickup_height,x,y,&value[1]);return value;};
        auto value=input.retain_height?scratch.heights.get(x,y,compute):compute();
        if(support)*support=value[1];return value[0];
    };
    // Civ III draws routes over a mountain tile's lower art but never over
    // its peak. Routes follow the whole rendered mountain surface (so the rock
    // never cuts them along a ragged contour), and a pattern route fades out
    // between these heights above the natural ground as the mountain rises.
    constexpr float pattern_route_fade_start=35.f,pattern_route_fade_end=65.f;
    // A railroad tunnel's portal stands on the ground at the mountain's foot,
    // where the rail rises tunnel_foot over the land with rock (the fade's
    // start) within tunnel_reach tiles; its grey block reaches back into the
    // mountain (tunnel_portal_length). Rise and rock are world fields, so
    // neighboring tiles agree.
    constexpr float tunnel_foot=4.f,tunnel_reach=.2f;
    // The rendered mesh's own shape (lab/shared/natural/mountain_shape.h),
    // built only for route and volcano tiles, so routes lie on its rock and a
    // plume leaves its crater.
    MountainShape mountain_shape;
    if(input.projection.tile.road_mask || input.projection.tile.railroad_mask || volcano)
        mountain_shape=MountainShape(natural,nc,nr,lookup_natural,[](int,int){return false;});
    unsigned const mountain_count=mountain_shape.count;
    auto route_height=[&](float x,float y){
        float base=height_natural(x,y);
        if(!mountain_count)return base;
        float displacement=mountain_shape.sample(natural,x,y).displacement;
        if(displacement<=0)return base;
        auto shore=queries.shore(x,y);
        float river_scale=input.river_ready?
            smooth01((float(scratch.rivers.river_sample({x,y}).distance)-6.f)/16.f):1.f;
        return base+displacement*coast_relief(float(shore.distance),float(shore.beach_width))*river_scale*queries.hidden_taper(x,y);
    };

    auto relief=[&](float u,float v){auto s=pickup_surface.sample(u,v);
        float farm_clearance=1.0f;
        if(input.farm_ready && (input.projection.tile.improvement_flags&C3X_RENDERER_IMPROVEMENT_IRRIGATION)){
            farm_clearance=float(queries.shore(u,v).distance);
            // The river surface ends at 7.4 source pixels from its centerline.
            // Leave a narrow bank before placing a field or raised farm piece.
            if(input.river_ready)farm_clearance=std::min(farm_clearance,
                (float(scratch.rivers.river_sample({u,v}).distance)-8.5f)/64.0f);
            // ...and stops at the foot of a mountain or a neighbouring hill.
            float hill_support=0,flat_ground=height_natural(u,v,&hill_support);
            farm_clearance=std::min(farm_clearance,farm_slope_clearance(
                mountain_count?route_height(u,v)-flat_ground:0.f,hill_support,input.projection.tile.real_terrain_type==5));
        }
        return std::array<float,3>{s.height,s.authored_height,farm_clearance};};
    auto const& tile=input.projection.tile;
    auto composition=input.composition_ready?city_fidelity::select(library,tile,nc,nr,world_lookup,shore_sample_at,
        [&](float x,float y){return scratch.rivers.river_sample({x,y}).distance;},height_natural):nullptr;
    if(composition)result->composition=unsigned(composition-library.compositions.data());
    Plan plan;
    select_routes(tile,assets,input.route_ready,input.routes_enabled,[&](int x,int y){
        auto key=observations.key(x,y);auto record=observations.current(key);
        result->topology.emplace(key,record?record->semantic:0);return record;
    },plan);
    if(input.river_ready && input.route_ready && input.routes_enabled)
        promote_river_crossings(tile,[&](float x,float y){
            return float(scratch.rivers.river_sample({x,y}).distance);
        },plan,&assets);
    // A railroad running into a mountain enters a tunnel (tunnel_route);
    // roads fade out as they climb. A rail line's end on an edge shared by
    // two mountain or volcano tiles (both mountain shapes) lies inside their
    // range.
    FeatureGroup const* tunnel=find_feature_group(assets[bridge_family],"tunnel_railroad");
    constexpr int natural_offsets[8][2]={{0,1},{1,1},{1,0},{1,-1},{0,-1},{-1,-1},{-1,0},{-1,1}};
    if(mountain_count)for(auto& route:plan.patterns){
        auto const* set=assets.patterns_for(route.style);
        if(!set || route.line>=set->lines.size())continue;
        auto const& line=set->lines[route.line];
        float tile_u=float(tile.tile_x+tile.tile_y)*.5f,tile_v=float(tile.tile_x-tile.tile_y)*.5f;
        std::vector<float> fade(line.count,0.f),rise(line.count,0.f);bool any=false;
        bool rail_tunnel=route.style>=4u && tunnel;
        std::vector<float> depth(line.count,-1.f);
        for(unsigned index=0;index<line.count;++index){
            auto const point=route.points.size()==line.count?route.points[index]:set->points[line.first+index];
            float x=tile_u+point[0],y=tile_v+1.f-point[1];
            rise[index]=route_height(x,y)-height_natural(x,y);
            fade[index]=smooth01((rise[index]-pattern_route_fade_start)/(pattern_route_fade_end-pattern_route_fade_start));
            any=any || fade[index]>0.f;
            if(rail_tunnel){
                float rock=rise[index];
                for(unsigned k=0;k<8u;++k){
                    float a=float(k)*.7853982f,sx=x+tunnel_reach*std::cos(a),sy=y+tunnel_reach*std::sin(a);
                    rock=std::max(rock,route_height(sx,sy)-height_natural(sx,sy));
                }
                depth[index]=std::min(rise[index]-tunnel_foot,rock-pattern_route_fade_start);
                any=any || depth[index]>=0.f;
            }
        }
        std::array<bool,2> inside{};
        auto range=[](int real){return real==6 || real==10;};
        if(rail_tunnel && range(tile.real_terrain_type))for(unsigned side=0;side<2;++side){
            int k=side?line.end:line.start;
            if(k>=0 && k<8)inside[side]=range(lookup_natural(nc+natural_offsets[k][0],nr+natural_offsets[k][1]).real);
        }
        bool tunnelled=rail_tunnel && (any || inside[0] || inside[1]);
        if(tunnelled){
            std::size_t first=plan.instances.size();
            tunnel_route(route,*set,depth,inside,*tunnel,plan,fade);
            // Each portal's block reaches back into its mountain: the rock
            // over its base along the mesh's +y (world u,v).
            for(std::size_t i=first;i<plan.instances.size();++i){
                auto& portal=plan.instances[i];
                if(tunnel->placements.size()<2 || portal.asset!=tunnel->placements[1].asset_index)continue;
                float fx=tile_u+portal.u,fy=tile_v+1.f-portal.v,base=route_height(fx,fy);
                float du=-std::sin(portal.rotation),dv=-std::cos(portal.rotation);
                portal.asset=tunnel->placements[tunnel_portal_length(assets[bridge_family],*tunnel,portal.scale,
                    [&](float s){return route_height(fx+du*s,fy+dv*s)-base;})].asset_index;
            }
        }
        if(any || tunnelled)route.fade=std::move(fade);
    }
    unsigned sites=tile.improvement_flags&(C3X_RENDERER_IMPROVEMENT_GOODY_HUT|C3X_RENDERER_IMPROVEMENT_BARBARIAN_CAMP|
        C3X_RENDERER_IMPROVEMENT_POLLUTION|C3X_RENDERER_IMPROVEMENT_CRATER|C3X_RENDERER_IMPROVEMENT_RUINS);
    if(!select_improvements(tile,assets,input.ground,sites,input.mine_ready,input.farm_ready,plan))return {};
    if(input.farm_ready && (tile.improvement_flags&C3X_RENDERER_IMPROVEMENT_IRRIGATION)){
        clear_farm(plan,plan,assets,input.farm_resource);
        // A plot kit lays out the areas it shares with neighbouring farms:
        // their routes and irrigation are dependencies, as the routes' are.
        settle_farm_fields(plan,tile,assets,relief,[&](int dx,int dy)->c3x_renderer_tile_v1 const*{
            auto key=observations.key(tile.tile_x+dx,tile.tile_y+dy);auto record=observations.current(key);
            result->topology.emplace(key,record?record->semantic:0);return record?&record->occurrence:nullptr;});
        settle_farm_props(plan,tile,assets,relief);
    }
    result->instances=unsigned(plan.instances.size());result->routes=unsigned(plan.routes.size()+plan.patterns.size());
    // Bound worker transients before expanded triangles are constructed. Large
    // valid packs can still use the same synchronous compiler on recovery.
    std::uint64_t raw_bytes=0;
    for(auto const& route:plan.routes)
        raw_bytes+=std::uint64_t(route.bridge?336u:192u)*sizeof(Vertex);
    for(auto const& route:plan.patterns)if(auto const* patterns=assets.patterns_for(route.style))
        if(route.line<patterns->lines.size())
            raw_bytes+=(std::uint64_t(patterns->lines[route.line].count)*18u+24u)*sizeof(Vertex);
    for(auto const& instance:plan.instances){
        if(instance.asset>=assets[instance.family].assets.size())continue; // same absent-asset behavior as append_instance
        auto const& asset=assets[instance.family].assets[instance.asset];
        auto triangles=asset.id.rfind("farm_",0)==0 && !shared_rigid_mesh(asset)
            ? asset.indices.size()*2u : asset.indices.size();
        raw_bytes+=(triangles+asset.vertices.size())*sizeof(Vertex);}
    if(composition){
        raw_bytes+=(composition->paving.indices.size()+composition->paving.vertices.size())*sizeof(Vertex);
        for(auto const& instance:composition->instances)for(auto const& part:library.models[instance.model].parts)
            raw_bytes+=(part.indices.size()+part.vertices.size())*sizeof(Vertex);
    }
    if(stop() || (bounded && raw_bytes>32u*1024u*1024u))return {};
    Surfaces surfaces;
    std::vector<unsigned> surface_draws;
    if(input.shared_rigid){
        std::vector<Instance> surfaces_only;
        std::array<unsigned,layer_count> indices{};
        for(auto const& instance:plan.instances){
            if(stop())return {};
            if(instance.asset>=assets[instance.family].assets.size())continue;
            auto const& asset=assets[instance.family].assets[instance.asset];
            if(asset.vertices.empty() || asset.indices.empty())continue;
            if(instance.family==farm_family && (asset.id.find(":base:")!=std::string::npos ||
                    asset.id.find(":building:")!=std::string::npos ||
                    asset.id.find(":tree:")!=std::string::npos)){
                float u=float(capture.tile_x+capture.tile_y)*.5f+instance.u;
                float v=float(capture.tile_x-capture.tile_y)*.5f+1.0f-instance.v;
                float shore=relief(u,v)[2];
                float clearance=asset.id.find(":base:")!=std::string::npos?.55f:
                    asset.id.find(":building:")!=std::string::npos?.14f:.11f;
                if(shore<clearance)continue;
            }
            if(shared_rigid_mesh(asset)){
                result->draws.push_back({unsigned(instance.layer),0,0,unsigned(result->rigid.size())});
                // A tunnel entrance stands on its rail.
                bool tunnel_part=asset.id.rfind("route/tunnel/",0)==0;
                result->rigid.push_back(prepare_rigid(instance,input.projection,assets,relief,
                    [&](float x,float y){return tunnel_part?route_height(x,y):height_natural(x,y);},
                    plan.farm_kit && instance.family==farm_family));
            }else{
                unsigned count=unsigned(asset.indices.size());
                if(!result->draws.empty() && result->draws.back().layer==unsigned(instance.layer) && result->draws.back().rigid==~0u)
                    result->draws.back().count+=count;
                else result->draws.push_back({unsigned(instance.layer),indices[instance.layer],count,~0u});
                indices[instance.layer]+=count;surfaces_only.push_back(instance);
                surface_draws.push_back(unsigned(result->draws.size()-1));
            }
        }
        plan.instances=std::move(surfaces_only);
    }
    std::vector<unsigned> instance_counts;
    compile(plan,input.projection,assets,relief,route_height,surfaces,true,
        input.shared_rigid?&instance_counts:nullptr);
    if(input.shared_rigid){
        for(auto& draw:result->draws)if(draw.rigid==~0u)draw.count=0;
        for(std::size_t index=0;index<instance_counts.size();++index)
            result->draws[surface_draws[index]].count+=instance_counts[index];
        std::array<unsigned,layer_count> offsets{};
        for(auto& draw:result->draws)if(draw.rigid==~0u){
            draw.first=offsets[draw.layer];offsets[draw.layer]+=draw.count;
        }
        result->draws.erase(std::remove_if(result->draws.begin(),result->draws.end(),
            [](PreparedObjects::Draw const& draw){return draw.rigid==~0u && draw.count==0;}),
            result->draws.end());
    }
    for(unsigned layer=0;layer<layer_count;++layer){
        render_core::MeshFormat format;format.feature=layer!=route_layer;
        format.projection_kind=layer==route_layer && input.projection.world_objects?1:2;
        if(!render_core::prepare_mesh(surfaces.layers[layer],layer==route_layer?nullptr:&surfaces.indices[layer],format,result->layers[layer].mesh,stop))return {};
        std::vector<Vertex>().swap(surfaces.layers[layer]);
        if(bounded && result->bytes()>Preparation::byte_limit/2)return {};
    }
    auto adopt_city=[&](city_fidelity::Surfaces& city){
        for(auto& chunk:city.chunks){
            PreparedPart part;part.material=chunk.material;part.environment=chunk.environment;
            part.terrain_conforming=chunk.terrain_conforming;part.lighting=chunk.lighting;part.effect=chunk.effect;
            std::copy(chunk.atlas,chunk.atlas+4,part.atlas.begin());
            render_core::MeshFormat format;format.projection_kind=4;format.city=true;
            if(!render_core::prepare_mesh(chunk.vertices,&chunk.indices,format,part.mesh,stop))return false;
            std::vector<Vertex>().swap(chunk.vertices);result->city.push_back(std::move(part));
            if(bounded && result->bytes()>Preparation::byte_limit/2)return false;
        }
        return true;
    };
    if(composition){
        city_fidelity::Surfaces city;
        GroundProjection projection{nc,nr,input.projection.half_w,input.projection.half_h,
            input.projection.relief_projection_scale,float(input.projection.content_view_height)};
        // Version-five packs let marked bodies yield to rivers, water and
        // steep or mountain ground (the same test as the direct path).
        if(!city_fidelity::compile(library,*composition,nc,nr,height_natural,projection,city,stop,true,
                city_fidelity::site_filter(*composition,nc,nr,world_lookup,shore_sample_at,
                    [&](float x,float y){return scratch.rivers.river_sample({x,y}).distance;},
                    [&](float x,float y){return height_natural(x,y);})))return {};
        if(!adopt_city(city))return {};
    }
    // An active volcano's plume (world bits 24/27, observed through the
    // queries) in the city effect layer, rising from the rendered crater.
    if(volcano && library.effect_material<library.materials.size()){
        auto self=queries.tile(nc,nr);
        if(self.real==10 && self.active){
            float x=float(nc)+.5f,y=float(nr)+.5f;
            city_fidelity::Surfaces plume;
            GroundProjection projection{nc,nr,input.projection.half_w,input.projection.half_h,
                input.projection.relief_projection_scale,float(input.projection.content_view_height)};
            if(city_fidelity::volcano_plume(library,nc,nr,route_height(x,y),self.erupting,projection,true,plume) &&
               !adopt_city(plume))return {};
        }
    }
    if(stop())return {};
    result->rivers.assign(river_inputs.begin(),river_inputs.end());
    result->proof_bytes=scratch.rivers.proof_bytes(result->rivers);
    return result;
}
}}
