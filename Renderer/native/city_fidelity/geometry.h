// Same production queries serve city assembly and forest exclusions. Compilation
// emits owned CPU chunks; GPU adoption and tracing remain on the render owner.
auto selected_city=[&](c3x_renderer_tile_v1 const& record,int column,int row){
    if(&record==&tile)return tile_city_composition;
    if(!city_profile || !cities.ready)return static_cast<c3x_renderer::city_fidelity::Composition const*>(nullptr);
    return c3x_renderer::city_fidelity::select(cities.library,record,column,row,world_lookup,shore_sample_at,
        [&](float x,float y){return natural.river_sample({x,y}).distance;},height_natural);
};
if(auto composition=selected_city(tile,nc,nr)){
    if(!prepared_objects){
    c3x_renderer::city_fidelity::Surfaces city_result;
    // Version-five packs let marked bodies yield to rivers, water and steep
    // or mountain ground; the same test excludes them from forest clearing.
    c3x_renderer::city_fidelity::compile(cities.library,*composition,nc,nr,height_natural,project_natural,city_result,
        c3x_renderer::city_fidelity::ContinueCompilation{},false,
        c3x_renderer::city_fidelity::site_filter(*composition,nc,nr,world_lookup,shore_sample_at,
            [&](float x,float y){return natural.river_sample({x,y}).distance;},height_natural));
    city_chunks=std::move(city_result.chunks);
    for(auto const& chunk:city_chunks){if(chunk.terrain_conforming)++city_deformed_parts;else ++city_rigid_parts;}
    }
    city_vertices.clear();
    char detail[256];sprintf_s(detail,"city=%d authority=%s instances=%u lights=%u paving=%u textures_bytes=%llu cache=world-immutable",
        tile.city_id,composition->authority.c_str(),unsigned(composition->instances.size()),
        prepared_objects && !prepared_objects->city.empty()?unsigned(prepared_objects->city.front().lighting->lights.size()):
        city_chunks.empty()?0u:unsigned(city_chunks.front().lighting->lights.size()),
        unsigned(composition->paving.vertices.size()),static_cast<unsigned long long>(cities.texture_bytes));trace.write("city-composition",detail,true);
} else if(city_profile && tile.city_id>=0){
    char detail[160];sprintf_s(detail,"city=%d culture=%d era=%d size=%d reason=no-legal-immutable-composition",
        tile.city_id,tile.city_culture_group,tile.city_era,tile.city_size);trace.write("city-composition-fallback",detail,true);
}
// Constructed wonders (city_fidelity/wonders.h) on the direct path, which runs on the
// frame (object preparation off; production always prepares objects): it takes only
// resident groups, else the era's Wonder District, and never loads.
{
    c3x_renderer::city_fidelity::WonderFact wonder_fact;
    if(!prepared_objects && city_profile && cities.ready && c3x_renderer::city_fidelity::wonder_fact(tile,wonder_fact)){
        auto& store=c3x_renderer::city_fidelity::wonder_store();
        auto group=store.resident(store.key(wonder_fact));
        if(!group)group=store.resident(store.district(wonder_fact));
        if(group)for(auto const& template_:group->library.compositions){
            c3x_renderer::city_fidelity::Surfaces bodies;
            auto const placed=c3x_renderer::city_fidelity::wonder_spans_river(template_.authority)?
                c3x_renderer::city_fidelity::across_river(template_,tile.river_code):template_;
            c3x_renderer::city_fidelity::compile(group->library,placed,nc,nr,height_natural,project_natural,bodies);
            auto const stand=static_cast<unsigned char>(c3x_renderer::city_fidelity::wonder_season_mask(template_.authority));
            for(auto& chunk:bodies.chunks){
                chunk.material+=group->material_base;chunk.season_stand=stand;chunk.lease=group;
                city_chunks.push_back(std::move(chunk));
            }
        }
    }
}
