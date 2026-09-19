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
    c3x_renderer::city_fidelity::compile(cities.library,*composition,nc,nr,height_natural,project_natural,city_result);
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
