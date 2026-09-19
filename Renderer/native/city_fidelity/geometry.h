// Same production queries serve city assembly and forest exclusions. Compilation
// emits owned CPU chunks; GPU adoption and tracing remain on the render owner.
auto selected_city=[&](c3x_renderer_tile_v1 const& record,int column,int row){
    if(!city_profile || !cities.ready)return static_cast<c3x_renderer::city_fidelity::Composition const*>(nullptr);
    return c3x_renderer::city_fidelity::select(cities.library,record,column,row,world_lookup,shore_sample_at,
        [&](float x,float y){return natural.river_sample({x,y}).distance;},height_natural);
};
if(auto composition=selected_city(tile,nc,nr)){
    c3x_renderer::city_fidelity::Surfaces output;
    c3x_renderer::city_fidelity::compile(cities.library,*composition,nc,nr,height_natural,project_natural,output);
    city_chunks=std::move(output.chunks);
    city_vertices.clear();
    char detail[256];sprintf_s(detail,"city=%d authority=%s instances=%u lights=%u paving=%u textures_bytes=%llu cache=world-immutable",
        tile.city_id,composition->authority.c_str(),unsigned(composition->instances.size()),
        city_chunks.empty()?0u:unsigned(city_chunks.front().lighting->lights.size()),
        unsigned(composition->paving.vertices.size()),static_cast<unsigned long long>(cities.texture_bytes));trace.write("city-composition",detail,true);
} else if(city_profile && tile.city_id>=0){
    char detail[160];sprintf_s(detail,"city=%d culture=%d era=%d size=%d reason=no-legal-immutable-composition",
        tile.city_id,tile.city_culture_group,tile.city_era,tile.city_size);trace.write("city-composition-fallback",detail,true);
}
