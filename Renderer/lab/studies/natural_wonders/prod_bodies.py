"""Natural-wonder bodies on the resource-body path (prod_tree.py merges these).

The kit named by a tile's record places its compositions from the NaturalWonderRuntime
bundle through objects::append_instance, seated like resource bodies. Each season mask
becomes its own feature chunk (season_stand), flagged wonder_body: drawing it swaps
the wonder bundle's eight textures for the frame season into slots 116-123 (where the
resource bundle sits) and back, in both draw paths and the shadow casters.
"""

MEMBERS = """    c3x_renderer::FeatureBundle resource_bundle;
    std::array<std::vector<std::uint8_t>, 8> resource_texture_dds;
    std::array<ID3D11ShaderResourceView *, 8> resource_texture_views = {};
    // Natural-wonder bodies (NaturalWonderRuntime): their own composition bundle and up
    // to eight season atlases that wonders.bin's season slot table swaps in.
    c3x_renderer::FeatureBundle wonder_bundle;
    std::array<std::vector<std::uint8_t>, 8> wonder_texture_dds, wonder_season_dds;
    std::array<ID3D11ShaderResourceView *, 8> wonder_texture_views = {}, wonder_season_views = {};
"""

VIEWS = """        return c3x_renderer::render_core::season_stand_shown(chunk.season_stand,season_index);
    }
    // The wonder bundle's eight textures for the current season (season slot 8 + n is
    // its season atlas n; a missing one keeps the summer slot).
    std::array<ID3D11ShaderResourceView*,8> wonder_views() const {
        auto views=wonder_texture_views;
        if(natural.wonder_pack.kits.empty())return views;
        for(unsigned i=0;i<8;++i){
            unsigned slot=natural.wonder_pack.season_slots[unsigned(season_index)&3u][i];
            auto* view=slot<8?wonder_texture_views[slot]:slot-8<8?wonder_season_views[slot-8]:nullptr;
            if(view)views[i]=view;
        }
        return views;
    }"""

LOAD = """        resource_assets_ready = load_runtime_bundle(
            resource_root, "resource_runtime.bin", resource_bundle, resource_texture_dds, true);
        // Natural-wonder bodies: optional; without the pack wonder tiles keep their terrain.
        std::string wonder_root = packs_root + "\\\\NaturalWonderRuntime";
        wonder_assets_ready = load_runtime_bundle(
            wonder_root, "resource_runtime.bin", wonder_bundle, wonder_texture_dds, true);
        for (std::size_t index = 0; wonder_assets_ready && index < wonder_season_dds.size(); ++index) {
            std::string name = "textures\\\\season_" + std::to_string(index) + ".dds";
            if (!load_dds_bytes(wonder_root.c_str(), name.c_str(), wonder_season_dds[index],
                                DXGI_FORMAT_BC1_UNORM_SRGB, DXGI_FORMAT_BC1_UNORM) &&
                !load_dds_bytes(wonder_root.c_str(), name.c_str(), wonder_season_dds[index], DXGI_FORMAT_BC3_UNORM_SRGB))
                wonder_season_dds[index].clear();
        }
"""

DEVICE = """        if (resource_assets_ready) {
            for (std::size_t index = 0; index < resource_texture_views.size(); ++index)
                resource_assets_ready = resource_assets_ready && ensure_dds_texture(
                    resource_texture_dds[index], resource_texture_views[index], true);
        }
        if (wonder_assets_ready) {
            for (std::size_t index = 0; index < wonder_texture_views.size(); ++index)
                wonder_assets_ready = wonder_assets_ready && ensure_dds_texture(
                    wonder_texture_dds[index], wonder_texture_views[index], true);
            for (std::size_t index = 0; index < wonder_season_views.size(); ++index)
                if (!wonder_season_dds[index].empty() &&
                    !ensure_dds_texture(wonder_season_dds[index], wonder_season_views[index], true))
                    wonder_season_dds[index].clear();
        }
"""

PLACEMENT = """                        item.lift, item.ground_fit);
                }
            }
            // A natural wonder's bodies (docs/natural_wonder_rendering.md): its record names the
            // kit, whose compositions place from the wonder bundle as resource bodies are, one
            // vertex list per season mask (emitted below as wonder_body chunks).
            if (auto const * kit = wonder_assets_ready && tile.natural_wonder[0] ?
                    natural.wonder_pack.find(tile.natural_wonder) : nullptr) {
                for (auto const & body : kit->bodies) {
                    auto const * composition = c3x_renderer::find_feature_composition(wonder_bundle, body.first.c_str());
                    if (!composition || composition->variants.empty() || !body.second) continue;
                    for (auto const & item : composition->variants.front().instances) {
                        c3x_renderer::FeaturePlacement placement; placement.asset_index = item.asset;
                        c3x_renderer::objects::append_instance(object_projection, wonder_bundle, placement,
                            item.u, item.v, item.rotation, item.scale, 21.0f, 0.0f, true, tile.terrain_type < 11,
                            relief_at_world, resource_height_at, wonder_vertices[body.second & 15u], shadow_vertices, nullptr,
                            item.lift, item.ground_fit);
                    }
                }
            }
            if(!prepared_objects){
                c3x_renderer::objects::Plan plan;"""

EMIT = """            // Natural-wonder bodies: their own feature chunks per season mask, drawn with the
            // wonder bundle's textures (wonder_body) and shown in their seasons (season_stand).
            // Placed like resource bodies, they take the feature layer's projection.
            if(!(shared_hit && !component_preparation && world_objects))for(unsigned mask=1;mask<16;++mask){
                auto& vertices=wonder_vertices[mask];if(vertices.empty())continue;
                auto prior=compiled.mesh->layers[geometry_feature].size();
                if(!cache_geometry_layer(mesh_uploads[shared_layer(geometry_feature)?1:0],vertices,
                        compiled.mesh->layers[geometry_feature],prewarming,
                        compiled.byte_count,foreground_pending,true,false,nullptr,nullptr,nullptr,nullptr,world_objects?2u:0u)){
                    tile_geometry_cache_bytes-=compiled.byte_count;release_geometry_vertex_buffers(compiled.mesh->layers);return false;
                }
                for(auto i=prior;i<compiled.mesh->layers[geometry_feature].size();++i){
                    auto& chunk=compiled.mesh->layers[geometry_feature][i];
                    chunk.wonder_body=true;chunk.season_stand=static_cast<unsigned char>(mask==15?0:mask);
                    season_stands_used|=std::uint16_t(1u<<chunk.season_stand);
                    // Its own version: shadow pages key casters by version, layer and count.
                    chunk.version=++tile_geometry_version;compiled.byte_count+=chunk.byte_count;
                }
            }
            // Keep allocation boundaries equal to residency/eviction owners:
"""

WONDER_BIND = """            if(layer>=20000){
                // Natural-wonder bodies: the wonder bundle's textures where the resource bundle's sit.
                auto views=alpha;auto wonder=wonder_views();std::copy(wonder.begin(),wonder.end(),views.begin()+21);
                context->PSSetShaderResources(0,33,views.data());return true;
            }
            if(layer>=10000){"""

FRESH_WONDER_BIND = """    bool bind_cutout(unsigned layer) {
        auto* context=renderer.context;
        if (layer>=20000) {
            // Natural-wonder bodies: the wonder bundle's textures where the resource bundle's sit.
            std::array<ID3D11ShaderResourceView*,33> views{};
            std::copy(renderer.feature_texture_views.begin(),renderer.feature_texture_views.end(),views.begin());
            std::copy(renderer.river_rock_texture_views.begin(),renderer.river_rock_texture_views.end(),views.begin()+8);
            std::copy(renderer.bridge_texture_views.begin(),renderer.bridge_texture_views.end(),views.begin()+13);
            auto wonder=renderer.wonder_views();std::copy(wonder.begin(),wonder.end(),views.begin()+21);
            std::copy(renderer.city_base_views.begin(),renderer.city_base_views.end(),views.begin()+29);
            context->PSSetShaderResources(0,33,views.data());return true;
        }
        if (layer>=10000) {"""

PATCHES = {
    "Renderer/native/c3x_renderer.cpp": [
        ("    unsigned char season_stand=0;\n    float city_atlas[4]={};",
         "    unsigned char season_stand=0;\n"
         "    // A natural-wonder body: drawn with the wonder bundle's textures for the frame\n"
         "    // season (RendererState::wonder_views) in slots 116-123.\n"
         "    bool wonder_body=false;\n    float city_atlas[4]={};"),
        ("    c3x_renderer::FeatureBundle resource_bundle;\n    std::array<std::vector<std::uint8_t>, 8> resource_texture_dds;\n"
         "    std::array<ID3D11ShaderResourceView *, 8> resource_texture_views = {};\n", MEMBERS),
        ("    bool resource_assets_ready = false;\n", "    bool resource_assets_ready = false;\n    bool wonder_assets_ready = false;\n"),
        ("        return c3x_renderer::render_core::season_stand_shown(chunk.season_stand,season_index);\n    }", VIEWS),
        ("        for (ID3D11ShaderResourceView *& view : resource_texture_views)\n            release(view);\n",
         "        for (ID3D11ShaderResourceView *& view : resource_texture_views)\n            release(view);\n"
         "        for (ID3D11ShaderResourceView *& view : wonder_texture_views)\n            release(view);\n"
         "        for (ID3D11ShaderResourceView *& view : wonder_season_views)\n            release(view);\n"),
        ("        resource_bundle = {};\n",
         "        resource_bundle = {};\n"
         "        for (std::size_t index = 0; index < wonder_texture_views.size(); ++index) {\n"
         "            release(wonder_texture_views[index]);release(wonder_season_views[index]);\n"
         "            wonder_texture_dds[index].clear();wonder_season_dds[index].clear();\n"
         "        }\n"
         "        wonder_bundle = {};wonder_assets_ready = false;\n"),
        ("        resource_assets_ready = load_runtime_bundle(\n"
         "            resource_root, \"resource_runtime.bin\", resource_bundle, resource_texture_dds, true);\n", LOAD),
        ("        if (resource_assets_ready) {\n"
         "            for (std::size_t index = 0; index < resource_texture_views.size(); ++index)\n"
         "                resource_assets_ready = resource_assets_ready && ensure_dds_texture(\n"
         "                    resource_texture_dds[index], resource_texture_views[index], true);\n        }\n", DEVICE),
        ("        std::vector<Vertex> feature_vertices;\n        std::vector<Vertex> city_vertices;\n",
         "        std::vector<Vertex> feature_vertices;\n"
         "        std::array<std::vector<Vertex>,16> wonder_vertices;  // natural-wonder bodies by season mask\n"
         "        std::vector<Vertex> city_vertices;\n"),
        ("                        item.lift, item.ground_fit);\n                }\n            }\n"
         "            if(!prepared_objects){\n                c3x_renderer::objects::Plan plan;", PLACEMENT),
        ("            // Keep allocation boundaries equal to residency/eviction owners:\n", EMIT),
        ("                if (chunk.content().animation_texture) context->PSSetShaderResources(116,1,&chunk.content().animation_texture);\n",
         "                if (chunk.content().animation_texture) context->PSSetShaderResources(116,1,&chunk.content().animation_texture);\n"
         "                if (chunk.content().wonder_body) context->PSSetShaderResources(116,8,wonder_views().data());\n"),
        ("                if (chunk.content().animation_texture) context->PSSetShaderResources(116,1,resource_texture_views.data());\n            };",
         "                if (chunk.content().animation_texture) context->PSSetShaderResources(116,1,resource_texture_views.data());\n"
         "                if (chunk.content().wonder_body) context->PSSetShaderResources(116,8,resource_texture_views.data());\n            };"),
        ("                    if(chunk.content().city_material!=0xffffffffu)c.binding=10000+chunk.content().city_material;\n",
         "                    if(chunk.content().city_material!=0xffffffffu)c.binding=10000+chunk.content().city_material;\n"
         "                    if(chunk.content().wonder_body)c.binding=20000;\n"),
        ("        auto bind=[&](unsigned layer) {\n            if(layer>=10000){",
         "        auto bind=[&](unsigned layer) {\n" + WONDER_BIND),
        ("        mix(tile.resource_name, sizeof(tile.resource_name));\n        return hash;",
         "        mix(tile.resource_name, sizeof(tile.resource_name));\n"
         "        mix(tile.natural_wonder, sizeof(tile.natural_wonder));\n"
         "        mix(&tile.natural_wonder_flags, sizeof(tile.natural_wonder_flags));\n        return hash;"),
        ("field(tile.resource_id);field(tile.resource_class);mix(tile.resource_name,sizeof(tile.resource_name));",
         "field(tile.resource_id);field(tile.resource_class);mix(tile.resource_name,sizeof(tile.resource_name));\n"
         "        mix(tile.natural_wonder,sizeof(tile.natural_wonder));field(tile.natural_wonder_flags);"),
        ("            std::memcmp(left.resource_name, right.resource_name,\n                        sizeof(left.resource_name)) == 0 &&\n",
         "            std::memcmp(left.resource_name, right.resource_name,\n                        sizeof(left.resource_name)) == 0 &&\n"
         "            std::memcmp(left.natural_wonder, right.natural_wonder, sizeof(left.natural_wonder)) == 0 &&\n"
         "            left.natural_wonder_flags == right.natural_wonder_flags &&\n")],
    "Renderer/sandbox/fresh_pipeline.h": [
        ("            return !m.rigid_source && !m.animation_texture && !m.resource_instance && m.city_material==0xffffffffu &&",
         "            return !m.rigid_source && !m.animation_texture && !m.resource_instance && !m.wonder_body &&\n"
         "                m.city_material==0xffffffffu &&"),
        ("                if(mesh.animation_texture)context->PSSetShaderResources(116,1,&mesh.animation_texture);\n"
         "                if(mesh.resource_instance){",
         "                if(mesh.animation_texture)context->PSSetShaderResources(116,1,&mesh.animation_texture);\n"
         "                if(mesh.wonder_body)context->PSSetShaderResources(116,8,renderer.wonder_views().data());\n"
         "                if(mesh.resource_instance){"),
        ("                if(mesh.animation_texture)context->PSSetShaderResources(116,1,\n"
         "                    renderer.resource_texture_views.data());\n                i=end;",
         "                if(mesh.animation_texture)context->PSSetShaderResources(116,1,\n"
         "                    renderer.resource_texture_views.data());\n"
         "                if(mesh.wonder_body)context->PSSetShaderResources(116,8,renderer.resource_texture_views.data());\n"
         "                i=end;"),
        ("    bool bind_cutout(unsigned layer) {\n        auto* context=renderer.context;\n        if (layer>=10000) {",
         FRESH_WONDER_BIND)],
    "Renderer/native/render_core/ordered_rigid_submission.h": [
        ("        first.animation_texture==next.animation_texture && first.resource_instance==next.resource_instance &&",
         "        first.animation_texture==next.animation_texture && first.resource_instance==next.resource_instance &&\n"
         "        first.wonder_body==next.wonder_body &&")],
}
