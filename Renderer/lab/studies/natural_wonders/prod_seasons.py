"""Winter for natural wonders (prod_tree.py merges these): marked additions inside the shared
season overlay's Winter edits (Renderer/lab/shared/seasons/seasons.py; the integration lead
reviews), plus the renderer binding the shading reads. Non-wonder pixels keep today's Winter.

- winter/mountain: a wonder's geothermal basin (region kind 2) stays bare.
- winter/objects: wonder bodies stay bare at and below the water plus their kit's snow floor
  (the reef), and foliage takes less snow, as the approved Lab winter did.
- C3XWonderBody (PS b13), declared with the Winter library: bound only while a wonder body
  chunk draws; unbound it reads zero.
"""

BASIN = """        winter_snow = lerp(ground_cover, rock_cover, rock_albedo_coverage);
        // A natural wonder's geothermal basin (region kind 2; docs/natural_wonder_rendering.md) stays bare.
        if (input.volcano_owner.z < 0.001 && abs(input.volcano_owner.w - 2) < 0.5)
            winter_snow *= 1 - saturate(input.volcano_owner.y * 1.6);
"""

BODIES = """            cover *= 0.7 + 0.6 * winter_noise(winter_world * 23.0);
            // Natural-wonder bodies (C3XWonderBody; docs/natural_wonder_rendering.md): bare at and
            // below the water plus their kit's snow floor, and lighter on foliage.
            float wonder_dry = 1;
            if (c3x_wonder_body.x > 0.5) {
                wonder_dry = smoothstep((2.6 + c3x_wonder_body.y) / 112.0, (6.0 + c3x_wonder_body.y) / 112.0, input.q6_world.z);
                cover *= wonder_dry * lerp(1.0, 0.45, saturate((albedo.g - albedo.r) * 25 + 0.75));
            }
            albedo = lerp(lerp(albedo, winter_dormant(albedo), WINTER_OBJ_DORMANT * wonder_dry), winter_white, saturate(cover));"""

DECLARATION = """    before("winter/library", INTEGRATED, RAW_FEATURE, library("winter_integrated.hlsl")),
    # Natural-wonder bodies: the renderer binds this only while a wonder body draws (x 1, y its
    # kit's snow floor in relief units); unbound it reads zero (docs/natural_wonder_rendering.md).
    before("winter/library", FIELD_SET, RAW_FEATURE, "cbuffer C3XWonderBody : register(b13) { float4 c3x_wonder_body; };\\n"),
"""

BIND = """    // Natural-wonder bodies: the C3XWonderBody constants (PS b13) Winter shading reads,
    // bound only for their draws (x 1, y the kit's snow floor in relief units).
    void bind_wonder_body(ID3D11DeviceContext* c,float snow_floor){
        if(!wonder_body_buffer){D3D11_BUFFER_DESC d={};d.ByteWidth=16;d.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
            if(FAILED(device->CreateBuffer(&d,nullptr,&wonder_body_buffer)))return;}
        float values[4]={1.f,snow_floor,0.f,0.f};c->UpdateSubresource(wonder_body_buffer,0,nullptr,values,0,0);
        c->PSSetConstantBuffers(13,1,&wonder_body_buffer);
    }
    static void unbind_wonder_body(ID3D11DeviceContext* c){ID3D11Buffer* none=nullptr;c->PSSetConstantBuffers(13,1,&none);}
    // Seasons: whether a chunk draws in the current season.
"""

PATCHES = {
    "Renderer/lab/shared/seasons/seasons.py": [
        ("        winter_snow = lerp(ground_cover, rock_cover, rock_albedo_coverage);\n", BASIN),
        ("            cover *= 0.7 + 0.6 * winter_noise(winter_world * 23.0);\n"
         "            albedo = lerp(lerp(albedo, winter_dormant(albedo), WINTER_OBJ_DORMANT), winter_white, saturate(cover));", BODIES),
        ("    before(\"winter/library\", INTEGRATED, RAW_FEATURE, library(\"winter_integrated.hlsl\")),\n", DECLARATION)],
    "Renderer/native/c3x_renderer.cpp": [
        ("    ID3D11Buffer* water_frame=nullptr;\n",
         "    ID3D11Buffer* water_frame=nullptr;\n    ID3D11Buffer* wonder_body_buffer=nullptr;\n"),
        ("release(wave_frame);release(water_frame);", "release(wave_frame);release(water_frame);release(wonder_body_buffer);"),
        ("    // Seasons: whether a chunk draws in the current season.\n", BIND),
        ("    bool wonder_body=false;\n", "    bool wonder_body=false;float wonder_snow_floor=0;\n"),
        ("        std::array<std::vector<Vertex>,16> wonder_vertices;  // natural-wonder bodies by season mask\n",
         "        std::array<std::vector<Vertex>,16> wonder_vertices;  // natural-wonder bodies by season mask\n"
         "        float wonder_snow_floor=0;                          // their kit's Winter snow floor\n"),
        ("                    natural.wonder_pack.find(tile.natural_wonder) : nullptr) {\n",
         "                    natural.wonder_pack.find(tile.natural_wonder) : nullptr) {\n"
         "                wonder_snow_floor = kit->snow_floor;\n"),
        ("                    chunk.wonder_body=true;", "                    chunk.wonder_body=true;chunk.wonder_snow_floor=wonder_snow_floor;"),
        ("                if (chunk.content().wonder_body) context->PSSetShaderResources(116,8,wonder_views().data());\n",
         "                if (chunk.content().wonder_body){context->PSSetShaderResources(116,8,wonder_views().data());\n"
         "                    bind_wonder_body(context,chunk.content().wonder_snow_floor);}\n"),
        ("                if (chunk.content().wonder_body) context->PSSetShaderResources(116,8,resource_texture_views.data());\n",
         "                if (chunk.content().wonder_body){context->PSSetShaderResources(116,8,resource_texture_views.data());\n"
         "                    unbind_wonder_body(context);}\n")],
    "Renderer/sandbox/fresh_pipeline.h": [
        ("                if(mesh.wonder_body)context->PSSetShaderResources(116,8,renderer.wonder_views().data());\n",
         "                if(mesh.wonder_body){context->PSSetShaderResources(116,8,renderer.wonder_views().data());\n"
         "                    renderer.bind_wonder_body(context,mesh.wonder_snow_floor);}\n"),
        ("                if(mesh.wonder_body)context->PSSetShaderResources(116,8,renderer.resource_texture_views.data());\n",
         "                if(mesh.wonder_body){context->PSSetShaderResources(116,8,renderer.resource_texture_views.data());\n"
         "                    renderer.unbind_wonder_body(context);}\n")],
}
