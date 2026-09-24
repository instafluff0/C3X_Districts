#pragma once

// The synthetic actor's authored mesh, textures, pose compiler and material
// shader are production assets. This owner places their posed vertices in the
// scene HDR/depth target; it never renders a unit-sized color target.
struct SandboxDirectUnits {
    struct Pose {
        int action=0,cursor=0,direction=0,hour=0,season=0;
        std::vector<Microsoft::WRL::ComPtr<ID3D11Buffer>> vertices;
        Microsoft::WRL::ComPtr<ID3D11Texture2D> shadow;
        Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> shadow_view;
        unsigned shadow_extent=0;
    };
    std::vector<Pose> poses;
    ID3D11VertexShader* vertex=nullptr;
    ID3D11PixelShader* pixel=nullptr;
    ID3D11InputLayout* layout=nullptr;
    ID3D11Buffer *material=nullptr,*beauty=nullptr,*placement=nullptr;
    ID3D11SamplerState* samplers[4]={};
    int previous_x=INT_MIN,previous_y=INT_MIN;
    int previous_viewer=INT_MIN,previous_incarnation=INT_MIN;
    int move_from_x=0,move_from_y=0;
    c3x_renderer_i64 move_started=-1;
    unsigned draws=0,pose_builds=0;
    template<class T>static void drop(T*& p){if(p)p->Release();p=nullptr;}
    ~SandboxDirectUnits(){drop(vertex);drop(pixel);drop(layout);drop(material);drop(beauty);
        drop(placement);for(auto& sampler:samplers)drop(sampler);}
    bool initialize(){
        if(pixel)return true;
        char const* source=R"(
cbuffer ScenePlacement:register(b2){float2 origin;float2 extent;float scale;float depth_base;float2 padding;};
struct Input{float3 p:POSITION;float3 n:NORMAL;float2 uv:TEXCOORD0;
 float3 shadow:TEXCOORD1;float3 tangent:TANGENT;float3 bitangent:BINORMAL;};
struct Output{float4 p:SV_Position;float3 n:NORMAL;float2 uv:TEXCOORD0;
 float3 shadow:TEXCOORD1;float3 tangent:TANGENT;float3 bitangent:BINORMAL;};
Output VS(Input i){
 Output o;float2 local=float2((i.p.x+1)*95.5,(1-i.p.y)*95.5);
 float2 pixel=(origin+local)*scale;
 o.p=float4(pixel/extent*float2(2,-2)+float2(-1,1),
  clamp(.5-(depth_base+(0.5-i.p.z)*100)/16384,.001,.999),1);
 o.n=i.n;o.uv=i.uv;o.shadow=i.shadow;
 o.tangent=i.tangent;o.bitangent=i.bitangent;return o;
})";
        ID3DBlob *vs=nullptr,*ps=nullptr,*errors=nullptr;
        HRESULT hr=D3DCompile(source,std::strlen(source),"sandbox_scene_unit",nullptr,nullptr,
            "VS","vs_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&vs,&errors);
        if(errors){if(FAILED(hr))std::printf("SANDBOX_UNIT_SHADER %s\n",
            static_cast<char const*>(errors->GetBufferPointer()));drop(errors);}
        if(SUCCEEDED(hr))hr=D3DCompile(c3x_renderer::unit_material_shader(),
            std::strlen(c3x_renderer::unit_material_shader()),"production_unit_material",
            nullptr,nullptr,"PS","ps_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&ps,&errors);
        if(errors){if(FAILED(hr))std::printf("SANDBOX_UNIT_MATERIAL %s\n",
            static_cast<char const*>(errors->GetBufferPointer()));drop(errors);}
        if(SUCCEEDED(hr))hr=renderer.device->CreateVertexShader(vs->GetBufferPointer(),
            vs->GetBufferSize(),nullptr,&vertex);
        if(SUCCEEDED(hr))hr=renderer.device->CreatePixelShader(ps->GetBufferPointer(),
            ps->GetBufferSize(),nullptr,&pixel);
        D3D11_INPUT_ELEMENT_DESC elements[]={{"POSITION",0,DXGI_FORMAT_R32G32B32_FLOAT,0,0,D3D11_INPUT_PER_VERTEX_DATA,0},
            {"NORMAL",0,DXGI_FORMAT_R32G32B32_FLOAT,0,12,D3D11_INPUT_PER_VERTEX_DATA,0},
            {"TEXCOORD",0,DXGI_FORMAT_R32G32_FLOAT,0,24,D3D11_INPUT_PER_VERTEX_DATA,0},
            {"TEXCOORD",1,DXGI_FORMAT_R32G32B32_FLOAT,0,32,D3D11_INPUT_PER_VERTEX_DATA,0},
            {"TANGENT",0,DXGI_FORMAT_R32G32B32_FLOAT,0,44,D3D11_INPUT_PER_VERTEX_DATA,0},
            {"BINORMAL",0,DXGI_FORMAT_R32G32B32_FLOAT,0,56,D3D11_INPUT_PER_VERTEX_DATA,0}};
        if(SUCCEEDED(hr))hr=renderer.device->CreateInputLayout(elements,6,
            vs->GetBufferPointer(),vs->GetBufferSize(),&layout);
        drop(vs);drop(ps);
        D3D11_BUFFER_DESC b={};b.ByteWidth=128;b.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
        if(SUCCEEDED(hr))hr=renderer.device->CreateBuffer(&b,nullptr,&material);
        b.ByteWidth=80;if(SUCCEEDED(hr))hr=renderer.device->CreateBuffer(&b,nullptr,&beauty);
        b.ByteWidth=32;if(SUCCEEDED(hr))hr=renderer.device->CreateBuffer(&b,nullptr,&placement);
        for(unsigned mode=0;mode<4 && SUCCEEDED(hr);++mode){
            D3D11_SAMPLER_DESC s={};s.Filter=D3D11_FILTER_ANISOTROPIC;s.MaxAnisotropy=16;
            s.AddressU=(mode&1)?D3D11_TEXTURE_ADDRESS_CLAMP:D3D11_TEXTURE_ADDRESS_WRAP;
            s.AddressV=(mode&2)?D3D11_TEXTURE_ADDRESS_CLAMP:D3D11_TEXTURE_ADDRESS_WRAP;
            s.AddressW=D3D11_TEXTURE_ADDRESS_WRAP;s.MaxLOD=D3D11_FLOAT32_MAX;
            hr=renderer.device->CreateSamplerState(&s,&samplers[mode]);
        }
        return SUCCEEDED(hr);
    }
    Pose* pose(c3x_renderer::UnitBodyRenderer::Unit const& unit,
            c3x_renderer::UnitBodyRenderer::Action const& action,
            int action_number,int cursor,int direction,int hour,int season){
        for(auto& saved:poses)if(saved.action==action_number && saved.cursor==cursor &&
            saved.direction==direction && saved.hour==hour && saved.season==season)
            return &saved;
        if(!renderer.prepare_unit_action(action))return nullptr;
        c3x_renderer::NativeUnitDraw draw{};
        draw.sprite=draw.expected_sprite=draw.canvas=draw.expected_canvas=1;
        draw.unit_id=1;draw.action=action_number;draw.direction=direction;
        draw.action_cursor=cursor;draw.frame_count=16;draw.sprite_width=draw.sprite_height=191;
        draw.projection_scale_milli=1000;
        c3x_renderer::UnitAnimationPose phase{};
        if(!c3x_renderer::prepare_native_unit_pose(draw,action.loop,phase))return nullptr;
        c3x_renderer::UnitPoseInput input{};
        input.phase=phase.phase;input.direction=direction;
        input.width=input.height=191;input.anchor_x=phase.anchor_x;
        input.anchor_y=phase.anchor_y;input.zoom=phase.projection_scale;
        auto environment=c3x_renderer::evaluate_environment(float(hour),season);
        auto light=c3x_renderer::lighting::key_light(environment);
        input.light_x=light.direction[0];input.light_y=light.direction[1];
        input.shadow_strength=environment.shadow_strength;
        input.source.scale=unit.scale;input.source.yaw_offset=unit.yaw_offset;
        input.source.offset_z=unit.offset_z;input.source.allow_exit_clip=action.allow_exit_clip;
        input.source.shadow_extent=unit.minimum_canvas?1536:128;
        for(auto const& part:action.parts){
            if(part.mesh>=renderer.unit_bodies.meshes.size() ||
                !renderer.unit_bodies.meshes[part.mesh].animation)return nullptr;
            input.source.meshes.push_back(renderer.unit_bodies.meshes[part.mesh].animation);
        }
        std::atomic<bool> cancelled{false};
        auto content=c3x_renderer::UnitPoseCompiler{}(input,cancelled,0);
        if(!content || content->shadow.heights.empty())return nullptr;
        Pose saved{};saved.action=action_number;saved.cursor=cursor;
        saved.direction=direction;saved.hour=hour;saved.season=season;
        saved.shadow_extent=unsigned(content->shadow.extent);
        D3D11_TEXTURE2D_DESC t={};t.Width=t.Height=saved.shadow_extent;
        t.MipLevels=t.ArraySize=t.SampleDesc.Count=1;t.Format=DXGI_FORMAT_R32_FLOAT;
        t.BindFlags=D3D11_BIND_SHADER_RESOURCE;
        D3D11_SUBRESOURCE_DATA shadow_data={content->shadow.heights.data(),
            saved.shadow_extent*4,0};
        if(FAILED(renderer.device->CreateTexture2D(&t,&shadow_data,&saved.shadow)) ||
            FAILED(renderer.device->CreateShaderResourceView(saved.shadow.Get(),nullptr,
                &saved.shadow_view)))return nullptr;
        for(auto const& upload:content->uploads){
            if(upload.empty())return nullptr;
            D3D11_BUFFER_DESC b={};b.ByteWidth=UINT(upload.size()*sizeof(upload[0]));
            b.Usage=D3D11_USAGE_IMMUTABLE;b.BindFlags=D3D11_BIND_VERTEX_BUFFER;
            D3D11_SUBRESOURCE_DATA data={upload.data(),0,0};
            Microsoft::WRL::ComPtr<ID3D11Buffer> buffer;
            if(FAILED(renderer.device->CreateBuffer(&b,&data,&buffer)))return nullptr;
            saved.vertices.push_back(std::move(buffer));
        }
        if(poses.size()>=48)poses.erase(poses.begin());
        poses.push_back(std::move(saved));++pose_builds;
        return &poses.back();
    }
    bool draw(c3x_renderer_frame_v1 const& frame,int world_x,int world_y,
            int incarnation,int viewer,bool visible,int camera_x,int camera_y,
            c3x_renderer::render_core::LinearTarget& scene,unsigned scene_scale){
        if(!initialize())return false;
        auto& bodies=renderer.unit_bodies;
        auto unit=std::find_if(bodies.units.begin(),bodies.units.end(),[](auto const& item){
            return std::find(item.keys.begin(),item.keys.end(),"PRTO_Warrior")!=item.keys.end();});
        if(unit==bodies.units.end())return false;
        if(incarnation!=previous_incarnation || viewer!=previous_viewer || previous_x==INT_MIN){
            previous_x=world_x;previous_y=world_y;move_started=-1;
        }else if(world_x!=previous_x || world_y!=previous_y){
            move_from_x=previous_x;move_from_y=previous_y;
            previous_x=world_x;previous_y=world_y;move_started=frame.presentation_time_ticks;
        }
        previous_incarnation=incarnation;previous_viewer=viewer;
        float travel=move_started<0?1.f:std::clamp(float(frame.presentation_time_ticks-
            move_started)/float(std::max<c3x_renderer_i64>(1,frame.presentation_frequency)),0.f,1.f);
        if(travel>=1)move_started=-1;
        int positions[4][2]={{world_x,world_y},{21,45},{23,55},{24,56}};
        auto locate=[&](int x,int y){
            c3x_renderer_tile_v1 const* found=nullptr;int best=INT_MAX;
            for(unsigned i=0;i<frame.tile_count;++i){auto const& tile=frame.tiles[i];
                if(tile.tile_y!=y || (tile.tile_x%frame.world_width_tiles+
                    frame.world_width_tiles)%frame.world_width_tiles!=x)continue;
                int distance=std::abs(tile.anchor_x-frame.target_width/2)+
                    std::abs(tile.anchor_y-frame.target_height/2);
                if(distance<best){best=distance;found=&tile;}
            }return found;
        };
        auto* context=renderer.context;
        context->OMSetRenderTargets(1,&scene.target,scene.depth);
        context->OMSetDepthStencilState(renderer.depth_state,0);
        context->OMSetBlendState(nullptr,nullptr,0xffffffffu);
        context->RSSetState(renderer.rasterizer_state);
        D3D11_VIEWPORT viewport={0,0,float(scene.width),float(scene.height),0,1};
        D3D11_RECT scissor={0,0,LONG(scene.width),LONG(scene.height)};
        context->RSSetViewports(1,&viewport);context->RSSetScissorRects(1,&scissor);
        context->IASetInputLayout(layout);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(vertex,nullptr,0);context->PSSetShader(pixel,nullptr,0);
        context->VSSetConstantBuffers(2,1,&placement);
        context->PSSetConstantBuffers(0,1,&material);
        context->PSSetConstantBuffers(1,1,&beauty);
        for(int index=0;index<4;++index){
            if(index==0&&!visible)continue;
            auto* anchor=locate(positions[index][0],positions[index][1]);
            if(!anchor)continue;
            float x=float(anchor->anchor_x),y=float(anchor->anchor_y);
            float world_row=float(anchor->tile_y);
            if(index==0&&move_started>=0)if(auto* from=locate(move_from_x,move_from_y)){
                x=float(from->anchor_x)+(x-float(from->anchor_x))*travel;
                y=float(from->anchor_y)+(y-float(from->anchor_y))*travel;
                world_row=float(from->tile_y)+(world_row-float(from->tile_y))*travel;
            }
            int body_x=int(std::lround(x))+camera_x+frame.tile_width/2-95;
            int body_y=int(std::lround(y))+camera_y+frame.tile_height/2-120;
            if(frame.world_wrap_x && frame.world_width_tiles>0){
                int span=frame.world_width_tiles*frame.tile_width/2;
                while(body_x>frame.target_width+191)body_x-=span;
                while(body_x< -191)body_x+=span;
            }
            if(body_x>frame.target_width || body_x+191<0 ||
                body_y>frame.target_height || body_y+191<0)continue;
            int action_number=index==0&&move_started>=0?2:1;
            int cursor=int((frame.presentation_time_ticks*15/
                std::max<c3x_renderer_i64>(1,frame.presentation_frequency)+index*4)%16);
            auto name=c3x_renderer::native_unit_action(action_number);
            auto action=std::find_if(unit->actions.begin(),unit->actions.end(),
                [&](auto const& item){return item.name==name;});
            if(action==unit->actions.end())return false;
            Pose* prepared=pose(*unit,*action,action_number,cursor,3,frame.hour,frame.season);
            if(!prepared)return false;
            // Resident terrain stores depth relative to its world row. Match
            // that basis instead of using the screen-space sprite rectangle.
            float ground_depth=world_row*frame.tile_height*.5f+
                renderer.geometry_viewport_settings.depth_translation+
                frame.tile_height*.5f+4.f;
            float placement_values[8]={float(body_x+4),float(body_y+4),float(scene.width),
                float(scene.height),float(scene_scale),ground_depth,0,0};
            context->UpdateSubresource(placement,0,nullptr,placement_values,0,0);
            auto environment=c3x_renderer::evaluate_environment(float(frame.hour),frame.season);
            auto noon=c3x_renderer::evaluate_environment(12,0);
            auto key_light=c3x_renderer::lighting::key_light(environment);
            float beauty_values[20]={};float const ambient_source[]={.34f,.45f,.60f};
            float const chromatic[]={1.f,4.5f/6.2f,3.5f/6.2f};
            auto light=key_light.direction.data();auto light_color=key_light.color.data();
            for(unsigned axis=0;axis<3;++axis){
                beauty_values[axis]=light[axis];
                beauty_values[4+axis]=chromatic[axis]*light_color[axis]/
                    std::max(.001f,noon.sun_color[axis]);
                beauty_values[8+axis]=ambient_source[axis]*environment.ambient_color[axis]/
                    std::max(.001f,noon.ambient_color[axis]);
            }
            beauty_values[3]=2.05f*(environment.sun_intensity+environment.moon_intensity)/
                (noon.sun_intensity+noon.moon_intensity);
            beauty_values[7]=1;beauty_values[11]=.62f;
            beauty_values[12]=.490290f;beauty_values[13]=-.735435f;
            beauty_values[14]=.469979f;beauty_values[16]=float(prepared->shadow_extent);
            context->UpdateSubresource(beauty,0,nullptr,beauty_values,0,0);
            ID3D11ShaderResourceView* shadow_view=prepared->shadow_view.Get();
            context->PSSetShaderResources(1,1,&shadow_view);
            context->PSSetSamplers(1,1,&samplers[3]);
            for(unsigned part_index=0;part_index<action->parts.size();++part_index){
                auto const& part=action->parts[part_index];
                if(part.mesh>=bodies.meshes.size() || part.texture>=bodies.textures.size() ||
                    !bodies.textures[part.texture].view)return false;
                auto& mesh=bodies.meshes[part.mesh];
                if(!mesh.indices){
                    D3D11_BUFFER_DESC b={};b.ByteWidth=UINT(mesh.animation->indices.size()*4);
                    b.Usage=D3D11_USAGE_IMMUTABLE;b.BindFlags=D3D11_BIND_INDEX_BUFFER;
                    D3D11_SUBRESOURCE_DATA data={mesh.animation->indices.data(),0,0};
                    if(FAILED(renderer.device->CreateBuffer(&b,&data,&mesh.indices)))return false;
                }
                float values[32]={part.tint[0],part.tint[1],part.tint[2],part.mask};
                for(unsigned axis=0;axis<3;++axis){
                    float color=float((0x205bddu>>(16-axis*8))&255)/255;
                    values[4+axis]=color<=.04045f?color/12.92f:
                        std::pow((color+.055f)/1.055f,2.4f);
                    values[8+axis]=environment.sun_direction[axis];
                    values[12+axis]=environment.sun_color[axis];
                    values[16+axis]=environment.moon_direction[axis];
                    values[20+axis]=environment.moon_color[axis];
                    values[24+axis]=environment.ambient_color[axis];
                }
                values[7]=part.strength;values[11]=environment.sun_intensity;
                values[19]=environment.moon_intensity;values[27]=part.cutout;
                ID3D11ShaderResourceView* extra[4]={};
                for(unsigned channel=0;channel<4;++channel)
                    if(part.material_textures[channel]!=UINT32_MAX){
                        unsigned texture=part.material_textures[channel];
                        if(texture>=bodies.textures.size()||!bodies.textures[texture].view)return false;
                        extra[channel]=bodies.textures[texture].view;values[28+channel]=1;
                    }
                values[23]=part.material_model;
                context->UpdateSubresource(material,0,nullptr,values,0,0);
                ID3D11Buffer* posed=prepared->vertices[part_index].Get();
                UINT stride=68,offset=0;context->IASetVertexBuffers(0,1,&posed,&stride,&offset);
                context->IASetIndexBuffer(mesh.indices,DXGI_FORMAT_R32_UINT,0);
                context->PSSetShaderResources(0,1,&bodies.textures[part.texture].view);
                context->PSSetShaderResources(2,4,extra);
                context->PSSetSamplers(0,1,&samplers[part.address]);
                context->DrawIndexed(UINT(mesh.animation->indices.size()),0,0);
                ++draws;
            }
        }
        ID3D11ShaderResourceView* empty[6]={};context->PSSetShaderResources(0,6,empty);
        context->OMSetRenderTargets(0,nullptr,nullptr);
        return true;
    }
};

SandboxDirectUnits sandbox_direct_units;
