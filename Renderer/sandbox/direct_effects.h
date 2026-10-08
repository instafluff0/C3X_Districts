#pragma once
// Combat effects in the live scene. The director (render_core/combat_effects.h)
// decides which pack profiles play where; this samples them statelessly
// (effect_sampler.h) and draws camera-facing sprites right after the unit
// bodies, with the unit placement math (anchor, ground lift, depth) so bodies
// and effects occlude each other. No depth or stencil writes: the fog pass
// reads the body stencil. Additive flashes stay emissive; smoke and dust take
// the scene's light. The pack is CombatEffectsRuntime beside the unit pack.
#include "../native/render_core/combat_effects.h"
#ifdef C3X_RENDERER64_FRESH

struct SandboxCombatEffects {
    c3x_renderer::effects::Pack pack;
    std::vector<ID3D11ShaderResourceView*> textures;
    bool attempted=false,ready=false;
    ID3D11VertexShader* vertex=nullptr;ID3D11PixelShader* pixel=nullptr;ID3D11InputLayout* layout=nullptr;
    ID3D11Buffer* vertices=nullptr;unsigned capacity=0;
    ID3D11BlendState* blend=nullptr;ID3D11DepthStencilState* depth=nullptr;ID3D11SamplerState* sampler=nullptr;
    std::vector<c3x_renderer::effects::UnitView> units;
    std::vector<c3x_renderer::effects::Particle> particles;
    struct Vertex{float x,y,z,u,v,r,g,b,a;unsigned mode;};
    struct Sprite{float depth;unsigned texture,alpha;bool additive;Vertex corners[4];};
    std::vector<Sprite> sprites;std::vector<Vertex> batch;
    std::unordered_map<long long,std::vector<unsigned>> tiles;
    unsigned drawn=0,live=0;

    template<class T>static void drop(T*& p){if(p)p->Release();p=nullptr;}
    ~SandboxCombatEffects(){drop(vertex);drop(pixel);drop(layout);drop(vertices);drop(blend);drop(depth);drop(sampler);}

    bool load(){
        if(attempted||renderer.combat_effects_root.empty())return ready;
        attempted=true;
        std::string const& root=renderer.combat_effects_root;
        std::vector<std::uint8_t> bytes;
        if(!renderer.read_file((root+"\\effects.bin").c_str(),bytes)||
           !c3x_renderer::effects::parse_pack(bytes.data(),bytes.size(),pack)){
            renderer.trace.write("combat-effects","pack unavailable; native combat effects remain",true);return false;}
        textures.assign(pack.textures.size(),nullptr);
        for(std::size_t i=0;i<pack.textures.size();++i){
            std::vector<std::uint8_t> dds;char path[4*MAX_PATH];
            if(!renderer.pack_path(root.c_str(),pack.textures[i].path.c_str(),path,std::size(path))||
               !renderer.read_file(path,dds)||!renderer.ensure_dds_texture(dds,textures[i],true,true)){
                renderer.trace.write("combat-effects",("texture failed "+pack.textures[i].path).c_str(),true);return false;}
        }
        char const* source=R"(
struct Input{float3 p:POSITION;float2 uv:TEXCOORD0;float4 c:COLOR0;uint mode:BLENDINDICES;};
struct Output{float4 p:SV_Position;float2 uv:TEXCOORD0;float4 c:COLOR0;nointerpolation uint mode:BLENDINDICES;};
Output VS(Input i){Output o;o.p=float4(i.p,1);o.uv=i.uv;o.c=i.c;o.mode=i.mode;return o;}
Texture2D Colour:register(t0);Texture2D Mask:register(t1);SamplerState Linear:register(s0);
float4 PS(Output i):SV_Target{
 float4 t=Colour.Sample(Linear,i.uv);
 // mode: blend 0 alpha / 1 additive / 2 premultiplied; +4 separate alpha mask.
 float a=(i.mode&4)?Mask.Sample(Linear,i.uv).r:t.a;
 uint blend=i.mode&3;
 float cover=saturate(a*i.c.a);
 if(blend==1)return float4(t.rgb*i.c.rgb*cover,0);
 if(blend==2)return float4(t.rgb*i.c.rgb*i.c.a,cover);
 return float4(t.rgb*i.c.rgb*cover,cover);
})";
        Microsoft::WRL::ComPtr<ID3DBlob> vs,ps,error;
        if(FAILED(D3DCompile(source,strlen(source),"combat_effects",nullptr,nullptr,"VS","vs_5_0",0,0,&vs,&error))||
           FAILED(D3DCompile(source,strlen(source),"combat_effects",nullptr,nullptr,"PS","ps_5_0",0,0,&ps,&error))||
           FAILED(renderer.device->CreateVertexShader(vs->GetBufferPointer(),vs->GetBufferSize(),nullptr,&vertex))||
           FAILED(renderer.device->CreatePixelShader(ps->GetBufferPointer(),ps->GetBufferSize(),nullptr,&pixel)))return false;
        D3D11_INPUT_ELEMENT_DESC elements[]={
            {"POSITION",0,DXGI_FORMAT_R32G32B32_FLOAT,0,0,D3D11_INPUT_PER_VERTEX_DATA,0},
            {"TEXCOORD",0,DXGI_FORMAT_R32G32_FLOAT,0,12,D3D11_INPUT_PER_VERTEX_DATA,0},
            {"COLOR",0,DXGI_FORMAT_R32G32B32A32_FLOAT,0,20,D3D11_INPUT_PER_VERTEX_DATA,0},
            {"BLENDINDICES",0,DXGI_FORMAT_R32_UINT,0,36,D3D11_INPUT_PER_VERTEX_DATA,0}};
        if(FAILED(renderer.device->CreateInputLayout(elements,4,vs->GetBufferPointer(),vs->GetBufferSize(),&layout)))return false;
        D3D11_BLEND_DESC b={};auto& o=b.RenderTarget[0];o.BlendEnable=TRUE;
        o.SrcBlend=o.SrcBlendAlpha=D3D11_BLEND_ONE;o.DestBlend=o.DestBlendAlpha=D3D11_BLEND_INV_SRC_ALPHA;
        o.BlendOp=o.BlendOpAlpha=D3D11_BLEND_OP_ADD;o.RenderTargetWriteMask=D3D11_COLOR_WRITE_ENABLE_ALL;
        D3D11_DEPTH_STENCIL_DESC d={};d.DepthEnable=TRUE;d.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ZERO;d.DepthFunc=D3D11_COMPARISON_LESS_EQUAL;
        D3D11_SAMPLER_DESC s={};s.Filter=D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        s.AddressU=s.AddressV=s.AddressW=D3D11_TEXTURE_ADDRESS_CLAMP;s.MaxLOD=D3D11_FLOAT32_MAX;
        if(FAILED(renderer.device->CreateBlendState(&b,&blend))||FAILED(renderer.device->CreateDepthStencilState(&d,&depth))||
           FAILED(renderer.device->CreateSamplerState(&s,&sampler)))return false;
        ready=true;c3x_renderer::effects::combat_director().enable(true);
        renderer.trace.write("combat-effects",("loaded profiles="+std::to_string(pack.profiles.size())).c_str(),true);
        return true;
    }

    // Tile occurrences of the captured scene (wrap copies included).
    void index_tiles(c3x_renderer_frame_v1 const& frame){
        tiles.clear();
        for(unsigned i=0;i<frame.tile_count;++i)
            tiles[(static_cast<long long>(frame.tiles[i].tile_x)<<32)^static_cast<unsigned>(frame.tiles[i].tile_y)].push_back(i);
    }
    std::vector<unsigned> const* occurrences(int x,int y)const{
        auto found=tiles.find((static_cast<long long>(x)<<32)^static_cast<unsigned>(y));
        return found==tiles.end()?nullptr:&found->second;
    }

    // The director sees the units as drawn this frame (main bodies).
    void update(c3x_renderer_frame_v1 const& frame,SandboxDirectUnits const& source){
        if(!load()||frame.presentation_frequency<=0)return;
        index_tiles(frame);
        units.clear();
        auto const& bodies=renderer.unit_bodies;
        float projection=frame.tile_width/128.f;
        for(auto const& prepared:source.prepared_units){
            if(!prepared.main||prepared.instance.unit>=bodies.units.size())continue;
            auto const* tile=occurrences(prepared.instance.tile_x,prepared.instance.tile_y);
            if(!tile)continue;
            // Ground point relative to the nearest occurrence's centre, in tile units.
            float best=1e30f,sx=0,sy=0;
            for(unsigned i:*tile){auto const& t=frame.tiles[i];
                float dx=float(prepared.pose.anchor_x)-(t.anchor_x+frame.tile_width*.5f);
                float dy=float(prepared.pose.anchor_y)-(t.anchor_y+frame.tile_height*.5f);
                if(dx*dx+dy*dy<best){best=dx*dx+dy*dy;sx=dx;sy=dy;}}
            auto const& unit=bodies.units[prepared.instance.unit];
            c3x_renderer::effects::UnitView view;
            view.unit_id=prepared.draw.unit_id;view.tile_x=prepared.instance.tile_x;view.tile_y=prepared.instance.tile_y;
            view.action=prepared.draw.action;view.phase=float(prepared.pose.phase);view.yaw=prepared.angle;
            view.x=(sx/(64*projection)+sy/(32*projection))*.5f;view.y=(sy/(32*projection)-sx/(64*projection))*.5f;
            view.scale=unit.scale;view.offset_z=unit.offset_z+prepared.lift;view.arms=&unit.arms;
            units.push_back(view);
        }
        auto water=[&](int x,int y){
            auto const* tile=occurrences(x,y);
            return tile&&frame.tiles[tile->front()].real_terrain_type>=11;
        };
        auto& director=c3x_renderer::effects::combat_director();
        director.update(pack,units,frame.presentation_time_ticks,frame.presentation_frequency,water);
        live=unsigned(director.live().size());
    }

    template<class Target>bool draw(c3x_renderer_frame_v1 const& frame,SandboxDirectUnits& source,Target& scene,
                                    float scene_scale,float visual_hour,float zoom){
        drawn=0;
        if(!ready)return true;
        auto const& live_effects=c3x_renderer::effects::combat_director().live();
        if(live_effects.empty())return true;
        // Smoke and dust take the scene's light relative to noon; flashes are emissive.
        auto environment=c3x_renderer::evaluate_environment(visual_hour,frame.season);
        auto noon=c3x_renderer::evaluate_environment(12,frame.season);
        auto level=[](c3x_renderer::EnvironmentState const& e){
            return e.sun_intensity+e.moon_intensity+(.2126f*e.ambient_color[0]+.7152f*e.ambient_color[1]+.0722f*e.ambient_color[2]);};
        float light=std::clamp(level(environment)/std::max(.001f,level(noon)),.08f,1.f);
        bool reduced=zoom<.8f;
        float const pixels_z=150.f*128.f/224.f;
        sprites.clear();
        for(auto const& effect:live_effects){
            auto const* profile=pack.find(effect.profile);
            auto const* tile=occurrences(effect.tile_x,effect.tile_y);
            if(!profile||!tile)continue;
            long long age=(frame.presentation_time_ticks-effect.start)*1000/std::max<long long>(1,effect.frequency);
            if(age<0)continue;
            particles.clear();
            c3x_renderer::effects::sample(*profile,effect.key,age,reduced,particles);
            if(particles.empty())continue;
            float c=std::cos(effect.yaw),s=std::sin(effect.yaw);
            float fx=(c-s)*64,fy=(c+s)*32;float forward=std::atan2(fy,fx);
            for(unsigned occurrence:*tile){
                auto const& t=frame.tiles[occurrence];
                float projection=frame.tile_width/128.f,guard=4.f;
                float cx=t.anchor_x+frame.tile_width*.5f,cy=t.anchor_y+frame.tile_height*.5f;
                float low=source.unit_low_ground(frame,cx,cy),ground_pixels=low*frame.tile_width/224.f*.82f;
                float ground_depth=float(t.tile_y)*frame.tile_height*.5f+renderer.geometry_viewport_settings.depth_translation+frame.tile_height*.5f+4.f;
                float values[8]={(cx+guard)/projection,(cy+guard-ground_pixels)/projection,float(scene.width),float(scene.height),
                    projection*scene_scale,ground_depth+low*.0016f*frame.target_height,0,0};
                c3x_renderer::SceneProjection(frame.target_width,frame.target_height,zoom).unit_placement(values,guard,scene_scale);
                for(auto const& p:particles){
                    auto const& emitter=profile->emitters[p.emitter];
                    // Effect frame -> tile frame -> the unit vertex shader's projection.
                    float x=effect.x+p.position[0]*c-p.position[1]*s,y=effect.y+p.position[0]*s+p.position[1]*c;
                    float z=effect.z+p.position[2];
                    float lx=(x-y)*64,ly=(x+y)*32-z*pixels_z;
                    float w=p.size[0]*128,h=p.size[1]*128;
                    float angle=p.rotation+(p.emitter_oriented?forward:0.f),ca=std::cos(angle),sa=std::sin(angle);
                    lx+=(.5f-p.pivot[0])*w*ca-(.5f-p.pivot[1])*h*sa;ly+=(.5f-p.pivot[0])*w*sa+(.5f-p.pivot[1])*h*ca;
                    float z_buffer=std::clamp(.5f-(values[5]+(x+y)*5+effect.bias*10+z*.1f)/16384.f,.001f,.999f);
                    Sprite sprite;sprite.depth=z_buffer;sprite.texture=emitter.texture;sprite.alpha=emitter.alpha_texture;
                    sprite.additive=emitter.blend==c3x_renderer::effects::Blend::additive;
                    unsigned mode=unsigned(emitter.blend)|(emitter.alpha_texture!=~0u?4u:0u);
                    float lit=sprite.additive?1.f:light;
                    float r=p.tint[0]*p.intensity*lit,g=p.tint[1]*p.intensity*lit,b=p.tint[2]*p.intensity*lit;
                    float corners[4][2]={{-.5f,-.5f},{.5f,-.5f},{-.5f,.5f},{.5f,.5f}};
                    for(int k=0;k<4;++k){
                        float ox=corners[k][0]*w,oy=corners[k][1]*h;
                        float px=(values[0]+lx+ox*ca-oy*sa)*values[4],py=(values[1]+ly+ox*sa+oy*ca)*values[4];
                        sprite.corners[k]={px/values[2]*2-1,1-py/values[3]*2,z_buffer,
                            p.atlas[0]+(corners[k][0]+.5f)*(p.atlas[2]-p.atlas[0]),p.atlas[1]+(corners[k][1]+.5f)*(p.atlas[3]-p.atlas[1]),
                            r,g,b,p.opacity,mode};
                    }
                    sprites.push_back(sprite);
                }
            }
        }
        if(sprites.empty())return true;
        // Back to front (larger depth first); emissive flashes over smoke.
        std::stable_sort(sprites.begin(),sprites.end(),[](Sprite const& a,Sprite const& b){
            return a.additive!=b.additive?!a.additive:a.depth>b.depth;});
        batch.clear();batch.reserve(sprites.size()*6);
        for(auto const& sprite:sprites){auto const* q=sprite.corners;
            for(int k:{0,1,2,2,1,3})batch.push_back(q[k]);}
        unsigned needed=unsigned(batch.size());
        if(needed>capacity){
            drop(vertices);capacity=std::max(needed,1536u);
            D3D11_BUFFER_DESC desc={};desc.ByteWidth=capacity*sizeof(Vertex);desc.Usage=D3D11_USAGE_DYNAMIC;
            desc.BindFlags=D3D11_BIND_VERTEX_BUFFER;desc.CPUAccessFlags=D3D11_CPU_ACCESS_WRITE;
            if(FAILED(renderer.device->CreateBuffer(&desc,nullptr,&vertices))){capacity=0;return false;}
        }
        auto* context=renderer.context;
        D3D11_MAPPED_SUBRESOURCE mapped={};
        if(FAILED(context->Map(vertices,0,D3D11_MAP_WRITE_DISCARD,0,&mapped)))return false;
        std::memcpy(mapped.pData,batch.data(),batch.size()*sizeof(Vertex));context->Unmap(vertices,0);
        context->OMSetRenderTargets(1,&scene.target,scene.depth);
        D3D11_VIEWPORT viewport={0,0,float(scene.width),float(scene.height),0,1};
        c3x_renderer::SceneProjection(frame.target_width,frame.target_height,zoom).viewport(viewport,4.f,0,0,scene_scale);
        D3D11_RECT scissor={0,0,LONG(scene.width),LONG(scene.height)};
        context->RSSetViewports(1,&viewport);context->RSSetScissorRects(1,&scissor);context->RSSetState(renderer.rasterizer_state);
        context->OMSetBlendState(blend,nullptr,0xffffffffu);context->OMSetDepthStencilState(depth,0);
        context->IASetInputLayout(layout);context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        UINT stride=sizeof(Vertex),offset=0;context->IASetVertexBuffers(0,1,&vertices,&stride,&offset);
        context->VSSetShader(vertex,nullptr,0);context->PSSetShader(pixel,nullptr,0);context->PSSetSamplers(0,1,&sampler);
        for(std::size_t first=0;first<sprites.size();){
            std::size_t last=first+1;
            while(last<sprites.size()&&sprites[last].texture==sprites[first].texture&&sprites[last].alpha==sprites[first].alpha)++last;
            ID3D11ShaderResourceView* views[2]={textures[sprites[first].texture],
                sprites[first].alpha!=~0u?textures[sprites[first].alpha]:nullptr};
            context->PSSetShaderResources(0,2,views);
            context->Draw(UINT((last-first)*6),UINT(first*6));drawn+=unsigned(last-first);
            first=last;
        }
        ID3D11ShaderResourceView* empty[2]={};context->PSSetShaderResources(0,2,empty);
        context->OMSetBlendState(nullptr,nullptr,0xffffffffu);context->OMSetDepthStencilState(nullptr,0);
        context->OMSetRenderTargets(0,nullptr,nullptr);
        return true;
    }
};

SandboxCombatEffects sandbox_combat_effects;
#endif
