#ifndef C3X_UNIT_BODY_RENDERER_H
#define C3X_UNIT_BODY_RENDERER_H

#include "unit_animation_runtime.h"

namespace c3x_renderer {

// Worker-owned body rendering. No terrain buffers, cache keys, simulation state
// or native window presentation are owned here. The caller supplies the canvas.
class UnitBodyRenderer {
public:
    struct Mesh { AnimationMesh animation; ID3D11Buffer *indices=nullptr; };
    struct Texture { std::vector<std::uint8_t> dds; ID3D11ShaderResourceView *view=nullptr; };
    struct Part { unsigned mesh=0,texture=0; float tint[3]={1,1,1}; float mask=0,strength=0,cutout=0; };
    struct Action { std::string name; bool loop=false; std::vector<Part> parts; };
    struct Unit { std::vector<std::string> keys; float scale=1,yaw_offset=0,offset_z=0; std::vector<Action> actions; };
    std::vector<Mesh> meshes;
    std::vector<Texture> textures;
    std::vector<Unit> units;
    std::vector<std::uint32_t> pixels;
    int image_width=0,image_height=0;
    bool cache_hit=false;
    char const* failure_reason="none";
    std::size_t cache_bytes=0;

    template<class T> void release(T*& p) { if(p) {p->Release();p=nullptr;} }
    void reset_gpu() {
        for(auto & mesh:meshes) release(mesh.indices);
        for(auto & texture:textures) release(texture.view);
        release(vertex);release(pixel);release(layout);release(settings);release(vertices);
        release(sampler);release(raster);release(target);release(output);release(readback);
        linear.reset();transfer.reset(); capacity=0; image_width=image_height=0;target_width=target_height=0;
        cache.clear();cache_bytes=0;pixels.clear();
    }
    ~UnitBodyRenderer() {reset_gpu();reset_blit();}
    void clear() {reset_gpu();meshes.clear();textures.clear();units.clear();}

    bool render(ID3D11Device* device,ID3D11DeviceContext* context,c3x_renderer_unit_v1 const & request) {
        cache_hit=false;failure_reason="invalid-request-or-device";
        if(!device || !context || request.struct_size!=sizeof(request) ||
           request.unit_key[63]!=0 || request.hour<0 || request.hour>23 ||
           (request.reduced!=0 && request.reduced!=1))return false;
        auto found=std::find_if(units.begin(),units.end(),[&](Unit const& unit){
            return std::find(unit.keys.begin(),unit.keys.end(),request.unit_key)!=unit.keys.end();});
        auto name=native_unit_action(request.action);
        failure_reason="unmapped-unit-or-action";
        if(found==units.end() || !name)return false;
        auto action=std::find_if(found->actions.begin(),found->actions.end(),[&](Action const& a){return a.name==name;});
        if(action==found->actions.end() || action->parts.empty())return false;
        NativeUnitDraw draw;
        draw.sprite=draw.expected_sprite=draw.canvas=draw.expected_canvas=1; // identity guard belongs to the bridge
        draw.unit_id=request.unit_id;draw.action=request.action;draw.direction=request.direction;
        draw.action_cursor=request.action_cursor;draw.frame_count=request.frame_count;
        draw.body_x=request.body_x;draw.body_y=request.body_y;
        draw.sprite_width=request.sprite_width;draw.sprite_height=request.sprite_height;draw.reduced=request.reduced!=0;
        UnitAnimationPose pose;
        failure_reason="invalid-native-pose";
        if(!prepare_native_unit_pose(draw,action->loop,pose))return false;
        int w=request.sprite_width/(draw.reduced?2:1),h=request.sprite_height/(draw.reduced?2:1);
        if(w<1 || h<1 || w>512 || h>512)return false;
        // Placement and identity are deliberately absent: the same posed body
        // can be reused at a different native anchor or wrapped occurrence.
        Key key={unsigned(found-units.begin()),request.action,request.direction,
            action->loop?request.action_cursor%request.frame_count:std::min(request.action_cursor,request.frame_count-1),
            request.frame_count,w,h,request.reduced,request.hour,request.season,request.display_color_rgb};
        for(auto & saved:cache)if(saved.key==key) {
            saved.used=++serial; pixels=saved.pixels;image_width=w;image_height=h;cache_hit=true;failure_reason="none";return true;
        }
        failure_reason="gpu-target-setup";
        if(!ensure(device,w,h))return false;
        auto environment=evaluate_environment(float(request.hour),request.season);
        float cosine=std::cos((found->yaw_offset+float(request.direction%8)*45)*.01745329252f);
        float sine=std::sin((found->yaw_offset+float(request.direction%8)*45)*.01745329252f);
        float zoom=pose.projection_scale,scale=found->scale;
        float clear_color[4]={};context->OMSetRenderTargets(1,&linear.target,linear.depth);
        context->ClearRenderTargetView(linear.target,clear_color);
        context->ClearDepthStencilView(linear.depth,D3D11_CLEAR_DEPTH,1,0);
        context->OMSetBlendState(nullptr,nullptr,0xffffffffu);context->OMSetDepthStencilState(nullptr,0);
        D3D11_VIEWPORT vp={0,0,float(w),float(h),0,1}; context->RSSetViewports(1,&vp);context->RSSetState(raster);
        context->IASetInputLayout(layout);context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(vertex,nullptr,0);context->PSSetShader(pixel,nullptr,0);
        context->PSSetSamplers(0,1,&sampler);context->PSSetConstantBuffers(0,1,&settings);
        std::vector<FeatureSourceVertex> posed;
        std::vector<std::array<float,9>> upload;
        for(auto const& part:action->parts) {
            failure_reason="missing-part-or-texture";
            if(part.mesh>=meshes.size() || part.texture>=textures.size() || !textures[part.texture].view)return false;
            auto & mesh=meshes[part.mesh];
            failure_reason="pose-sampling";
            if(!sample_animation_mesh(mesh.animation,pose.phase*mesh.animation.duration,false,posed))return false;
            upload.resize(posed.size());
            for(std::size_t i=0;i<posed.size();++i) {
                auto const& p=posed[i];
                float x=(p.position[0]*cosine-p.position[1]*sine)*scale;
                float y=(p.position[0]*sine+p.position[1]*cosine)*scale,z=(p.position[2]+found->offset_z)*scale;
                float sx=float(pose.anchor_x-request.body_x)+(x-y)*64*zoom;
                float sy=float(pose.anchor_y-request.body_y)+(x+y)*32*zoom-z*(150.f*128/224)*zoom;
                upload[i]={2*sx/w-1,1-2*sy/h,.5f-(x+y)*.05f-z*.001f,
                    p.normal[0]*cosine-p.normal[1]*sine,p.normal[0]*sine+p.normal[1]*cosine,p.normal[2],p.uv[0],p.uv[1],z};
            }
            // The ground plane hides buried anatomy/stowed equipment. Bound
            // the visible polygon, including intersections of crossing edges,
            // so clipping never extends beyond Civ III's native dirty region.
            failure_reason="visible-body-outside-native-sprite";
            auto inside=[&](float x,float y){return x>=2.f/w-1 && x<=1-2.f/w && y>=2.f/h-1 && y<=1-2.f/h;};
            for(std::size_t i=0;i<mesh.animation.indices.size();i+=3) {
                for(unsigned edge=0;edge<3;++edge) {
                    auto const& a=upload[mesh.animation.indices[i+edge]];
                    auto const& b=upload[mesh.animation.indices[i+(edge+1)%3]];
                    if(a[8]>=0 && !inside(a[0],a[1]))return false;
                    if((a[8]<0)!=(b[8]<0)) {
                        float t=a[8]/(a[8]-b[8]);
                        if(!inside(a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1])))return false;
                    }
                }
            }
            failure_reason="gpu-geometry-upload";
            UINT bytes=UINT(upload.size()*sizeof(upload[0]));
            if(bytes>capacity) {
                release(vertices);capacity=0;
                D3D11_BUFFER_DESC d={};d.ByteWidth=bytes;d.Usage=D3D11_USAGE_DYNAMIC;
                d.BindFlags=D3D11_BIND_VERTEX_BUFFER;d.CPUAccessFlags=D3D11_CPU_ACCESS_WRITE;
                if(FAILED(device->CreateBuffer(&d,nullptr,&vertices)))return false;capacity=bytes;
            }
            if(!mesh.indices) {
                D3D11_BUFFER_DESC d={};d.ByteWidth=UINT(mesh.animation.indices.size()*4);d.Usage=D3D11_USAGE_IMMUTABLE;
                d.BindFlags=D3D11_BIND_INDEX_BUFFER;D3D11_SUBRESOURCE_DATA data={};data.pSysMem=mesh.animation.indices.data();
                if(FAILED(device->CreateBuffer(&d,&data,&mesh.indices)))return false;
            }
            D3D11_MAPPED_SUBRESOURCE mapped={};
            if(FAILED(context->Map(vertices,0,D3D11_MAP_WRITE_DISCARD,0,&mapped)))return false;
            std::memcpy(mapped.pData,upload.data(),bytes);context->Unmap(vertices,0);
            float values[28]={part.tint[0],part.tint[1],part.tint[2],part.mask};
            for(unsigned a=0;a<3;++a) {
                float color=float((request.display_color_rgb>>(16-a*8))&255)/255;
                values[4+a]=color<=.04045f?color/12.92f:std::pow((color+.055f)/1.055f,2.4f);
                values[8+a]=environment.sun_direction[a];values[12+a]=environment.sun_color[a];
                values[16+a]=environment.moon_direction[a];values[20+a]=environment.moon_color[a];
                values[24+a]=environment.ambient_color[a];
            }
            values[7]=part.strength;values[11]=environment.sun_intensity;values[19]=environment.moon_intensity;values[27]=part.cutout;
            context->UpdateSubresource(settings,0,nullptr,values,0,0);
            UINT stride=36,offset=0;context->IASetVertexBuffers(0,1,&vertices,&stride,&offset);
            context->IASetIndexBuffer(mesh.indices,DXGI_FORMAT_R32_UINT,0);
            context->PSSetShaderResources(0,1,&textures[part.texture].view);
            context->DrawIndexed(UINT(mesh.animation.indices.size()),0,0);
        }
        ID3D11ShaderResourceView* empty=nullptr;context->PSSetShaderResources(0,1,&empty);
        failure_reason="gpu-body-readback";
        transfer.draw(context,linear,target,environment.exposure);
        context->OMSetRenderTargets(0,nullptr,nullptr);context->CopyResource(readback,output);
        D3D11_MAPPED_SUBRESOURCE mapped={};
        if(FAILED(context->Map(readback,0,D3D11_MAP_READ,0,&mapped)))return false;
        pixels.resize(std::size_t(w)*h);
        for(int y=0;y<h;++y)for(int x=0;x<w;++x) {
            auto p=static_cast<std::uint8_t const*>(mapped.pData)+std::size_t(y)*mapped.RowPitch+x*4;
            unsigned alpha=p[3];
            pixels[std::size_t(y)*w+x]=(alpha<<24)|(((p[2]*alpha+127)/255)<<16)|(((p[1]*alpha+127)/255)<<8)|((p[0]*alpha+127)/255);
        }
        context->Unmap(readback,0);image_width=w;image_height=h;
        std::size_t size=pixels.size()*4;
        while(!cache.empty() && (cache_bytes+size>8u*1024u*1024u || cache.size()>=128)) {
            auto old=std::min_element(cache.begin(),cache.end(),[](Cached const& a,Cached const& b){return a.used<b.used;});
            cache_bytes-=old->pixels.size()*4;cache.erase(old);
        }
        cache.push_back({key,++serial,pixels});cache_bytes+=size;
        failure_reason="none";return true;
    }

    // Runs on the game's calling thread, after the complete body succeeds.
    bool blit(HDC destination,int x,int y) {
        if(!destination || pixels.size()!=std::size_t(image_width)*image_height)return false;
        if(blit_width!=image_width || blit_height!=image_height) {
            reset_blit();BITMAPINFO info={};info.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);
            info.bmiHeader.biWidth=image_width;info.bmiHeader.biHeight=-image_height;
            info.bmiHeader.biPlanes=1;info.bmiHeader.biBitCount=32;info.bmiHeader.biCompression=BI_RGB;
            dc=CreateCompatibleDC(destination);if(!dc)return false;
            bitmap=CreateDIBSection(destination,&info,DIB_RGB_COLORS,&bits,nullptr,0);
            if(!bitmap){reset_blit();return false;}previous=SelectObject(dc,bitmap);
            blit_width=image_width;blit_height=image_height;
        }
        std::memcpy(bits,pixels.data(),pixels.size()*4);
        BLENDFUNCTION blend={AC_SRC_OVER,0,255,AC_SRC_ALPHA};
        return AlphaBlend(destination,x,y,image_width,image_height,dc,0,0,image_width,image_height,blend)!=FALSE;
    }

private:
    struct Key {
        unsigned unit;int action,direction,cursor,frames,width,height,reduced,hour,season;unsigned color;
        bool operator==(Key const& b) const {return unit==b.unit && action==b.action && direction==b.direction &&
            cursor==b.cursor && frames==b.frames && width==b.width && height==b.height && reduced==b.reduced && hour==b.hour && season==b.season && color==b.color;}
    };
    struct Cached {Key key;std::uint64_t used;std::vector<std::uint32_t> pixels;};
    std::vector<Cached> cache;std::uint64_t serial=0;
    ID3D11VertexShader *vertex=nullptr;ID3D11PixelShader *pixel=nullptr;ID3D11InputLayout *layout=nullptr;
    ID3D11Buffer *settings=nullptr,*vertices=nullptr;UINT capacity=0;
    ID3D11SamplerState *sampler=nullptr;ID3D11RasterizerState *raster=nullptr;
    profile_v2::LinearTarget linear;profile_v2::LinearOutput transfer;
    ID3D11Texture2D *output=nullptr,*readback=nullptr;ID3D11RenderTargetView *target=nullptr;
    int target_width=0,target_height=0;
    HDC dc=nullptr;HBITMAP bitmap=nullptr;HGDIOBJ previous=nullptr;void* bits=nullptr;
    int blit_width=0,blit_height=0;
    void reset_blit() {
        if(dc && previous)SelectObject(dc,previous);previous=nullptr;
        if(bitmap)DeleteObject(bitmap);bitmap=nullptr;if(dc)DeleteDC(dc);dc=nullptr;
        bits=nullptr;blit_width=blit_height=0;
    }
    bool ensure(ID3D11Device* device,int w,int h) {
        if(!pixel) {
            char const* source=R"(
Texture2D<float4> base : register(t0);SamplerState sample_base : register(s0);
cbuffer Material : register(b0) {float4 tint,owner,sun,sun_color,moon,moon_color,ambient;};
struct Input {float3 p:POSITION;float3 n:NORMAL;float2 uv:TEXCOORD0;float ground:TEXCOORD1;};
struct Output {float4 p:SV_Position;float3 n:NORMAL;float2 uv:TEXCOORD0;float ground:TEXCOORD1;};
Output VS(Input i){Output o;o.p=float4(i.p,1);o.n=i.n;o.uv=i.uv;o.ground=i.ground;return o;}
float4 PS(Output i):SV_Target {
 clip(i.ground);
 float4 b=base.Sample(sample_base,i.uv);if(ambient.w>.5)clip(b.a-.5);
 float3 albedo=b.rgb*tint.rgb;
 float mask=tint.w<.5?0:(tint.w<1.5?smoothstep(.06,.94,1-b.a):1);
 float value=dot(albedo,float3(.2126,.7152,.0722));
 float3 ramp=lerp(owner.rgb*.32,saturate(owner.rgb*.90+float3(.24,.24,.20)),smoothstep(.08,.86,value));
 albedo=lerp(albedo,lerp(ramp,albedo,.14),mask*owner.w);
 float3 n=normalize(i.n);
 float3 light=ambient.rgb+sun_color.rgb*sun.w*saturate(dot(n,sun.xyz))+moon_color.rgb*moon.w*saturate(dot(n,moon.xyz));
 return float4(albedo*max(light,.02),1);
})";
            ID3DBlob *vs=nullptr,*ps=nullptr,*error=nullptr;
            HRESULT hr=D3DCompile(source,std::strlen(source),"unit_body",nullptr,nullptr,"VS","vs_4_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&vs,&error);
            if(error){OutputDebugStringA(static_cast<char const*>(error->GetBufferPointer()));release(error);}
            if(SUCCEEDED(hr))hr=D3DCompile(source,std::strlen(source),"unit_body",nullptr,nullptr,"PS","ps_4_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&ps,&error);
            if(error){OutputDebugStringA(static_cast<char const*>(error->GetBufferPointer()));release(error);}
            if(SUCCEEDED(hr))hr=device->CreateVertexShader(vs->GetBufferPointer(),vs->GetBufferSize(),nullptr,&vertex);
            if(SUCCEEDED(hr))hr=device->CreatePixelShader(ps->GetBufferPointer(),ps->GetBufferSize(),nullptr,&pixel);
            D3D11_INPUT_ELEMENT_DESC elements[]={{"POSITION",0,DXGI_FORMAT_R32G32B32_FLOAT,0,0,D3D11_INPUT_PER_VERTEX_DATA,0},
                {"NORMAL",0,DXGI_FORMAT_R32G32B32_FLOAT,0,12,D3D11_INPUT_PER_VERTEX_DATA,0},
                {"TEXCOORD",0,DXGI_FORMAT_R32G32_FLOAT,0,24,D3D11_INPUT_PER_VERTEX_DATA,0},
                {"TEXCOORD",1,DXGI_FORMAT_R32_FLOAT,0,32,D3D11_INPUT_PER_VERTEX_DATA,0}};
            if(SUCCEEDED(hr))hr=device->CreateInputLayout(elements,4,vs->GetBufferPointer(),vs->GetBufferSize(),&layout);
            release(vs);release(ps);
            D3D11_BUFFER_DESC b={};b.ByteWidth=112;b.Usage=D3D11_USAGE_DEFAULT;b.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
            if(SUCCEEDED(hr))hr=device->CreateBuffer(&b,nullptr,&settings);
            D3D11_SAMPLER_DESC s={};s.Filter=D3D11_FILTER_MIN_MAG_MIP_LINEAR;s.AddressU=s.AddressV=s.AddressW=D3D11_TEXTURE_ADDRESS_WRAP;s.MaxLOD=D3D11_FLOAT32_MAX;
            if(SUCCEEDED(hr))hr=device->CreateSamplerState(&s,&sampler);
            D3D11_RASTERIZER_DESC r={};r.FillMode=D3D11_FILL_SOLID;r.CullMode=D3D11_CULL_NONE;r.DepthClipEnable=TRUE;r.MultisampleEnable=TRUE;
            if(SUCCEEDED(hr))hr=device->CreateRasterizerState(&r,&raster);
            if(FAILED(hr)){reset_gpu();return false;}
        }
        if(!linear.ensure(device,UINT(w),UINT(h)) || !transfer.ensure(device))return false;
        if(!target || target_width!=w || target_height!=h) {
            release(target);release(output);release(readback);
            D3D11_TEXTURE2D_DESC d={};d.Width=UINT(w);d.Height=UINT(h);d.MipLevels=d.ArraySize=1;
            d.Format=DXGI_FORMAT_B8G8R8A8_UNORM;d.SampleDesc.Count=1;d.Usage=D3D11_USAGE_DEFAULT;d.BindFlags=D3D11_BIND_RENDER_TARGET;
            HRESULT hr=device->CreateTexture2D(&d,nullptr,&output);
            if(SUCCEEDED(hr))hr=device->CreateRenderTargetView(output,nullptr,&target);
            d.Usage=D3D11_USAGE_STAGING;d.BindFlags=0;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
            if(SUCCEEDED(hr))hr=device->CreateTexture2D(&d,nullptr,&readback);
            if(FAILED(hr)){release(target);return false;}target_width=w;target_height=h;
        }
        return true;
    }
};
} // namespace c3x_renderer
#endif
