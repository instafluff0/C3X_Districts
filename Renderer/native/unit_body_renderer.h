#ifndef C3X_UNIT_BODY_RENDERER_H
#define C3X_UNIT_BODY_RENDERER_H

#include "unit_animation_runtime.h"
#include "navigation_options.h"
#include "unit_shadow.h"
#include "unit_pose_content.h"
#include "environment_refresh/unit_shader.h"

namespace c3x_renderer {

// Worker-owned body rendering. No terrain buffers, cache keys, simulation state
// or native window presentation are owned here. The caller supplies the canvas.
class UnitBodyRenderer {
public:
    struct Mesh { std::shared_ptr<AnimationMesh const> animation; ID3D11Buffer *indices=nullptr; std::string path; std::size_t bytes=0; std::uint64_t used=0; bool failed=false; };
    struct Texture { std::vector<std::uint8_t> dds; ID3D11ShaderResourceView *view=nullptr; std::string path; std::size_t bytes=0; std::uint64_t used=0; bool failed=false; };
    struct Part { unsigned mesh=0,texture=0,address=0; unsigned material_textures[4]={UINT32_MAX,UINT32_MAX,UINT32_MAX,UINT32_MAX}; float material_model=0; float tint[3]={1,1,1}; float mask=0,strength=0,cutout=0; };
    struct Action { std::string name; bool loop=false,ambient=false,allow_exit_clip=false;
        float duration=0;unsigned frames=0;std::vector<Part> parts; };
    struct Unit { std::vector<std::string> keys; float scale=1,yaw_offset=0,offset_z=0; int sample_scale=1,minimum_canvas=0; std::vector<Action> actions; };
    std::vector<Mesh> meshes;
    std::vector<Texture> textures;
    std::vector<Unit> units;
    std::size_t resident_bytes=0;
    std::uint64_t payload_serial=0;
    std::vector<std::uint32_t> pixels;
    int image_width=0,image_height=0;
    bool cache_hit=false;
    char const* failure_reason="none";
    double payload_ms=0,pose_ms=0,submission_ms=0,readback_ms=0,output_ms=0;
    std::size_t cache_bytes=0;
    std::size_t pose_cache_budget=8u*1024u*1024u,pose_cache_entries=128;
    std::size_t cached_pose_entries() const {return cache.size();}
#ifdef C3X_RENDERER_BENCHMARK_ORACLE
    std::size_t benchmark_pose_cache_budget=SIZE_MAX;
    unsigned benchmark_limit_pose_cache() {
        benchmark_pose_cache_budget=256u*1024u*1024u;
        unsigned evicted=0;
        while(!cache.empty() && cache_bytes>benchmark_pose_cache_budget) {
            auto old=std::min_element(cache.begin(),cache.end(),[](Cached const& a,Cached const& b){return a.used<b.used;});
            cache_bytes-=old->pixels.capacity()*4;cache.erase(old);++evicted;
        }
        return evicted;
    }
    void benchmark_reset_pose_limit(){benchmark_pose_cache_budget=SIZE_MAX;}
#endif

    void configure_pose_cache(bool larger,bool dense=false) {
        pose_cache_budget=(larger?(dense?512u:256u):8u)*1024u*1024u;pose_cache_entries=larger?4096u:128u;
#ifdef C3X_RENDERER_BENCHMARK_ORACLE
        pose_cache_budget=(std::min)(pose_cache_budget,benchmark_pose_cache_budget);
#endif
        while(!cache.empty() && (cache_bytes>pose_cache_budget || cache.size()>pose_cache_entries)) {
            auto old=std::min_element(cache.begin(),cache.end(),[](Cached const& a,Cached const& b){return a.used<b.used;});
            cache_bytes-=old->pixels.capacity()*4;cache.erase(old);
        }
    }

    template<class T> void release(T*& p) { if(p) {p->Release();p=nullptr;} }
    void reset_gpu() {
        reset_pose_preparation();
        for(auto & mesh:meshes) release(mesh.indices);
        for(auto & texture:textures) release(texture.view);
        release(vertex);release(pixel);release(layout);release(settings);release(beauty_frame);release(vertices);
        release(shadow_view);release(shadow_texture);
        for(auto & sampler:samplers)release(sampler);
        release(raster);release(target);release(output);release(readback);
        linear.reset();transfer.reset(); capacity=0; image_width=image_height=0;target_width=target_height=0;
        cache.clear();cache_bytes=0;pixels.clear();
    }
    ~UnitBodyRenderer() {reset_gpu();reset_blit();}
    void clear() {reset_gpu();meshes.clear();textures.clear();units.clear();resident_bytes=0;payload_serial=0;}

    // A completed pose is CPU-owned and needs neither the D3D context nor a
    // map-worker takeover. The caller serializes this unit sub-owner while a
    // camera render may continue on the independent terrain/map state.
    bool restore_cached(c3x_renderer_unit_v1 const& request) {
        cache_hit=false;keyed_pixels=cast_pixels=0;failure_reason="cache-miss";
        if(request.struct_size!=sizeof(request) || request.unit_key[63]!=0 ||
           request.hour<0 || request.hour>23 || (request.reduced!=0 && request.reduced!=1) ||
           (request.projection_scale_milli!=0 &&
            (request.projection_scale_milli<250 || request.projection_scale_milli>2000)))return false;
        auto found=std::find_if(units.begin(),units.end(),[&](Unit const& unit){
            return std::find(unit.keys.begin(),unit.keys.end(),request.unit_key)!=unit.keys.end();});
        auto name=native_unit_action(request.action);
        if(found==units.end() || !name)return false;
        auto action=std::find_if(found->actions.begin(),found->actions.end(),[&](Action const& a){return a.name==name;});
        if(action==found->actions.end() || action->parts.empty())return false;
        NativeUnitDraw draw;draw.sprite=draw.expected_sprite=draw.canvas=draw.expected_canvas=1;
        draw.unit_id=request.unit_id;draw.action=request.action;draw.direction=request.direction;
        draw.action_cursor=request.action_cursor;draw.frame_count=request.frame_count;
        draw.body_x=request.body_x;draw.body_y=request.body_y;
        draw.sprite_width=request.sprite_width;draw.sprite_height=request.sprite_height;draw.reduced=request.reduced!=0;
        draw.projection_scale_milli=request.projection_scale_milli;
        UnitAnimationPose pose;
        if(!prepare_native_unit_pose(draw,action->loop,pose))return false;
        int pose_cursor=action->loop?request.action_cursor%request.frame_count:
            std::min(request.action_cursor,request.frame_count-1);
        int scale_milli=request.projection_scale_milli>0?request.projection_scale_milli:(draw.reduced?500:1000);
        int w=request.sprite_width*scale_milli/1000,h=request.sprite_height*scale_milli/1000;
        if(w<1 || h<1 || w>1024 || h>1024)return false;
        Key key={unsigned(found-units.begin()),int(action-found->actions.begin()),request.direction,
            pose_cursor,request.frame_count,w,h,scale_milli,request.hour,request.season,request.display_color_rgb};
        unsigned pose_memory=NavigationOptions::unit_pose_mib(GetEnvironmentVariableA);
        configure_pose_cache(pose_memory>=256,pose_memory==512);
        for(auto& saved:cache)if(saved.key==key) {
            saved.used=++serial;pixels=saved.pixels;image_width=w;image_height=h;
            cache_hit=true;cast_pixels=saved.cast_pixels;failure_reason="none";
            schedule_pose_content(request,*found,*action,pose,key);return true;
        }
        for(auto const& saved:cache)if(saved.key.unit==key.unit && saved.key.action==key.action &&
           saved.key.cursor==key.cursor) {
            if(saved.key.direction!=key.direction)failure_reason="cache-miss-direction";
            else if(saved.key.frames!=key.frames)failure_reason="cache-miss-frame-count";
            else if(saved.key.width!=key.width || saved.key.height!=key.height)failure_reason="cache-miss-size";
            else if(saved.key.scale_milli!=key.scale_milli)failure_reason="cache-miss-scale";
            else if(saved.key.hour!=key.hour)failure_reason="cache-miss-hour";
            else if(saved.key.season!=key.season)failure_reason="cache-miss-season";
            else if(saved.key.color!=key.color)failure_reason="cache-miss-color";
            break;
        }
        return false;
    }

    template<class Prepare>
    bool render(ID3D11Device* device,ID3D11DeviceContext* context,c3x_renderer_unit_v1 const & request,Prepare prepare) {
        payload_ms=pose_ms=submission_ms=readback_ms=output_ms=0;
        LARGE_INTEGER stage_begin={},stage_end={},frequency={};QueryPerformanceFrequency(&frequency);QueryPerformanceCounter(&stage_begin);
        auto elapsed=[&](){QueryPerformanceCounter(&stage_end);double ms=1000.0*(stage_end.QuadPart-stage_begin.QuadPart)/frequency.QuadPart;stage_begin=stage_end;return ms;};
        if(restore_cached(request))return true;
        cache_hit=false;keyed_pixels=cast_pixels=0;failure_reason="invalid-request-or-device";
        if(!device || !context || request.struct_size!=sizeof(request) ||
           request.unit_key[63]!=0 || request.hour<0 || request.hour>23 ||
           (request.reduced!=0 && request.reduced!=1) ||
           (request.projection_scale_milli!=0 &&
            (request.projection_scale_milli<250 || request.projection_scale_milli>2000)))return false;
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
        draw.projection_scale_milli=request.projection_scale_milli;
        UnitAnimationPose pose;
        failure_reason="invalid-native-pose";
        if(!prepare_native_unit_pose(draw,action->loop,pose))return false;
        int pose_cursor=action->loop?request.action_cursor%request.frame_count:
            std::min(request.action_cursor,request.frame_count-1);
        int pose_frames=request.frame_count;
        // Civ III's action director owns whether a unit advances. Ordinary
        // unselected units keep a fixed native cursor; workers, selected units
        // and directed actions advance it through the existing bridge.
        int scale_milli=request.projection_scale_milli>0?request.projection_scale_milli:(draw.reduced?500:1000);
        int w=request.sprite_width*scale_milli/1000,h=request.sprite_height*scale_milli/1000;
        if(w<1 || h<1 || w>1024 || h>1024)return false;
        // Placement and identity are deliberately absent: the same posed body
        // can be reused at a different native anchor or wrapped occurrence.
        Key key={unsigned(found-units.begin()),int(action-found->actions.begin()),request.direction,
            pose_cursor,pose_frames,w,h,scale_milli,request.hour,request.season,request.display_color_rgb};
        unsigned pose_memory=NavigationOptions::unit_pose_mib(GetEnvironmentVariableA);
        configure_pose_cache(pose_memory>=256,pose_memory==512);
        for(auto & saved:cache)if(saved.key==key) {
            saved.used=++serial; pixels=saved.pixels;image_width=w;image_height=h;cache_hit=true;cast_pixels=saved.cast_pixels;failure_reason="none";return true;
        }
        failure_reason="animation-payload-load";
        if(!prepare(*action))return false;
        payload_ms=elapsed();
        failure_reason="gpu-target-setup";
        // Pack-selected material supersampling changes scratch resolution only.
        // Native placement, clipping, readback and cached sprite sizes stay exact.
        int samples=found->sample_scale;
        if((samples!=1 && samples!=2 && samples!=4) || !ensure(device,w,h,samples,found->minimum_canvas?1536:128))return false;
        auto environment=evaluate_environment(float(request.hour),request.season);
        float zoom=pose.projection_scale;
        float clear_color[4]={};context->OMSetRenderTargets(1,&linear.target,linear.depth);
        context->ClearRenderTargetView(linear.target,clear_color);
        context->ClearDepthStencilView(linear.depth,D3D11_CLEAR_DEPTH,1,0);
        context->OMSetBlendState(nullptr,nullptr,0xffffffffu);context->OMSetDepthStencilState(nullptr,0);
        D3D11_VIEWPORT vp={0,0,float(w*samples),float(h*samples),0,1}; context->RSSetViewports(1,&vp);context->RSSetState(raster);
        context->IASetInputLayout(layout);context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(vertex,nullptr,0);context->PSSetShader(pixel,nullptr,0);
        context->PSSetConstantBuffers(0,1,&settings);
        submission_ms+=elapsed();
        auto prepared=prepare_pose_content(request,*found,*action,pose,key);
        if(!prepared){failure_reason="pose-content";return false;}
        auto const& shadow=prepared->shadow;
        // Selected BeautyStudies response, driven by the same native phase as
        // the existing pose-local caster. No independent sun or animation clock.
        auto noon=evaluate_environment(12,0);
        auto key_light=lighting::key_light(environment);
        float const* light_color=key_light.color.data();
        float beauty[20]={};float const ambient_source[]={.34f,.45f,.60f};
        float const chromatic[]={1.f,4.5f/6.2f,3.5f/6.2f};
        auto light=key_light.direction.data();
        for(unsigned axis=0;axis<3;++axis) {
            beauty[axis]=light[axis];beauty[4+axis]=chromatic[axis]*light_color[axis]/std::max(.001f,noon.sun_color[axis]);
            beauty[8+axis]=ambient_source[axis]*environment.ambient_color[axis]/std::max(.001f,noon.ambient_color[axis]);
        }
        beauty[3]=2.05f*(environment.sun_intensity+environment.moon_intensity)/(noon.sun_intensity+noon.moon_intensity);
        beauty[7]=1;beauty[11]=.62f;beauty[12]=.490290f;beauty[13]=-.735435f;beauty[14]=.469979f;
        beauty[16]=float(shadow.extent);
        context->UpdateSubresource(beauty_frame,0,nullptr,beauty,0,0);
        context->PSSetConstantBuffers(1,1,&beauty_frame);context->PSSetSamplers(1,1,&samplers[3]);
        pose_ms+=elapsed();
        context->UpdateSubresource(shadow_texture,0,nullptr,shadow.heights.data(),shadow.extent*4,0);
        context->PSSetShaderResources(1,1,&shadow_view);
        for(std::size_t part_index=0;part_index<action->parts.size();++part_index) {
            auto const& part=action->parts[part_index];
            failure_reason="missing-part-or-texture";
            if(part.mesh>=meshes.size() || part.texture>=textures.size() || !textures[part.texture].view)return false;
            auto & mesh=meshes[part.mesh];
            auto const& upload=prepared->uploads[part_index];
            failure_reason="gpu-geometry-upload";
            UINT bytes=UINT(upload.size()*sizeof(upload[0]));
            if(bytes>capacity) {
                release(vertices);capacity=0;
                D3D11_BUFFER_DESC d={};d.ByteWidth=bytes;d.Usage=D3D11_USAGE_DYNAMIC;
                d.BindFlags=D3D11_BIND_VERTEX_BUFFER;d.CPUAccessFlags=D3D11_CPU_ACCESS_WRITE;
                if(FAILED(device->CreateBuffer(&d,nullptr,&vertices)))return false;capacity=bytes;
            }
            if(!mesh.indices) {
                D3D11_BUFFER_DESC d={};d.ByteWidth=UINT(mesh.animation->indices.size()*4);d.Usage=D3D11_USAGE_IMMUTABLE;
                d.BindFlags=D3D11_BIND_INDEX_BUFFER;D3D11_SUBRESOURCE_DATA data={};data.pSysMem=mesh.animation->indices.data();
                if(FAILED(device->CreateBuffer(&d,&data,&mesh.indices)))return false;
            }
            D3D11_MAPPED_SUBRESOURCE mapped={};
            if(FAILED(context->Map(vertices,0,D3D11_MAP_WRITE_DISCARD,0,&mapped)))return false;
            std::memcpy(mapped.pData,upload.data(),bytes);context->Unmap(vertices,0);
            float values[32]={part.tint[0],part.tint[1],part.tint[2],part.mask};
            for(unsigned a=0;a<3;++a) {
                float color=float((request.display_color_rgb>>(16-a*8))&255)/255;
                values[4+a]=color<=.04045f?color/12.92f:std::pow((color+.055f)/1.055f,2.4f);
                values[8+a]=environment.sun_direction[a];values[12+a]=environment.sun_color[a];
                values[16+a]=environment.moon_direction[a];values[20+a]=environment.moon_color[a];
                values[24+a]=environment.ambient_color[a];
            }
            values[7]=part.strength;values[11]=environment.sun_intensity;values[19]=environment.moon_intensity;values[27]=part.cutout;
            ID3D11ShaderResourceView* extra[4]={};
            for(unsigned channel=0;channel<4;++channel)if(part.material_textures[channel]!=UINT32_MAX) {
                unsigned index=part.material_textures[channel];
                if(index>=textures.size() || !textures[index].view)return false;
                extra[channel]=textures[index].view;values[28+channel]=1;
            }
            context->PSSetShaderResources(2,4,extra);
            values[23]=part.material_model;
            context->UpdateSubresource(settings,0,nullptr,values,0,0);
            UINT stride=68,offset=0;context->IASetVertexBuffers(0,1,&vertices,&stride,&offset);
            context->IASetIndexBuffer(mesh.indices,DXGI_FORMAT_R32_UINT,0);
            context->PSSetShaderResources(0,1,&textures[part.texture].view);
            context->PSSetSamplers(0,1,&samplers[part.address]);
            context->DrawIndexed(UINT(mesh.animation->indices.size()),0,0);
        }
        ID3D11ShaderResourceView* empty[6]={};context->PSSetShaderResources(0,6,empty);
        failure_reason="gpu-body-readback";
        transfer.draw(context,linear,target,environment.exposure,samples);
        context->OMSetRenderTargets(0,nullptr,nullptr);context->CopyResource(readback,output);
        submission_ms+=elapsed();
        D3D11_MAPPED_SUBRESOURCE mapped={};
        if(FAILED(context->Map(readback,0,D3D11_MAP_READ,0,&mapped)))return false;
        readback_ms=elapsed();
        pixels.resize(std::size_t(w)*h);
        cast_pixels=0;
        for(int y=0;y<h;++y)for(int x=0;x<w;++x) {
            auto p=static_cast<std::uint8_t const*>(mapped.pData)+std::size_t(y)*mapped.RowPitch+x*4;
            unsigned alpha=p[3];
            float sx=(float(x)+.5f-float(pose.anchor_x-request.body_x))/(64*zoom);
            float sy=(float(y)+.5f-float(pose.anchor_y-request.body_y))/(32*zoom);
            float fade=std::clamp(float(std::min({x,y,w-1-x,h-1-y}))/3,0.f,1.f);
            unsigned shade=unsigned(255*lighting::c3x_dynamic_shadow_opacity*environment.shadow_strength*fade*shadow.coverage((sx+sy)*.5f,(sy-sx)*.5f));
            unsigned combined=alpha+(shade*(255-alpha)+127)/255;
            if(shade && alpha<255)++cast_pixels;
            pixels[std::size_t(y)*w+x]=(combined<<24)|(((p[2]*alpha+127)/255)<<16)|(((p[1]*alpha+127)/255)<<8)|((p[0]*alpha+127)/255);
        }
        context->Unmap(readback,0);image_width=w;image_height=h;
        output_ms=elapsed();
        // This optional owner retains exact posed pixels across native anchors
        // and repeated authored loops. Admission failure leaves this completed
        // body available for the current draw; it never drops a visible unit.
        try {
            Cached saved={key,++serial,pixels,cast_pixels};
            std::size_t size=saved.pixels.capacity()*4;
            if(size<=pose_cache_budget) {
                while(!cache.empty() && (cache_bytes>pose_cache_budget-size || cache.size()>=pose_cache_entries)) {
                    auto old=std::min_element(cache.begin(),cache.end(),[](Cached const& a,Cached const& b){return a.used<b.used;});
                    cache_bytes-=old->pixels.capacity()*4;cache.erase(old);
                }
                cache.push_back(std::move(saved));cache_bytes+=size;
            }
        }catch(...) {} // Cache growth is optional; current pixels are complete.
        failure_reason="none";return true;
    }

    // The native Animator canvas uses magenta as a color key. Resolve partial
    // coverage against its current pixels, substituting the supplied terrain
    // underlay only at keyed pixels; never blend a fringe with magenta.
    bool blit(HDC destination,int x,int y,HDC background=nullptr) {
        if(!destination || pixels.size()!=std::size_t(image_width)*image_height)return false;
        if(blit_width!=image_width || blit_height!=image_height) {
            reset_blit();BITMAPINFO info={};info.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);
            info.bmiHeader.biWidth=image_width;info.bmiHeader.biHeight=-image_height;
            info.bmiHeader.biPlanes=1;info.bmiHeader.biBitCount=32;info.bmiHeader.biCompression=BI_RGB;
            dc=CreateCompatibleDC(destination);underlay_dc=CreateCompatibleDC(destination);
            if(!dc || !underlay_dc)return false;
            bitmap=CreateDIBSection(destination,&info,DIB_RGB_COLORS,&bits,nullptr,0);
            underlay_bitmap=CreateDIBSection(destination,&info,DIB_RGB_COLORS,&underlay_bits,nullptr,0);
            if(!bitmap || !underlay_bitmap){reset_blit();return false;}
            previous=SelectObject(dc,bitmap);underlay_previous=SelectObject(underlay_dc,underlay_bitmap);
            blit_width=image_width;blit_height=image_height;
        }
        RECT clip={};int clip_type=GetClipBox(destination,&clip);
        if(clip_type==ERROR)return false;if(clip_type==NULLREGION)return true;
        auto target_pixels=static_cast<std::uint32_t*>(bits);
        auto ground=static_cast<std::uint32_t*>(underlay_bits);
        std::fill_n(target_pixels,pixels.size(),0xffff00ffu);
        std::fill_n(ground,pixels.size(),0xffff00ffu);
        if(!BitBlt(dc,0,0,image_width,image_height,destination,x,y,SRCCOPY))return false;
        if(background && !BitBlt(underlay_dc,0,0,image_width,image_height,background,x,y,SRCCOPY))return false;
        GdiFlush();
        keyed_pixels=0;
        auto keyed=[](std::uint32_t value){return (value&0x00f800f8u)==0x00f800f8u && (value&0x0000f800u)==0;};
        for(std::size_t i=0;i<pixels.size();++i) {
            int px=x+int(i%image_width),py=y+int(i/image_width);
            if(px<clip.left || px>=clip.right || py<clip.top || py>=clip.bottom){target_pixels[i]=0x00ff00ffu;continue;}
            auto source=pixels[i];unsigned alpha=source>>24;
            if(!alpha){target_pixels[i]=0x00ff00ffu;continue;}
            auto below=target_pixels[i];
            if(alpha<255 && keyed(below)) {
                if(!background)return false; // Reject atomically; native body remains.
                below=ground[i];++keyed_pixels;
                if(keyed(below)) {target_pixels[i]=0x00ff00ffu;continue;} // outside the native underlay
            }
            std::uint32_t result=0;
            for(unsigned shift:{0u,8u,16u}) {
                unsigned channel=((source>>shift)&255)+((((below>>shift)&255)*(255-alpha)+127)/255);
                result|=std::min(channel,255u)<<shift;
            }
            // An actual body color must not become the native transparent key.
            if(keyed(result))result^=0x00000800u;
            target_pixels[i]=result;
        }
        return TransparentBlt(destination,x,y,image_width,image_height,dc,0,0,
                              image_width,image_height,RGB(255,0,255))!=FALSE;
    }
    unsigned keyed_pixels=0,cast_pixels=0;

private:
    struct Key {
        unsigned unit;int action,direction,cursor,frames,width,height,scale_milli,hour,season;unsigned color;
        bool operator==(Key const& b) const {return unit==b.unit && action==b.action && direction==b.direction &&
            cursor==b.cursor && frames==b.frames && width==b.width && height==b.height && scale_milli==b.scale_milli && hour==b.hour && season==b.season && color==b.color;}
    };
    struct Cached {Key key;std::uint64_t used;std::vector<std::uint32_t> pixels;unsigned cast_pixels;};
    std::vector<Cached> cache;std::uint64_t serial=0;
    using PoseKey=std::array<int,10>;
    using PosePreparation=render_core::ContentPreparation<PoseKey,UnitPoseInput,UnitPoseContent>;
    PosePreparation pose_preparation;
    struct RetainedPose {PoseKey key;std::shared_ptr<UnitPoseContent const> value;};
    std::deque<RetainedPose> retained_poses;
    struct ObservedPose {int id;PoseKey key;};
    std::deque<ObservedPose> observed_poses;
    std::size_t retained_pose_bytes=0;
    bool pose_preparation_configured=false;
    PoseKey content_key(Key const& key) const {
        return {int(key.unit),key.action,key.direction,key.cursor,key.frames,key.width,key.height,key.scale_milli,key.hour,key.season};
    }
    unsigned preparation_workers() const {
        char value[16]={};auto n=GetEnvironmentVariableA("C3X_RENDERER_UNIT_PREPARATION",value,sizeof(value));
        if(!n)return NavigationOptions::enabled(GetEnvironmentVariableA,"C3X_RENDERER_WORLD_REGIONS")?2u:0u;
        return n<sizeof(value)?(std::strcmp(value,"1")==0?1u:std::strcmp(value,"2")==0?2u:std::strcmp(value,"4")==0?4u:0u):0u;
    }
    UnitPoseInput pose_input(c3x_renderer_unit_v1 const& request,Unit const& unit,Action const& action,
                             UnitAnimationPose const& pose,Key const& key) const {
        UnitPoseInput input;input.phase=pose.phase;input.direction=key.direction;input.width=key.width;input.height=key.height;
        input.anchor_x=pose.anchor_x-request.body_x;input.anchor_y=pose.anchor_y-request.body_y;input.zoom=pose.projection_scale;
        auto light=lighting::key_light(evaluate_environment(float(request.hour),request.season));
        input.light_x=light.direction[0];input.light_y=light.direction[1];
        input.source.scale=unit.scale;input.source.yaw_offset=unit.yaw_offset;input.source.offset_z=unit.offset_z;
        input.source.allow_exit_clip=action.allow_exit_clip;input.source.shadow_extent=unit.minimum_canvas?1536:128;
        for(auto const& part:action.parts) {
            if(part.mesh>=meshes.size() || !meshes[part.mesh].animation){input.source.meshes.clear();return input;}
            input.source.meshes.push_back(meshes[part.mesh].animation);
        }
        return input;
    }
    bool preparation_fits(UnitPoseInput const& input) const {
        // Bound each helper's result and intermediate pose/position/tangent
        // scratch before dispatch, independently of the shared asset budget.
        std::size_t bytes=std::size_t(input.source.shadow_extent)*input.source.shadow_extent*4;
        for(auto const& mesh:input.source.meshes)bytes+=mesh->vertices.size()*208;
        return !input.source.meshes.empty() && bytes<=24u*1024u*1024u;
    }
    void schedule_pose_content(c3x_renderer_unit_v1 const& request,Unit const& unit,Action const& action,
                               UnitAnimationPose const& pose,Key const& key) {
        auto workers=preparation_workers();if(!workers)return;
        try {
            MEMORYSTATUSEX memory={};memory.dwLength=sizeof(memory);
            if(!GlobalMemoryStatusEx(&memory) || memory.ullAvailVirtual<(512u+160u+24u*workers)*1024ull*1024ull) {
                reset_pose_preparation();return;
            }
            auto current=content_key(key);
            auto observed=std::find_if(observed_poses.begin(),observed_poses.end(),[&](auto const& item){return item.id==request.unit_id;});
            bool advancing=observed!=observed_poses.end();
            if(advancing && observed->key==current)return; // No autonomous cursor or repeated frozen predictions.
            if(advancing)observed_poses.erase(observed);
            if(observed_poses.size()>=128)observed_poses.pop_front();
            observed_poses.push_back({request.unit_id,current});
            auto input=pose_input(request,unit,action,pose,key);
            if(!preparation_fits(input))return;
            if(!pose_preparation_configured){
                pose_preparation.configure({},UnitPoseCompiler{},workers,{},128u*1024u*1024u);
                pose_preparation_configured=true;
            }
            Key next=key;next.cursor=action.loop?(key.cursor+1)%key.frames:std::min(key.cursor+1,key.frames-1);
            if(next.cursor==key.cursor)return;
            // A single next-pose window fits the active native set. Retaining
            // two distant predictions per unit displaced work due this frame.
            if(std::any_of(cache.begin(),cache.end(),[&](auto const& p){return p.key==next;}))return;
            auto prepared_key=content_key(next);
            if(std::any_of(retained_poses.begin(),retained_poses.end(),[&](auto const& p){return p.key==prepared_key;}))return;
            input.phase=action.loop?double(next.cursor)/key.frames:(key.frames==1?1.0:double(next.cursor)/(key.frames-1));
            pose_preparation.offer({prepared_key,input},32,advancing);
        }catch(...){} // Optional preparation cannot make a native draw fail.
    }
    std::shared_ptr<UnitPoseContent const> prepare_pose_content(c3x_renderer_unit_v1 const& request,Unit const& unit,
                                Action const& action,UnitAnimationPose const& pose,Key const& key) {
        auto wanted=content_key(key);pose_content_hit=false;
        std::shared_ptr<UnitPoseContent const> value;
        if(preparation_workers()) {
            auto found=std::find_if(retained_poses.begin(),retained_poses.end(),[&](auto const& p){return p.key==wanted;});
            if(found!=retained_poses.end()) {
                value=found->value;auto saved=*found;retained_poses.erase(found);retained_poses.push_back(std::move(saved));
            }else value=pose_preparation.take(wanted,true);
            pose_content_hit=bool(value);
        }
        if(!value) {
            std::atomic<bool> cancelled{false};auto input=pose_input(request,unit,action,pose,key);
            if(input.source.meshes.empty())return {};
            value=UnitPoseCompiler{}(input,cancelled,0);
        }
        if(value && preparation_workers()) {
            try {
            constexpr std::size_t budget=32u*1024u*1024u;
            bool existing=std::any_of(retained_poses.begin(),retained_poses.end(),[&](auto const& p){return p.key==wanted;});
            if(!existing && value->bytes()<=budget) {
                while(!retained_poses.empty() && retained_pose_bytes>budget-value->bytes()) {
                    retained_pose_bytes-=retained_poses.front().value->bytes();retained_poses.pop_front();
                }
                retained_poses.push_back({wanted,value});retained_pose_bytes+=value->bytes();
            }
            schedule_pose_content(request,unit,action,pose,key);
            }catch(...){} // Retention remains optional after exact compilation.
        }
        return value;
    }
public:
    bool pose_content_hit=false;
    auto pose_preparation_statistics(){return pose_preparation.statistics();}
    std::size_t pose_retained_bytes() const{return retained_pose_bytes;}
    void release_pose_leases(){pose_preparation.clear();pose_preparation_configured=false;observed_poses.clear();}
    void reset_pose_preparation(){release_pose_leases();retained_poses.clear();retained_pose_bytes=0;}
private:

    ID3D11VertexShader *vertex=nullptr;ID3D11PixelShader *pixel=nullptr;ID3D11InputLayout *layout=nullptr;
    ID3D11Buffer *settings=nullptr,*beauty_frame=nullptr,*vertices=nullptr;UINT capacity=0;
    ID3D11SamplerState *samplers[4]={};ID3D11RasterizerState *raster=nullptr;
    int shadow_size=0;ID3D11Texture2D* shadow_texture=nullptr;ID3D11ShaderResourceView* shadow_view=nullptr;
    render_core::LinearTarget linear;render_core::LinearOutput transfer;
    ID3D11Texture2D *output=nullptr,*readback=nullptr;ID3D11RenderTargetView *target=nullptr;
    int target_width=0,target_height=0;
    HDC dc=nullptr;HBITMAP bitmap=nullptr;HGDIOBJ previous=nullptr;void* bits=nullptr;
    int blit_width=0,blit_height=0;
    HDC underlay_dc=nullptr;HBITMAP underlay_bitmap=nullptr;HGDIOBJ underlay_previous=nullptr;void* underlay_bits=nullptr;
    void reset_blit() {
        if(dc && previous)SelectObject(dc,previous);previous=nullptr;
        if(bitmap)DeleteObject(bitmap);bitmap=nullptr;if(dc)DeleteDC(dc);dc=nullptr;
        if(underlay_dc && underlay_previous)SelectObject(underlay_dc,underlay_previous);
        if(underlay_bitmap)DeleteObject(underlay_bitmap);if(underlay_dc)DeleteDC(underlay_dc);
        underlay_dc=nullptr;underlay_bitmap=nullptr;underlay_previous=nullptr;underlay_bits=nullptr;
        bits=nullptr;blit_width=blit_height=0;
    }
    bool ensure(ID3D11Device* device,int w,int h,int samples,int shadow_extent) {
        if(!pixel) {
            char const* source=unit_material_shader();
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
                {"TEXCOORD",1,DXGI_FORMAT_R32G32B32_FLOAT,0,32,D3D11_INPUT_PER_VERTEX_DATA,0},
                {"TANGENT",0,DXGI_FORMAT_R32G32B32_FLOAT,0,44,D3D11_INPUT_PER_VERTEX_DATA,0},
                {"BINORMAL",0,DXGI_FORMAT_R32G32B32_FLOAT,0,56,D3D11_INPUT_PER_VERTEX_DATA,0}};
            if(SUCCEEDED(hr))hr=device->CreateInputLayout(elements,6,vs->GetBufferPointer(),vs->GetBufferSize(),&layout);
            release(vs);release(ps);
            D3D11_BUFFER_DESC b={};b.ByteWidth=128;b.Usage=D3D11_USAGE_DEFAULT;b.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
            if(SUCCEEDED(hr))hr=device->CreateBuffer(&b,nullptr,&settings);
            b.ByteWidth=80;if(SUCCEEDED(hr))hr=device->CreateBuffer(&b,nullptr,&beauty_frame);
            // The selected unit witness uses MSAA4, anisotropy16 and zero mip
            // bias at native sprite resolution. Address axes are material data.
            for(unsigned mode=0;mode<4 && SUCCEEDED(hr);++mode) {
                D3D11_SAMPLER_DESC s={};s.Filter=D3D11_FILTER_ANISOTROPIC;s.MaxAnisotropy=16;
                s.AddressU=(mode&1)?D3D11_TEXTURE_ADDRESS_CLAMP:D3D11_TEXTURE_ADDRESS_WRAP;
                s.AddressV=(mode&2)?D3D11_TEXTURE_ADDRESS_CLAMP:D3D11_TEXTURE_ADDRESS_WRAP;
                s.AddressW=D3D11_TEXTURE_ADDRESS_WRAP;s.MaxLOD=D3D11_FLOAT32_MAX;
                hr=device->CreateSamplerState(&s,&samplers[mode]);
            }
            D3D11_RASTERIZER_DESC r={};r.FillMode=D3D11_FILL_SOLID;r.CullMode=D3D11_CULL_NONE;r.DepthClipEnable=TRUE;r.MultisampleEnable=TRUE;
            if(SUCCEEDED(hr))hr=device->CreateRasterizerState(&r,&raster);
            if(FAILED(hr)){reset_gpu();return false;}
        }
        if(!shadow_texture || shadow_size!=shadow_extent) {
            release(shadow_view);release(shadow_texture);shadow_size=shadow_extent;
            D3D11_TEXTURE2D_DESC d={};d.Width=d.Height=shadow_extent;d.MipLevels=d.ArraySize=1;
            d.Format=DXGI_FORMAT_R32_FLOAT;d.SampleDesc.Count=1;d.Usage=D3D11_USAGE_DEFAULT;d.BindFlags=D3D11_BIND_SHADER_RESOURCE;
            HRESULT hr=device->CreateTexture2D(&d,nullptr,&shadow_texture);
            if(SUCCEEDED(hr))hr=device->CreateShaderResourceView(shadow_texture,nullptr,&shadow_view);
            if(FAILED(hr)){reset_gpu();return false;}
        }
        if(!linear.ensure(device,UINT(w*samples),UINT(h*samples)) || !transfer.ensure(device))return false;
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
