#pragma once
#include "render_core/captured_scene.h"
#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <vector>
#include <memory>

namespace c3x_renderer {
// One finished area and its complete captured-content proof. Publication owns
// immutable CPU or GPU storage; selection and validity are identical for both.
template<class Publication> struct PreparedViewArea {
    Publication map,center;
    c3x_renderer_frame_v1 input={};
    c3x_renderer_camera_identity_v1 identity={};
    std::vector<c3x_renderer_tile_v1> tiles,source_tiles;
    std::vector<c3x_renderer_u32> topology;
    std::unordered_map<std::uint64_t,unsigned> index;
    std::vector<std::uint64_t> dependencies;
    int viewport_width=0,viewport_height=0,pad_x=0,pad_y=0;
    unsigned detail_class=0;
    static unsigned detail(unsigned count){return count<=512?0:count<=768?1:count<=2048?2:3;}
    static constexpr std::size_t budget=32u*1024u*1024u;
    std::uint64_t key(c3x_renderer_tile_v1 const& t) const {
        auto canonical=[](int x,int n,bool wraps){int r=wraps&&n>0?x%n:x;return r<0&&wraps?r+n:r;};
        return (std::uint64_t(std::uint32_t(canonical(t.tile_x,input.world_width_tiles,input.world_wrap_x!=0)))<<32)|
            std::uint32_t(canonical(t.tile_y,input.world_height_tiles,input.world_wrap_y!=0));
    }
    static bool same_content(c3x_renderer_tile_v1 a,c3x_renderer_tile_v1 b) {
        // Canonical content can be shared across wrapping; finished pixels also
        // depend on the actual occurrence's world/projection basis.
        if(a.tile_x!=b.tile_x || a.tile_y!=b.tile_y ||
           a.visibility_mask!=b.visibility_mask || a.tile_visibility!=b.tile_visibility || a.fog_status!=b.fog_status ||
           ((a.tile_flags^b.tile_flags)&C3X_RENDERER_TILE_VISIBILITY_BITS))return false;
        a=render_core::CapturedScene::content(a);b=render_core::CapturedScene::content(b);
        return !std::memcmp(&a,&b,sizeof(a));
    }
    bool prepare(c3x_renderer_frame_v1 const& source,c3x_renderer_camera_identity_v1 epochs) {
        if(source.target_width<8 || source.target_height<8 || source.target_width>2240 || source.target_height>1260 ||
           !source.world_topology || !source.world_topology_count || source.world_topology_count>1024u*1024u || source.tile_count>8192 || !source.tile_count || !source.tiles)return false;
        viewport_width=source.target_width;viewport_height=source.target_height;
        pad_x=std::min(128,((2240-viewport_width)/16)*8);
        pad_y=std::min(128,((1260-viewport_height)/16)*8);
        if(!pad_x && !pad_y)return false;
        input=source;identity=epochs;
        tiles.assign(source.tiles,source.tiles+source.tile_count);source_tiles=tiles;
        unsigned count=0;for(auto const& tile:source_tiles)
            count+=!(tile.tile_flags&C3X_RENDERER_TILE_TOPOLOGY_HALO) || (tile.tile_flags&C3X_RENDERER_TILE_RENDER);
        detail_class=detail(count);
        topology.assign(source.world_topology,source.world_topology+source.world_topology_count);
        input.target_width+=2*pad_x;input.target_height+=2*pad_y;
        input.clip_left=input.clip_top=0;input.clip_right=input.target_width;input.clip_bottom=input.target_height;
        int l=2147483647,t=l,r=-2147483647,b=r;
        for(auto const& tile:tiles)if(tile.tile_flags&C3X_RENDERER_TILE_RENDER){
            l=std::min(l,tile.anchor_x);r=std::max(r,tile.anchor_x);t=std::min(t,tile.anchor_y);b=std::max(b,tile.anchor_y);
        }
        if(l>r)return false;
        for(unsigned i=0;i<tiles.size();++i){auto& tile=tiles[i];
            if(tile.anchor_x<-100000000 || tile.anchor_x>100000000 || tile.anchor_y<-100000000 || tile.anchor_y>100000000)return false;
            // Full captured appearance only. Lightweight topology never grants
            // permission to manufacture a newly visible object or replacement.
            if((tile.tile_flags&C3X_RENDERER_TILE_PREFETCH) && tile.anchor_x>=l-pad_x && tile.anchor_x<=r+pad_x &&
               tile.anchor_y>=t-pad_y && tile.anchor_y<=b+pad_y)
                tile.tile_flags=(tile.tile_flags&~(C3X_RENDERER_TILE_PREFETCH|C3X_RENDERER_TILE_TOPOLOGY_HALO))|C3X_RENDERER_TILE_RENDER;
            tile.anchor_x+=pad_x;tile.anchor_y+=pad_y;
            if(!index.emplace(key(tile),i).second)return false; // ambiguous wrapped occurrence: exact fallback
        }
        input.tiles=tiles.data();input.world_topology=topology.data();
        return std::uint64_t(input.target_width)*input.target_height*4+bytes()<budget;
    }
    bool finish(c3x_renderer_output_v1 const& output,std::vector<std::uint64_t> proof,Publication* resident=nullptr) {
        if(output.fallback_tile_count)return false;
        if(resident)map.swap(*resident);
        else if(!map.capture(output,0,0,&input,identity))return false;
        dependencies=std::move(proof);
        for(auto const& tile:tiles)if(tile.tile_flags&C3X_RENDERER_TILE_RENDER)dependencies.push_back(key(tile));
        std::sort(dependencies.begin(),dependencies.end());dependencies.erase(std::unique(dependencies.begin(),dependencies.end()),dependencies.end());
        auto centered=input;centered.target_width=viewport_width;centered.target_height=viewport_height;
        centered.clip_left=centered.clip_top=0;centered.clip_right=viewport_width;centered.clip_bottom=viewport_height;
        centered.tiles=source_tiles.data();
        auto crop_bytes=std::uint64_t(viewport_width)*viewport_height*4+
            source_tiles.size()*(sizeof(c3x_renderer_tile_v1)+4);
        if(bytes()+crop_bytes>budget)return false;
        return project(centered,identity,center) && bytes()<=budget;
    }
    // Static content validity and ambient sample freshness are independent.
    // A delayed ambient refresh may hold its actual old sample; it must not
    // force the native camera to wait for a still-valid world image.
    bool project(c3x_renderer_frame_v1 const& current,c3x_renderer_camera_identity_v1 epochs,Publication& result,
                 c3x_renderer_output_v1 const* sample=nullptr,c3x_renderer_i64 sample_ticks=0,Publication const* resident_sample=nullptr) const {
        if(!map.has_image() || std::memcmp(&identity,&epochs,sizeof(epochs)) ||
           current.target_width!=viewport_width || current.target_height!=viewport_height ||
           current.tile_width!=input.tile_width || current.tile_height!=input.tile_height ||
           current.hour!=input.hour || current.season!=input.season || current.presentation_frequency!=input.presentation_frequency ||
           current.presentation_time_ticks<input.presentation_time_ticks ||
           current.world_width_tiles!=input.world_width_tiles || current.world_height_tiles!=input.world_height_tiles ||
           current.world_wrap_x!=input.world_wrap_x || current.world_wrap_y!=input.world_wrap_y ||
           current.world_topology_revision!=input.world_topology_revision || current.world_topology_count!=topology.size() ||
           !current.world_topology || std::memcmp(current.world_topology,topology.data(),topology.size()*4) ||
           !current.tiles || current.tile_count>8192)return false;
        std::unordered_map<std::uint64_t,unsigned> fresh;
        int x=-1,y=-1;unsigned count=0;
        for(unsigned i=0;i<current.tile_count;++i){auto const& tile=current.tiles[i];
            count+=!(tile.tile_flags&C3X_RENDERER_TILE_TOPOLOGY_HALO) || (tile.tile_flags&C3X_RENDERER_TILE_RENDER);
            auto id=key(tile);if(!fresh.emplace(id,i).second)return false;
            if(x<0 && (tile.tile_flags&C3X_RENDERER_TILE_RENDER)){
                auto found=index.find(id);if(found==index.end())return false;
                auto const& old=tiles[found->second];
                auto dx=std::int64_t(old.anchor_x)-tile.anchor_x,dy=std::int64_t(old.anchor_y)-tile.anchor_y;
                if(dx<0 || dy<0 || dx>2*pad_x || dy>2*pad_y)return false;
                x=int(dx);y=int(dy);
            }
        }
        if(x<0 || y<0 || x>2*pad_x || y>2*pad_y || detail(count)!=detail_class)return false;
        for(auto id:dependencies){
            auto old=index.find(id);auto now=fresh.find(id);
            if(old==index.end()){if(now!=fresh.end())return false;continue;}
            if(now==fresh.end())return false;
            auto const& a=tiles[old->second];auto const& b=current.tiles[now->second];
            if(((a.tile_flags&(C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_PREFETCH)) &&
                !(b.tile_flags&(C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_PREFETCH))) ||
               !same_content(a,b) || std::int64_t(a.anchor_x)-b.anchor_x!=x || std::int64_t(a.anchor_y)-b.anchor_y!=y)return false;
        }
        std::vector<unsigned> replacements(current.tile_count);
        unsigned previous_occurrence=0;bool have_occurrence=false;
        for(unsigned i=0;i<current.tile_count;++i){auto const& tile=current.tiles[i];
            if(!(tile.tile_flags&C3X_RENDERER_TILE_RENDER))continue;
            auto found=index.find(key(tile));if(found==index.end())return false;
            auto n=found->second;auto const& old=tiles[n];
            // Alpha/depth ties make native occurrence order part of pixel
            // identity. Matching content cannot relabel a reordered donor.
            if(have_occurrence && n<=previous_occurrence)return false;
            previous_occurrence=n;have_occurrence=true;
            if(!(old.tile_flags&C3X_RENDERER_TILE_RENDER) || !same_content(old,tile) ||
               std::int64_t(old.anchor_x)-tile.anchor_x!=x || std::int64_t(old.anchor_y)-tile.anchor_y!=y)return false;
            replacements[i]=map.replacements[n];
        }
        auto const& sampled=sample?*sample:map.output;
        auto sampled_ticks=sample?sample_ticks:input.presentation_time_ticks;
        if(sampled.content_revision!=map.output.content_revision || sampled.device_generation!=map.output.device_generation ||
           sampled.width!=input.target_width || sampled.height!=input.target_height || (!sampled.bgra_pixels && !(resident_sample?resident_sample->has_image():map.has_image())) ||
           sampled_ticks>current.presentation_time_ticks)return false;
        if(result.has_image() && result.frame.presentation_time_ticks==sampled_ticks &&
           result.output.content_revision==map.output.content_revision && result.output.device_generation==map.output.device_generation &&
           !std::memcmp(&result.identity,&epochs,sizeof(epochs)) && result.refresh_view(current))return true;
        auto output=sampled;output.visible_animation_count=current.visible_animation_count+
            (map.output.visible_animation_count>input.visible_animation_count?map.output.visible_animation_count-input.visible_animation_count:0);
        output.request_continuous_redraw=output.visible_animation_count!=0;
        output.width=viewport_width;output.height=viewport_height;output.stride_bytes=viewport_width*4;
        output.clip_left=current.clip_left;output.clip_top=current.clip_top;
        output.clip_right=current.clip_right;output.clip_bottom=current.clip_bottom;
        output.bgra_pixels=nullptr;output.replacement_tile_count=current.tile_count;output.replacement_tile_flags=replacements.data();
        auto frame=current;frame.presentation_time_ticks=sampled_ticks;
        if(resident_sample)return result.capture_crop(*resident_sample,output,x,y,frame,epochs);
        if(!sample)return result.capture_crop(map,output,x,y,frame,epochs);
        Publication transient;transient.output=sampled; // borrowed CPU sample, never published
        return result.capture_crop(transient,output,x,y,frame,epochs);
    }
    bool centered(c3x_renderer_frame_v1 const& current) const {
        for(unsigned i=0;i<current.tile_count;++i)if(current.tiles[i].tile_flags&C3X_RENDERER_TILE_RENDER){
            auto found=index.find(key(current.tiles[i]));if(found==index.end())return false;
            auto const& old=tiles[found->second];
            auto dx=std::int64_t(old.anchor_x)-current.tiles[i].anchor_x-pad_x;
            auto dy=std::int64_t(old.anchor_y)-current.tiles[i].anchor_y-pad_y;
            return dx>=-pad_x/2 && dx<=pad_x/2 && dy>=-pad_y/2 && dy<=pad_y/2;
        }
        return false;
    }
    std::size_t bytes() const {
        return map.bytes()+center.bytes()+(tiles.capacity()+source_tiles.capacity())*sizeof(c3x_renderer_tile_v1)+topology.capacity()*4+
            dependencies.capacity()*8+index.size()*40+index.bucket_count()*sizeof(void*)+sizeof(*this);
    }
    void swap(PreparedViewArea& other) {
        map.swap(other.map);center.swap(other.center);std::swap(input,other.input);std::swap(identity,other.identity);tiles.swap(other.tiles);source_tiles.swap(other.source_tiles);topology.swap(other.topology);
        index.swap(other.index);dependencies.swap(other.dependencies);std::swap(viewport_width,other.viewport_width);
        std::swap(viewport_height,other.viewport_height);std::swap(pad_x,other.pad_x);std::swap(pad_y,other.pad_y);std::swap(detail_class,other.detail_class);
    }
    void clear(){PreparedViewArea empty;swap(empty);}
};
}
