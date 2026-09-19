#pragma once
#include "animation_runtime.h"
#include "unit_animation_runtime.h"
#include "unit_shadow.h"
#include "render_core/content_preparation.h"

namespace c3x_renderer {
// Immutable animation leases and scalar native pose inputs. Helpers never read
// renderer scratch, native objects/canvases, textures or the D3D context.
struct UnitPoseSource {
    std::vector<std::shared_ptr<AnimationMesh const>> meshes;
    float scale=1,yaw_offset=0,offset_z=0;
    bool allow_exit_clip=false;
    int shadow_extent=128;
};
struct UnitPoseInput {
    UnitPoseSource source;
    double phase=0;
    bool gpu_shadow=false;
    int direction=1,width=1,height=1,anchor_x=0,anchor_y=0;
    float zoom=1,light_x=0,light_y=0,shadow_strength=1;
};
struct UnitPoseContent {
    UnitShadow shadow;
    std::vector<unsigned char> ground_shadow;
    std::vector<std::array<float,12>> shadow_triangles;
    std::array<float,16> ground_projection{};
    std::vector<std::vector<std::array<float,17>>> uploads;
    std::array<int,4> coverage{}; // Conservative body plus filtered ground-shadow pixels.
    explicit UnitPoseContent(int extent,bool gpu=false):shadow(extent,!gpu){}
    std::size_t bytes() const {
        std::size_t size=sizeof(*this)+shadow_triangles.capacity()*sizeof(shadow_triangles[0])+ground_shadow.capacity()+shadow.heights.capacity()*sizeof(float)+uploads.capacity()*sizeof(uploads[0]);
        for(auto const& part:uploads)size+=part.capacity()*sizeof(part[0]);
        return size;
    }
};
struct UnitPoseCompiler {
    // Authored tangent sampling. Source vertex directions use the linear
    // skin matrix, independently normalized as in the inspected object VS.
    std::vector<std::array<std::array<float,3>,2>> sample_animation_frames(AnimationMesh const& mesh,double phase) {
        double frame=std::clamp(phase,0.0,1.0)*(mesh.frames-1);
        unsigned first=std::min(mesh.frames-1,unsigned(frame));
        unsigned second=std::min(mesh.frames-1,first+1);float fraction=float(frame-first);
        std::vector<std::array<std::array<float,3>,2>> out(mesh.vertices.size());
        for(std::size_t i=0;i<mesh.vertices.size();++i) {
            auto const& v=mesh.vertices[i];
            for(unsigned basis=0;basis<2;++basis) {
                auto const& source=basis?v.bitangent:v.tangent;auto & result=out[i][basis];
                for(unsigned influence=0;influence<4;++influence) {
                    if(v.weights[influence]==0)continue;
                    auto a=mesh.palettes.data()+(std::size_t(first)*mesh.bones+v.joints[influence])*16;
                    auto b=mesh.palettes.data()+(std::size_t(second)*mesh.bones+v.joints[influence])*16;
                    for(unsigned axis=0;axis<3;++axis)for(unsigned c=0;c<3;++c)
                        result[axis]+=v.weights[influence]*source[c]*(a[c*4+axis]+(b[c*4+axis]-a[c*4+axis])*fraction);
                }
                float length=std::sqrt(result[0]*result[0]+result[1]*result[1]+result[2]*result[2]);
                for(unsigned axis=0;axis<3;++axis)result[axis]=length>1e-12f?result[axis]/length:source[axis];
            }
        }
        return out;
    }


    std::unique_ptr<UnitPoseContent> operator()(UnitPoseInput const& input,std::atomic<bool> const& cancelled,unsigned) {
        auto const& source=input.source;
        std::size_t caster_count=0;
        if(input.gpu_shadow){
            constexpr std::size_t limit=16u*1024u*1024u/sizeof(std::array<float,12>);
            for(auto const& mesh:source.meshes){
                auto count=mesh->indices.size()/3;
                if(count>limit-caster_count)return {};
                caster_count+=count;
            }
        }
        auto result=std::make_unique<UnitPoseContent>(source.shadow_extent,input.gpu_shadow);
        result->shadow_triangles.reserve(caster_count);
        auto& shadow=result->shadow;
        auto cosine=std::cos((source.yaw_offset+float(input.direction%8)*45)*.01745329252f);
        auto sine=std::sin((source.yaw_offset+float(input.direction%8)*45)*.01745329252f);
        float scale=source.scale,zoom=input.zoom;int w=input.width,h=input.height;
        std::vector<std::vector<FeatureSourceVertex>> poses(source.meshes.size());
        std::vector<std::vector<UnitShadow::Point>> positions(source.meshes.size());
        std::vector<UnitShadow::Point> all_points;
        for(std::size_t part_index=0;part_index<source.meshes.size();++part_index) {
            if(cancelled.load(std::memory_order_relaxed))return {};
            if(!sample_animation_mesh(*source.meshes[part_index],
                input.phase*source.meshes[part_index]->duration,false,poses[part_index]))return {};
            for(auto const& p:poses[part_index]) {
                UnitShadow::Point point={(p.position[0]*cosine-p.position[1]*sine)*scale,
                    (p.position[0]*sine+p.position[1]*cosine)*scale,(p.position[2]+source.offset_z)*scale};
                positions[part_index].push_back(point);all_points.push_back(point);
            }
        }
        if(!shadow.fit(all_points,input.light_x,input.light_y))return {};
        for(std::size_t part_index=0;part_index<source.meshes.size();++part_index) {
            auto const& mesh=*source.meshes[part_index];auto const& points=positions[part_index];
            for(std::size_t i=0;i<mesh.indices.size();i+=3) {
                if((i%192)==0 && cancelled.load(std::memory_order_relaxed))return {};
                auto a=points[mesh.indices[i]],b=points[mesh.indices[i+1]],c=points[mesh.indices[i+2]];
                if(input.gpu_shadow){
                    a=shadow.project(a);b=shadow.project(b);c=shadow.project(c);
                    for(auto p:{&a,&b,&c}){(*p)[0]=((*p)[0]-shadow.left)/shadow.width*shadow.extent;(*p)[1]=((*p)[1]-shadow.top)/shadow.height*shadow.extent;}
                    float area=(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
                    if(std::abs(area)>=1e-7f)result->shadow_triangles.push_back({a[0],a[1],a[2],area,b[0],b[1],b[2],0,c[0],c[1],c[2],0});
                }else shadow.triangle(a,b,c);
            }
        }
        // Translation-free finishing input, prepared by the same CPU pose owner.
        // Keep the native shadow arithmetic exact; the GPU combines this coverage
        // with the rendered body's alpha without reading the body back.
        if(w<1||h<1||w>1024||h>1024)return {};
        result->ground_projection={float(input.anchor_x),float(input.anchor_y),zoom,
            255*lighting::c3x_dynamic_shadow_opacity*input.shadow_strength,
            shadow.left,shadow.top,shadow.width,shadow.height,float(shadow.extent)};
        if(!input.gpu_shadow)result->ground_shadow.resize(std::size_t(w)*h);
        for(int y=0;!input.gpu_shadow && y<h;++y){
            if(cancelled.load(std::memory_order_relaxed))return {};
            for(int x=0;x<w;++x){
                float sx=(float(x)+.5f-float(input.anchor_x))/(64*zoom);
                float sy=(float(y)+.5f-float(input.anchor_y))/(32*zoom);
                float fade=std::clamp(float(std::min({x,y,w-1-x,h-1-y}))/3,0.f,1.f);
                result->ground_shadow[std::size_t(y)*w+x]=static_cast<unsigned char>(255*lighting::c3x_dynamic_shadow_opacity*input.shadow_strength*fade*shadow.coverage((sx+sy)*.5f,(sy-sx)*.5f));
            }
        }
        result->uploads.resize(source.meshes.size());
        for(std::size_t part_index=0;part_index<source.meshes.size();++part_index) {
            if(cancelled.load(std::memory_order_relaxed))return {};
            auto const& mesh=*source.meshes[part_index];auto const& posed=poses[part_index];
            auto& upload=result->uploads[part_index];
            upload.resize(posed.size());
            auto frames=sample_animation_frames(mesh,input.phase);

            for(std::size_t i=0;i<posed.size();++i) {
                auto const& p=posed[i];
                float x=(p.position[0]*cosine-p.position[1]*sine)*scale;
                float y=(p.position[0]*sine+p.position[1]*cosine)*scale,z=(p.position[2]+source.offset_z)*scale;
                float sx=float(input.anchor_x)+(x-y)*64*zoom;
                float sy=float(input.anchor_y)+(x+y)*32*zoom-z*(150.f*128/224)*zoom;
                auto normal=lighting::object_normal(p.normal[0]*cosine-p.normal[1]*sine,
                    p.normal[0]*sine+p.normal[1]*cosine,p.normal[2]);
                upload[i]={2*sx/w-1,1-2*sy/h,.5f-(x+y)*.05f-z*.001f,
                    normal[0],normal[1],normal[2],p.uv[0],p.uv[1],z,
                    (x-shadow.dx*z-shadow.left)/shadow.width,(y-shadow.dy*z-shadow.top)/shadow.height};
                for(unsigned basis=0;basis<2;++basis) {
                    auto f=frames[i][basis];
                    auto direction=lighting::object_normal(f[0]*cosine-f[1]*sine,f[0]*sine+f[1]*cosine,f[2]);
                    for(unsigned axis=0;axis<3;++axis)upload[i][11+basis*3+axis]=direction[axis];
                }
            }
            // The ground plane hides buried anatomy/stowed equipment. Bound
            // the visible polygon, including intersections of crossing edges,
            // so clipping never extends beyond Civ III's native dirty region.
            auto inside=[&](float x,float y){return x>=2.f/w-1 && x<=1-2.f/w && y>=2.f/h-1 && y<=1-2.f/h;};
            for(std::size_t i=0;i<mesh.indices.size();i+=3) {
                for(unsigned edge=0;edge<3;++edge) {
                    auto const& a=upload[mesh.indices[i+edge]];
                    auto const& b=upload[mesh.indices[i+(edge+1)%3]];
                    // Some terminal clips send a mount offscreen. The GPU
                    // still clips strictly to this same native-sized target.
                    if(!source.allow_exit_clip && a[8]>=0 && !inside(a[0],a[1]))return {};
                    if(!source.allow_exit_clip && (a[8]<0)!=(b[8]<0)) {
                        float t=a[8]/(a[8]-b[8]);
                        if(!inside(a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1])))return {};
                    }
                }
            }
        }
        float left=float(w),top=float(h),right=0,bottom=0;
        auto include=[&](float x,float y){left=std::min(left,x);top=std::min(top,y);right=std::max(right,x);bottom=std::max(bottom,y);};
        for(auto const& part:result->uploads)for(auto const& v:part)include((v[0]+1)*w*.5f,(1-v[1])*h*.5f);
        // The 3x3 height filter reaches one cell beyond its stored domain.
        // Extra cells and two pixels conservatively cover floor/MSAA rounding.
        if(input.shadow_strength>0){float dx=shadow.width*2/shadow.extent,dy=shadow.height*2/shadow.extent;
            for(float x:{shadow.left-dx,shadow.left+shadow.width+dx})for(float y:{shadow.top-dy,shadow.top+shadow.height+dy})
                include(input.anchor_x+(x-y)*64*zoom,input.anchor_y+(x+y)*32*zoom);}
        result->coverage={std::clamp(int(std::floor(left))-2,0,w),std::clamp(int(std::floor(top))-2,0,h),
            std::clamp(int(std::ceil(right))+2,0,w),std::clamp(int(std::ceil(bottom))+2,0,h)};
        return result;
    }
};
}
