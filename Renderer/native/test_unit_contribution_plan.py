"""Host-only executable contracts for the actual unit selection and preparation."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
DIRECT = ROOT / "Renderer/sandbox/direct_units.h"
BODY = ROOT / "Renderer/native/unit_body_renderer.h"


def host_cpp(program, sources=()):
    # This test never calls native_cpp_test/platform or any Windows/VM tool.
    if os.name == "nt":
        raise unittest.SkipTest("Explicit host-only contract")
    compiler = shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        raise unittest.SkipTest("Host C++17 compiler unavailable")
    if "windows.h" in program:
        raise AssertionError("Windows dependency in host contract")
    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / "contract.cpp"
        binary = Path(folder) / "contract"
        source.write_text(program)
        result = subprocess.run([compiler, "-std=c++17", "-O1", "-Wall", "-Wextra", "-I", str(ROOT),
                                 str(source), *(str(ROOT / s) for s in sources), "-o", str(binary)], capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(result.stderr)
        subprocess.run([str(binary)], check=True, timeout=30)


BASE = r'''
#include "Renderer/native/render_core/unit_contribution_plan.h"
#include "Renderer/native/render_core/unit_pose_transition.h"
#include "Renderer/native/unit_animation_runtime.h"
#include "Renderer/native/render_core/combat_effects.h"
#include <cassert>
#include <memory>
#include <string>
using namespace c3x_renderer;
using namespace c3x_renderer::render_core;
AnimationMesh fixture(float translate=0){
 AnimationMesh m;m.frames=2;m.bones=1;m.duration=1;m.vertices.resize(1);
 m.vertices[0].source.position[0]=1;m.vertices[0].joints={0,0,0,0};m.vertices[0].weights={1,0,0,0};
 m.palettes.resize(32);for(int f=0;f<2;++f){auto* p=m.palettes.data()+f*16;p[0]=p[5]=p[10]=p[15]=1;p[12]=translate;}
 return m;
}
UnitContributionCandidate candidate(double x,double y,double radius=0){
 UnitContributionCandidate c;c.visible=true;c.anchor_x=x;c.anchor_y=y;c.bounds={radius,true};c.ground_known=true;return c;
}
'''


def metadata_owner():
    source = BODY.read_text()
    block = source[source.index("    // Compact all-clip proof"):source.index("    std::size_t resident_bytes=0;")]
    declarations = source[source.index("    struct Mesh {"):source.index("    std::vector<Mesh> meshes;")]
    return r'''
struct ID3D11Buffer{};
struct ID3D11ShaderResourceView{};
struct Owner {
''' + declarations + r'''
 std::vector<Mesh> meshes;std::vector<Unit> units;
''' + block + r'''
};
'''



def direct_stub():
    body = BODY.read_text()
    declarations = body[body.index("    struct Mesh {"):body.index("    std::vector<Mesh> meshes;")]
    metadata = body[body.index("    // Compact all-clip proof"):body.index("    std::size_t resident_bytes=0;")]
    source = DIRECT.read_text()
    fields = source[source.index("    using ScenePose="):source.index("#endif", source.index("    using ScenePose="))]
    fields = fields.replace("c3x_renderer::render_core::UnitInstances::ScenePose", "::ScenePose")
    methods = source[source.index("    using ContributionPlan="):source.rindex("#endif")]
    methods = methods.replace("std::vector<c3x_renderer::render_core::UnitInstances::ScenePose>", "std::vector<ScenePose>")
    return BASE + r'''
#include "Renderer/native/c3x_renderer_api.h"
#include "Renderer/native/scene_projection.h"
#include "Renderer/native/source_fidelity/light_frame.h"
#include "Renderer/native/render_core/frame_sample_cache.h"
#include "Renderer/native/render_core/skin_shadow_bounds.h"
#include "Renderer/native/render_core/combat_effects.h"
#include <climits>
#include <cfloat>
#define FALSE 0
#define TRUE 1
#define FAILED(x) ((x)<0)
#define SUCCEEDED(x) ((x)>=0)
using UINT=unsigned;using LONG=long;
struct D3D11_RECT{LONG left=0,top=0,right=0,bottom=0;};
struct D3D11_VIEWPORT{float TopLeftX,TopLeftY,Width,Height,MinDepth,MaxDepth;};
struct D3D11_DEPTH_STENCIL_DESC{bool StencilEnable=false;unsigned StencilReadMask=0,StencilWriteMask=0;
 struct Face{int StencilFunc,StencilFailOp,StencilDepthFailOp,StencilPassOp;};Face FrontFace{},BackFace{};};
struct D3D11_BUFFER_DESC{unsigned ByteWidth=0,BindFlags=0,MiscFlags=0,StructureByteStride=0,Usage=0;};
struct D3D11_SUBRESOURCE_DATA{void const* pSysMem=nullptr;};
struct D3D11_SHADER_RESOURCE_VIEW_DESC{int ViewDimension=0;struct{unsigned NumElements=0;}Buffer;};
enum {D3D11_BIND_SHADER_RESOURCE=1,D3D11_RESOURCE_MISC_BUFFER_STRUCTURED=2,D3D11_SRV_DIMENSION_BUFFER=3,
 D3D11_CLEAR_DEPTH=16,D3D11_CLEAR_STENCIL=4,D3D11_COMPARISON_ALWAYS=5,D3D11_STENCIL_OP_KEEP=6,D3D11_STENCIL_OP_REPLACE=7,
 D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST=8,DXGI_FORMAT_R32_UINT=9,D3D11_USAGE_DEFAULT=10,D3D11_BIND_CONSTANT_BUFFER=11,
 D3D11_STENCIL_OP_INCR_SAT=12,D3D11_COMPARISON_EQUAL=13};
struct ID3D11Buffer{unsigned references=1;bool heap=false,immutable=false;std::array<float,32> values{};
 static inline unsigned live_heap=0;
 void retain(){++references;}void release(){if(!--references&&heap){--live_heap;delete this;}}};
struct ID3D11ShaderResourceView{};struct ID3D11Texture2D{};struct ID3D11RenderTargetView{};
struct ID3D11DepthStencilState{void GetDesc(D3D11_DEPTH_STENCIL_DESC* out){*out={};}};
namespace Microsoft{namespace WRL{template<class T>struct ComPtr{
 T* pointer=nullptr;
 static void retain(T* p){if constexpr(std::is_same<T,ID3D11Buffer>::value)if(p)p->retain();}
 static void release(T* p){if constexpr(std::is_same<T,ID3D11Buffer>::value)if(p)p->release();}
 ComPtr()=default;ComPtr(T* p):pointer(p){retain(pointer);}ComPtr(ComPtr const& p):pointer(p.pointer){retain(pointer);}
 ComPtr& operator=(ComPtr const& p){retain(p.pointer);release(pointer);pointer=p.pointer;return *this;}
 ~ComPtr(){release(pointer);}T* Get()const{return pointer;}
 explicit operator bool()const{return pointer!=nullptr;}T** operator&(){return &pointer;}
 ComPtr& operator=(T* p){retain(p);release(pointer);pointer=p;return *this;}void Reset(){release(pointer);pointer=nullptr;}
};}}
struct Device{
 bool fail_material=false;unsigned material_creates=0;
 int CreateBuffer(D3D11_BUFFER_DESC* d,D3D11_SUBRESOURCE_DATA const* initial,ID3D11Buffer** p){
  if(d->BindFlags==D3D11_BIND_CONSTANT_BUFFER){if(fail_material)return -1;auto* b=new ID3D11Buffer;b->heap=true;
   ++ID3D11Buffer::live_heap;++material_creates;assert(!initial&&d->ByteWidth==128&&d->Usage==D3D11_USAGE_DEFAULT);*p=b;return 0;}
  static ID3D11Buffer value;*p=&value;return 0;}
 template<class T,class Desc,class U>int CreateShaderResourceView(T*,Desc*,U** p){static U value;*p=&value;return 0;}
 int CreateDepthStencilState(D3D11_DEPTH_STENCIL_DESC*,ID3D11DepthStencilState** p){static ID3D11DepthStencilState value;*p=&value;return 0;}
};
struct Context{
 void* shader=nullptr;unsigned bodies=0,shadows=0,uploads=0,material_uploads=0;void* shadow_shader=nullptr;
 ID3D11Buffer* legacy_material=nullptr;ID3D11Buffer* bound_material=nullptr;std::vector<std::array<float,32>> body_materials;
 template<class... A>void ClearDepthStencilView(A&&...){}
 template<class... A>void OMSetRenderTargets(A&&...){}
 template<class... A>void OMSetDepthStencilState(A&&...){}
 template<class... A>void OMSetBlendState(A&&...){}
 template<class... A>void RSSetState(A&&...){}
 template<class... A>void RSSetViewports(A&&...){}
 template<class... A>void RSSetScissorRects(A&&...){}
 template<class... A>void IASetInputLayout(A&&...){}
 template<class... A>void IASetPrimitiveTopology(A&&...){}
 template<class... A>void VSSetShader(A&&...){}
 template<class T>void PSSetShader(T* p,void*,unsigned){shader=p;}
 template<class... A>void VSSetConstantBuffers(A&&...){}
 void PSSetConstantBuffers(unsigned slot,unsigned,ID3D11Buffer* const* b){if(slot==0)bound_material=*b;}
 template<class... A>void PSSetShaderResources(A&&...){}
 template<class... A>void VSSetShaderResources(A&&...){}
 template<class... A>void PSSetSamplers(A&&...){}
 void UpdateSubresource(ID3D11Buffer* buffer,unsigned,void const*,void const* source,unsigned,unsigned){
  ++uploads;if(buffer->heap||buffer==legacy_material){++material_uploads;std::memcpy(buffer->values.data(),source,128);}}
 template<class... A>void IASetVertexBuffers(A&&...){}
 template<class... A>void IASetIndexBuffer(A&&...){}
 void DrawIndexed(unsigned,unsigned,int){if(shader==shadow_shader)++shadows;else{++bodies;assert(bound_material);body_materials.push_back(bound_material->values);}}
};
struct ScenePose{c3x_renderer_unit_v1 draw{};unsigned unit=0,action=0;std::uint64_t pose_identity=1;int tile_x=0,tile_y=0;bool cursor=false;
 long long pose_ticks=-1;bool travelling=false,owner_ring=false;};
inline unsigned GetEnvironmentVariableA(char const*,char*,unsigned){return 0;}
template<class... A>int sscanf_s(A&&...){return 0;}
namespace c3x_renderer{namespace tactical{
struct Input{std::vector<int> primitives;void ring(float,float,float,bool){primitives.push_back(1);}
 void owner_disc(float,float,float,std::array<float,4>){primitives.push_back(2);}};
}}
struct Tactical{struct Rect{int left,top,right,bottom;};void draw_into(Device*,Context*,c3x_renderer::tactical::Input const&,Rect,double,ID3D11RenderTargetView*,unsigned,unsigned,float,float){}};
namespace c3x_renderer{struct UnitBodyRenderer{
''' + declarations + r'''
 std::vector<Mesh> meshes;std::vector<Texture> textures;std::vector<Unit> units;float look[3]={};
''' + metadata + r'''
};}
struct Renderer{
 Device device_value;Context context_value;Device* device=&device_value;Context* context=&context_value;
 c3x_renderer::UnitBodyRenderer unit_bodies;std::array<float,3> unit_look{};unsigned content_view_width=1000,content_view_height=800;
 struct{bool enabled=true;}reflection;
 struct{struct Field{std::vector<unsigned char> pixels;float amplitude=0;};struct{std::array<Field,2> fields;}low_relief;
 ID3D11DepthStencilState depth;ID3D11DepthStencilState* decal_depth=&depth;}natural;
 struct{float depth_translation=0;}geometry_viewport_settings;
 ID3D11DepthStencilState depth_value;ID3D11DepthStencilState* depth_state=&depth_value;
 void* rasterizer_state=nullptr;void* blend_state=nullptr;Tactical tactical_gpu;
} renderer;
struct SandboxPassWorkload{
 enum{reflected_units,units};struct Scope{Scope(SandboxPassWorkload&,int){}};
 template<class T>void clear(T*){}template<class T>void upload_buffer(T*){}void draw(std::size_t){}void upload(std::size_t){}
};
struct Direct{
 struct Vertex{char values[88];};
 struct Mesh{Microsoft::WRL::ComPtr<ID3D11Buffer> vertices,indices;Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> palette_view;
 c3x_renderer::render_core::SkinShadowBounds shadow_bounds;};
 std::vector<Mesh> meshes;UnitPoseTransitions transitions;UnitShadow shadow_fit{512,false};
 Microsoft::WRL::ComPtr<ID3D11Texture2D> self_shadow;
 Microsoft::WRL::ComPtr<ID3D11RenderTargetView> self_shadow_target;
 Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> self_shadow_view;
 std::vector<UnitShadow::Point> shadow_points;unsigned draws=0,ground_queries=0,self_maps=0;
 ID3D11DepthStencilState* visible_depth=nullptr;ID3D11DepthStencilState* shadow_once=nullptr;void* layout=nullptr;
 float look[3]={};bool look_read=false,look_override=false;int owner_rings=-1;void* vertex=nullptr;int pixel_value=0,shadow_value=1;
 int* pixel=&pixel_value;int* shadow_pixel=&shadow_value;ID3D11Buffer placement_value,material_value,beauty_value;
 ID3D11Buffer* placement=&placement_value;ID3D11Buffer* material=&material_value;ID3D11Buffer* beauty=&beauty_value;
 ID3D11ShaderResourceView srv;ID3D11ShaderResourceView* unshadowed_view=&srv;void* samplers[4]={};
 SandboxPassWorkload work_value;SandboxPassWorkload* work=&work_value;
 // Streamed part placements: one upload per unit, one bound record per body draw.
 struct Stream{static constexpr unsigned stride=256;unsigned uploads=0,binds=0,records=0;
  bool available(Device*,Context*){return true;}
  template<class T>bool upload(T const*,unsigned count){++uploads;records=count;return count<=256;}
  void bind(unsigned slot,unsigned record){assert(slot==2&&record<records);++binds;}void bind_pixel(unsigned,unsigned){}} placement_stream;
 std::vector<std::array<float,28>> part_placements;
 bool initialize(){return true;}
 float unit_low_ground(c3x_renderer_frame_v1 const&,float,float){++ground_queries;return 0;}
 bool bind_palette(Mesh const&,float const*){return true;}
''' + fields + r'''
 template<class... A>bool draw_self_shadow(A&&...){++self_maps;static ID3D11Texture2D texture;static ID3D11ShaderResourceView view;
 self_shadow=&texture;self_shadow_view=&view;return true;}
''' + methods + r'''
};
struct Target{unsigned width=1008,height=808;ID3D11RenderTargetView view;ID3D11DepthStencilState state;
 ID3D11RenderTargetView* target=&view;ID3D11DepthStencilState* depth=&state;};
void setup(Direct& direct){
 renderer.unit_bodies.units.resize(1);UnitBodyRenderer::Action action;action.name="idle";action.loop=true;action.parts.push_back({});
 renderer.unit_bodies.units[0].actions.push_back(action);renderer.unit_bodies.meshes.resize(1);renderer.unit_bodies.textures.resize(1);
 renderer.unit_bodies.meshes[0].animation=std::make_shared<AnimationMesh const>(fixture());
 static ID3D11ShaderResourceView texture;renderer.unit_bodies.textures[0].view=&texture;
 direct.meshes.resize(1);static ID3D11Buffer buffer;direct.meshes[0].vertices=&buffer;direct.meshes[0].indices=&buffer;
 direct.meshes[0].palette_view=&texture;direct.meshes[0].shadow_bounds.prepare(*renderer.unit_bodies.meshes[0].animation);
 renderer.context->shadow_shader=direct.shadow_pixel;renderer.context->legacy_material=direct.material;
}
ScenePose occurrence(int id,int x,int y){ScenePose p;p.draw.unit_id=id;p.draw.action=1;p.draw.direction=1;
 p.draw.frame_count=10;p.draw.sprite_width=128;p.draw.sprite_height=64;p.draw.body_x=x-64;p.draw.body_y=y-32;
 p.draw.projection_scale_milli=1000;return p;}
'''


class UnitContributionTests(unittest.TestCase):
    def test_independent_pass_union_hidden_unknown_and_wrap_occurrences(self):
        host_cpp(BASE + r'''
int main(){
 UnitContributionView v;v.width=1000;v.height=800;v.shadow=true;v.shadow_x=-2;v.shadow_y=0;
 v.reflection=true;v.receivers={{400,500,600,650}};
 auto main=candidate(500,400);auto shadow=candidate(1500,900);shadow.offset_z=10;
 auto mirror=candidate(500,-300);mirror.offset_z=10;
 auto far=candidate(9000,9000);auto hidden=main;hidden.visible=false;
 assert(UnitContributionPlan::select(main,v)==(unit_main_body|unit_ground_shadow));
 assert(UnitContributionPlan::select(shadow,v)==unit_ground_shadow);
 assert(UnitContributionPlan::select(mirror,v)==unit_reflection);
 assert(UnitContributionPlan::select(far,v)==0&&UnitContributionPlan::select(hidden,v)==0);
 auto unknown=far;unknown.bounds.known=false;
 assert(UnitContributionPlan::select(unknown,v)==7);unknown.visible=false;assert(UnitContributionPlan::select(unknown,v)==0);
 // Same native actor at different wrapped occurrences retains each anchor.
 std::vector<UnitContributionCandidate> all={main,far,shadow,mirror,main};
 auto p=UnitContributionPlan::build(all,v);assert(p.entries.size()==4&&p.main==2&&p.shadow==3&&p.reflection==1);
 auto union_poses=p.required(all);assert(union_poses.size()==4&&union_poses[1].anchor_x==1500&&union_poses[3].anchor_x==500);
}
''')

    def test_zoom_native_scale_overhang_ground_and_receiver_distortion(self):
        host_cpp(BASE + r'''
int main(){
 UnitContributionView v;v.width=1000;v.height=800;
 auto edge=candidate(1120,400,2);assert(UnitContributionPlan::select(edge,v)==unit_main_body);
 edge.projection_scale=.5;assert(UnitContributionPlan::select(edge,v)==0);
 auto zoom=candidate(850,400);assert(UnitContributionPlan::select(zoom,v)==1);v.zoom=3;assert(UnitContributionPlan::select(zoom,v)==0);
 v.zoom=1;auto elevated=candidate(500,900);elevated.ground_min=110;elevated.ground_max=130;
 assert(UnitContributionPlan::select(elevated,v)==1);elevated.ground_known=false;assert(UnitContributionPlan::select(elevated,v)==1);
 v.reflection=true;v.receivers={{400,450,600,550}};auto reflected=candidate(500,-300);reflected.offset_z=10;
 assert(UnitContributionPlan::select(reflected,v)==0);
 // Receiver selection already carries water distortion/filter expansion.
 v.receivers[0].bottom+=36;assert(UnitContributionPlan::select(reflected,v)==unit_reflection);
 reflected.anchor_x=INFINITY;assert(UnitContributionPlan::select(reflected,v)==(unit_main_body|unit_reflection));
}
''')

    def test_baked_bound_survives_payload_release_and_invalid_metadata_admits(self):
        host_cpp(BASE + r'''
int main(){
 UnitMeshContributionBounds proof;
 {auto m=fixture(4);assert(proof.prepare(m));assert(proof.baked_radius>=5);}
 auto envelope=unit_contribution_bounds({&proof});assert(envelope.known&&envelope.radius>=5);
 auto bad=fixture();bad.vertices[0].joints[0]=1;UnitMeshContributionBounds invalid;assert(!invalid.prepare(bad));
 // Shared action paths require authored and stripped envelopes. A large
 // middle-frame reversal can lie outside both endpoint spheres after strip.
 auto travel=fixture();travel.frames=3;travel.palettes.resize(48);
 for(unsigned f=0;f<3;++f){auto* m=travel.palettes.data()+f*16;m[0]=m[5]=m[10]=m[15]=1;}
 travel.palettes[28]=-10;travel.palettes[44]=3;
 UnitMeshContributionBounds both;assert(both.prepare(travel)&&both.baked_radius>=12.5);
 assert(travel.palettes[28]==-10&&travel.palettes[44]==3); // extraction is read-only
 assert(!unit_contribution_bounds({&proof,&invalid}).known);
 auto c=candidate(50000,50000);c.bounds=unit_contribution_bounds({&invalid});
 UnitContributionView v;v.width=1000;v.height=800;assert(UnitContributionPlan::select(c,v)==1);
}
''')

    def test_rig_local_envelope_contains_rotational_midpoint_and_interruption(self):
        host_cpp(BASE + r'''
AnimationMesh rig(float sign){
 auto m=fixture();m.rig.binding[0]=7;m.rig.parents={-1};m.rig.skin_joints={0};
 m.rig.inverse_bind={JointMatrix{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1}};
 JointPose p;p.scale={1,0,0,0,1,0,0,0,1};p.rotation={0,0,sign*std::sqrt(.5f),std::sqrt(.5f)};
 p.position={2*sign,0,0};m.rig.poses={p,p};return m;
}
int main(){
 auto a=rig(-1),b=rig(1);UnitMeshContributionBounds ma,mb;assert(ma.prepare(a)&&mb.prepare(b));
 auto envelope=unit_contribution_bounds({&ma,&mb});assert(envelope.known);
 auto from=a.rig.poses[0],to=b.rig.poses[0];
 for(int i=0;i<=100;++i){auto mixed=mix_joint(from,to,i/100.f);auto matrix=joint_matrix(mixed);
   double x=matrix[0]+matrix[12],y=matrix[1]+matrix[13],z=matrix[2]+matrix[14];
   assert(std::sqrt(x*x+y*y+z*z)<=envelope.radius);
   auto interrupted=mix_joint(mixed,from,.33f);matrix=joint_matrix(interrupted);
   x=matrix[0]+matrix[12];y=matrix[1]+matrix[13];z=matrix[2]+matrix[14];assert(std::sqrt(x*x+y*y+z*z)<=envelope.radius);
 }
 // Unit scale norm uses identity=1, avoiding sqrt(3)^hierarchy inflation.
 assert(UnitMeshContributionBounds::linear_norm(a.rig.poses[0].scale.data(),3)==1);
 mb.parents[0]=0;assert(!unit_contribution_bounds({&ma,&mb}).known);
}
''')

    def test_actual_owner_initializes_only_captured_types_once_with_bound_budget(self):
        host_cpp(BASE + metadata_owner() + r'''
struct Pose{unsigned unit;};
int main(){
 Owner owner;owner.meshes.resize(4);owner.units.resize(2);
 Owner::Action idle;idle.name="idle";idle.parts.push_back({0});owner.units[0].actions.push_back(idle);
 auto move=idle;move.name="move";move.parts[0].mesh=1;owner.units[0].actions.push_back(move);
 idle.parts[0].mesh=2;owner.units[1].actions.push_back(idle);
 unsigned calls=0;auto load=[&](unsigned index,bool strip,UnitMeshContributionBounds& out){
  ++calls;assert(index<2);assert(!strip);auto temporary=fixture();return out.prepare(temporary);};
 assert(!owner.ensure_contribution_bounds(std::vector<Pose>{{0},{0}},load,[]{return false;},1));
 assert(calls==1&&!owner.contribution_bound(0).known);
 assert(owner.ensure_contribution_bounds(std::vector<Pose>{{0}},load,[]{return false;},1));
 assert(calls==2&&owner.contribution_bound(0).known&&!owner.meshes[2].contribution_attempted);
 auto bytes=owner.contribution_bytes;
 assert(owner.ensure_contribution_bounds(std::vector<Pose>{{0}},load,[]{return false;}));assert(calls==2&&owner.contribution_bytes==bytes);
 assert(!owner.ensure_contribution_bounds(std::vector<Pose>{{1}},load,[]{return true;}));assert(calls==2);
 auto fail=[&](unsigned,bool,UnitMeshContributionBounds&){++calls;return false;};
 assert(!owner.ensure_contribution_bounds(std::vector<Pose>{{1}},fail,[]{return false;}));
 assert(!owner.ensure_contribution_bounds(std::vector<Pose>{{1}},fail,[]{return false;}));assert(calls==3);
 // Optional bound ownership never grows past its hard allowance.
 Owner capped;capped.meshes.resize(1);capped.units.resize(1);idle.parts[0].mesh=0;capped.units[0].actions.push_back(idle);
 capped.contribution_bytes=capped.contribution_limit;
 assert(!capped.ensure_contribution_bounds(std::vector<Pose>{{0}},load,[]{return false;}));
 assert(capped.contribution_bytes==capped.contribution_limit&&!capped.contribution_bound(0).known&&capped.contribution_unknown==1);
 // Allocation failure in the optional certificate decoder admits unknown.
 Owner memory;memory.meshes.resize(1);memory.units.resize(1);memory.units[0].actions.push_back(idle);
 assert(!memory.ensure_contribution_bounds(std::vector<Pose>{{0}},[](unsigned,bool,auto&)->bool{throw std::bad_alloc();},[]{return false;}));
 assert(memory.contribution_bytes==0&&memory.contribution_unknown==1);
 // No strong payload ownership is necessary after proof adoption/eviction.
 auto strong=std::make_shared<AnimationMesh const>(fixture());owner.meshes[3].animation=strong;
 assert(!owner.remember_contribution_bounds(3));assert(owner.remember_contribution_bounds(3,true));owner.meshes[3].animation.reset();assert(strong.use_count()==1&&owner.meshes[3].contribution_bounds.known);
}
''')

    def test_actual_preparation_union_reuses_material_and_shadow_samples(self):
        host_cpp(direct_stub() + r'''
int main(){Direct direct;setup(direct);c3x_renderer_frame_v1 frame{};frame.target_width=1000;frame.target_height=800;
 frame.tile_width=128;frame.tile_height=64;frame.presentation_frequency=1000;frame.presentation_time_ticks=1;
 std::vector<ScenePose> poses={occurrence(1,500,400),occurrence(2,9000,9000),occurrence(3,1500,900)};
 UnitContributionPlan plan;plan.entries={{0,unit_main_body},{2,unit_ground_shadow}};
 assert(direct.prepare_real(frame,poses,12,plan));assert(direct.required_samples==2&&direct.ground_queries==2);
 assert(direct.material_builds==1&&direct.material_reuses==1&&direct.main_contributors==1&&direct.shadow_contributors==1);
 auto fit=direct.prepared_units[0].shadow_slot;Target target;
 assert(direct.draw_real(frame,poses,target,1,12,true,1));assert(renderer.context->bodies==0&&direct.self_maps==0);
 assert(direct.draw_real(frame,poses,target,1,12,false,1));assert(renderer.context->bodies==1&&renderer.context->shadows==1&&direct.self_maps==1);
 assert(direct.placement_stream.uploads>=2&&direct.placement_stream.binds>=2); // ground shadow and body records
 ++frame.presentation_time_ticks;assert(direct.prepare_real(frame,poses,12,plan));
 assert(direct.material_builds==0&&direct.material_reuses==2&&direct.prepared_units[0].shadow_slot==fit);
 assert(direct.draw_real(frame,poses,target,1,12,false,1));assert(direct.self_maps==1&&direct.shadow_reuses==1);
 // Camera changes only placement; immutable materials/shadow sample still borrow.
 for(auto& p:poses){p.draw.body_x+=100;p.draw.body_y+=50;}
 ++frame.presentation_time_ticks;assert(direct.prepare_real(frame,poses,12,plan));assert(direct.material_builds==0);
 renderer.unit_bodies.catalogue_generation++;
 ++frame.presentation_time_ticks;assert(direct.prepare_real(frame,poses,12,plan));assert(direct.material_builds==1);
}
''', sources=("Renderer/native/environment_runtime.cpp",))

    def test_material_gpu_slots_share_exact_values_across_passes_and_changes(self):
        host_cpp(direct_stub() + r'''
int main(){
 assert(ID3D11Buffer::live_heap==0);
 {
 Direct direct;setup(direct);c3x_renderer_frame_v1 frame{};frame.target_width=1000;frame.target_height=800;
 frame.tile_width=128;frame.tile_height=64;frame.presentation_frequency=1000;frame.presentation_time_ticks=1;
 std::vector<ScenePose> poses;UnitContributionPlan plan;
 for(unsigned i=0;i<438;++i){auto p=occurrence(i+1,100+i,300);p.draw.display_color_rgb=(i%9)*12000;
  poses.push_back(p);plan.entries.push_back({i,unit_main_body|unit_ground_shadow|unit_reflection});}
 assert(direct.prepare_real(frame,poses,12,plan));
 assert(direct.material_buffer_builds==9&&direct.material_buffer_uploads==9&&direct.material_buffer_reuses==429);
 auto creates=renderer.device->material_creates,uploads=renderer.context->material_uploads;
 std::vector<std::array<float,32>> expected;
 for(auto const& prepared:direct.prepared_units)expected.push_back(prepared.parts[0].material);
 Target target;assert(direct.draw_real(frame,poses,target,1,12,true,1));
 assert(direct.draw_real(frame,poses,target,1,12,false,1));
 assert(direct.material_upload_fallbacks==0&&renderer.context->material_uploads==uploads);
 assert(renderer.context->bodies==876&&renderer.context->shadows==438);
 assert(renderer.context->body_materials.size()==876);
 for(unsigned pass=0;pass<2;++pass)for(unsigned i=0;i<438;++i)
  assert(renderer.context->body_materials[pass*438+i]==expected[i]);
 // Movement, direction, frame selection and zoom preserve exact material ownership.
 for(auto& pose:poses){pose.draw.body_x+=17;pose.draw.body_y-=8;pose.draw.direction=2;pose.draw.action_cursor=3;}
 ++frame.presentation_time_ticks;assert(direct.prepare_real(frame,poses,12,plan));
 assert(direct.material_buffer_builds==0&&direct.material_buffer_uploads==0&&direct.material_buffer_reuses==438);
 assert(direct.draw_real(frame,poses,target,1,12,true,2));assert(direct.draw_real(frame,poses,target,1,12,false,2));
 assert(renderer.device->material_creates==creates&&renderer.context->material_uploads==uploads);
 // Lighting and tint/catalogue changes select new exact values; draw sees every changed byte.
 for(auto& p:poses)p.draw.display_color_rgb^=0x404040;
 renderer.unit_bodies.catalogue_generation++;
 ++frame.presentation_time_ticks;assert(direct.prepare_real(frame,poses,19,plan));
 assert(direct.material_buffer_uploads==9);
 renderer.context->body_materials.clear();assert(direct.draw_real(frame,poses,target,1,19,false,1));
 for(unsigned i=0;i<438;++i)assert(renderer.context->body_materials[i]==direct.prepared_units[i].parts[0].material);
 }
 assert(ID3D11Buffer::live_heap==0);
}
''', sources=("Renderer/native/environment_runtime.cpp",))

    def test_material_gpu_slot_overflow_replacement_and_failure_keep_all_draws(self):
        host_cpp(direct_stub() + r'''
int main(){
 {
 Direct direct;setup(direct);c3x_renderer_frame_v1 frame{};frame.target_width=1000;frame.target_height=800;
 frame.tile_width=128;frame.tile_height=64;frame.presentation_frequency=1000;frame.presentation_time_ticks=1;
 std::vector<ScenePose> poses;UnitContributionPlan plan;
 for(unsigned i=0;i<300;++i){auto p=occurrence(i+1,500,400);p.draw.display_color_rgb=i;
  poses.push_back(p);plan.entries.push_back({i,unit_main_body|unit_ground_shadow|unit_reflection});}
 assert(direct.prepare_real(frame,poses,12,plan));
 assert(direct.material_samples.size()==256&&direct.material_buffer_builds==256&&direct.material_buffer_uploads==256);
 assert(ID3D11Buffer::live_heap==256);
 for(unsigned i=0;i<300;++i)assert(bool(direct.prepared_units[i].parts[0].material_buffer)==(i<256));
 Target target;assert(direct.draw_real(frame,poses,target,1,12,true,1));assert(direct.draw_real(frame,poses,target,1,12,false,1));
 // Uncached materials upload separately for reflection, ground shadow and body.
 assert(direct.material_upload_fallbacks==132&&renderer.context->bodies==600&&renderer.context->shadows==300);
 for(unsigned pass=0;pass<2;++pass)for(unsigned i=0;i<300;++i)
  assert(renderer.context->body_materials[pass*300+i]==direct.prepared_units[i].parts[0].material);
 // Next frame may replace unpinned slots, reusing GPU buffers instead of allocating.
 for(auto& p:poses)p.draw.display_color_rgb+=1000;
 ++frame.presentation_time_ticks;assert(direct.prepare_real(frame,poses,12,plan));
 assert(direct.material_buffer_builds==0&&direct.material_buffer_uploads==256&&ID3D11Buffer::live_heap==256);
 renderer.context->body_materials.clear();assert(direct.draw_real(frame,poses,target,1,12,false,1));
 for(unsigned i=0;i<300;++i)assert(renderer.context->body_materials[i]==direct.prepared_units[i].parts[0].material);
 }
 assert(ID3D11Buffer::live_heap==0);
 {
 Direct direct;setup(direct);renderer.device->fail_material=true;
 c3x_renderer_frame_v1 frame{};frame.target_width=1000;frame.target_height=800;
 frame.tile_width=128;frame.tile_height=64;frame.presentation_frequency=1000;
 std::vector<ScenePose> poses={occurrence(1,500,400)};UnitContributionPlan plan;plan.entries={{0,7}};
 assert(direct.prepare_real(frame,poses,12,plan));assert(!direct.prepared_units[0].parts[0].material_buffer);
 assert(direct.material_buffer_builds==0&&direct.material_buffer_uploads==0);
 Target target;auto bodies=renderer.context->bodies;
 assert(direct.draw_real(frame,poses,target,1,12,true,1));assert(direct.draw_real(frame,poses,target,1,12,false,1));
 assert(renderer.context->bodies==bodies+2&&direct.material_upload_fallbacks==3);
 ++frame.presentation_time_ticks;assert(direct.prepare_real(frame,poses,12,plan));
 assert(direct.material_builds==0&&direct.material_buffer_builds==0);
 }
 assert(ID3D11Buffer::live_heap==0);
}
''', sources=("Renderer/native/environment_runtime.cpp",))

    def test_actual_select_uses_exact_native_center_and_no_ground_query(self):
        host_cpp(direct_stub() + r'''
int main(){Direct direct;setup(direct);c3x_renderer_frame_v1 frame{};frame.target_width=1000;frame.target_height=800;
 frame.tile_width=128;frame.tile_height=64;frame.world_wrap_x=1;frame.world_width_tiles=80;
 assert(renderer.unit_bodies.remember_contribution_bounds(0,true));
 std::vector<ScenePose> poses={occurrence(1,500,400),occurrence(1,5620,400)};
 UnitContributionPlan plan;assert(direct.select_real(frame,poses,12,1,{},plan));
 assert(plan.entries.size()==1&&plan.entries[0].candidate==0&&direct.ground_queries==0);
 auto copy=plan.required(poses);assert(copy[0].draw.body_x==436&&poses[1].draw.body_x==5556);
 renderer.unit_bodies.meshes[0].animation.reset();assert(direct.select_real(frame,poses,12,1,{},plan));assert(plan.entries.size()==1);
 poses[0].draw.direction=0;assert(!direct.select_real(frame,poses,12,1,{},plan)&&!plan.valid);
}
''', sources=("Renderer/native/environment_runtime.cpp",))

    def test_actual_immutable_pose_reuse_touches_state_and_preserves_action_blend(self):
        host_cpp(BASE + r"""
int main(){
 auto m=fixture();m.rig.binding[0]=9;m.rig.parents={-1};m.rig.skin_joints={0};
 m.rig.inverse_bind={JointMatrix{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1}};
 JointPose p;p.scale={1,0,0,0,1,0,0,0,1};p.rotation={0,0,0,1};p.position={1,0,0};
 m.rig.poses={p,p};m.rig.poses[1].position[0]=4;
 UnitPoseTransitions transitions;
 assert(!transitions.sample(1,7,1,0,1000,m,0,11));transitions.finish(0);assert(transitions.size()==1);
 for(long long ticks:{10,20,30}){
  assert(!transitions.sample(1,7,1,ticks,1000,m,0,11));transitions.finish(ticks);assert(transitions.size()==1);
 }
 // Reuse must touch ticks or finish() would erase the remembered local pose,
 // making this next action snap directly to its destination.
 auto palette=transitions.sample(1,7,2,40,1000,m,1,22);assert(palette&&std::abs(palette[12]-1)<1e-5);
 transitions.finish(40);
 palette=transitions.sample(1,7,2,100,1000,m,1,22);assert(palette&&std::abs(palette[12]-2.5f)<1e-4);
 assert(!transitions.sample(1,7,2,160,1000,m,1,22));transitions.finish(160);assert(transitions.size()==1);
 // A rollback/frequency change resets interpolation, never reuses stale time.
 assert(!transitions.sample(1,7,2,0,2000,m,1,22));transitions.finish(0);assert(transitions.size()==1);
}
""")

    def test_actual_sample_limits_preserve_frame_union_and_scratch_overflow(self):
        host_cpp(direct_stub() + r"""
int main(){Direct direct;setup(direct);c3x_renderer_frame_v1 frame{};frame.target_width=1000;frame.target_height=800;
 frame.tile_width=128;frame.tile_height=64;frame.presentation_frequency=1000;frame.presentation_time_ticks=1;
 auto original=renderer.unit_bodies.units[0];renderer.unit_bodies.units.resize(65,original);
 std::vector<ScenePose> poses;UnitContributionPlan plan;
 for(unsigned i=0;i<65;++i){auto p=occurrence(int(i+1),500,400);p.unit=i;
  renderer.unit_bodies.units[i].scale=1+i*.001f;poses.push_back(p);plan.entries.push_back({i,unit_main_body});}
 assert(direct.prepare_real(frame,poses,12,plan));assert(direct.shared_shadows.size()==64&&direct.shadow_overflow==1);
 Target target;assert(direct.draw_real(frame,poses,target,1,12,false,1));assert(direct.self_maps==65);
 ++frame.presentation_time_ticks;assert(direct.prepare_real(frame,poses,12,plan));assert(direct.shadow_overflow==1);
 assert(direct.draw_real(frame,poses,target,1,12,false,1));assert(direct.shadow_reuses==64&&direct.self_maps==66);
 // A separate scalar material cache is hard bounded; overflow computes local
 // values without evicting another material borrowed in the current frame.
 poses.clear();plan.entries.clear();
 for(unsigned i=0;i<300;++i){auto p=occurrence(int(i+1),500,400);p.draw.display_color_rgb=i;
  poses.push_back(p);plan.entries.push_back({i,unit_main_body});}
 ++frame.presentation_time_ticks;assert(direct.prepare_real(frame,poses,12,plan));
 assert(direct.material_samples.size()==256&&direct.material_builds+direct.material_reuses==300);
 ++frame.presentation_time_ticks;assert(direct.prepare_real(frame,poses,12,plan));
 assert(direct.material_samples.size()==256&&direct.material_builds==44&&direct.material_reuses==256);
 assert(direct.material_sample_bytes()<256u*1024u);
}
""", sources=("Renderer/native/environment_runtime.cpp",))


if __name__ == "__main__":
    unittest.main()
