// Test scaffolding; the bridge functions are extracted from injected_code.c.
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>
#include "c3x_renderer_api.h"
#ifndef _MSC_VER
#define __fastcall
#endif
#define __ 0
using HDC=void*;
struct LARGE_INTEGER {long long QuadPart=0;};
struct Unit;struct PCX_Image;struct JGL_Color_Table;
struct JGLV {std::uintptr_t m04_Get_Palette_Colors;};
struct JGL_Color_Table {JGLV* vtable;};
struct PCX_Color_Table {JGL_Color_Table* JGL_Color_Table;};
struct Sprite {int Width=191,Height=191;};
struct JGL_Image;
struct PCXV {HDC (__fastcall *acquire_dc)(JGL_Image*);void (__fastcall *release_dc)(JGL_Image*,int,int);};
struct JGL_Image {PCXV* vtable;};
struct PCX_Image {struct {JGL_Image* Image;} JGL;};
struct Animation_Info {int* Frame_Counts;};
struct Summary {int current_anim_type=2,queued_anim_type=0,direction_2=3,pixel_loc_x=640,pixel_loc_y=480,pixel_target_x=720,pixel_target_y=520;};
struct Animation {struct {void* Flic_Info;Sprite sprite;} Frame_1;Animation_Info* Animation_Info;Summary summary;int field_FC=7;};
struct Rect {int left=20,top=30,right=45,bottom=55;};
using RECT=Rect;
struct Unit {struct {Rect Rect;int ID=42,UnitTypeID=0,X=2,Y=4,Damage=2;int army_top_defender_id=-1;Animation Animation;} Body;bool army=false,visible=true;};
struct UnitType {char Civilipedia_Entry[32]="PRTO_Archer";};
struct Bic {int UnitTypeCount=1;UnitType* UnitTypes;bool is_zoomed_out=false;};
struct State {int custom_renderer_native_operation=123;Unit* custom_renderer_unit_context=nullptr;PCX_Image* custom_renderer_unit_canvas=nullptr;
 c3x_renderer_unit_forget_fn custom_renderer_unit_forget=nullptr;
 c3x_renderer_unit_draw_background_fn custom_renderer_unit_draw=nullptr;
 c3x_renderer_unit_draw_playback_fn custom_renderer_unit_draw_playback=nullptr;
 c3x_renderer_unit_visual_fn custom_renderer_unit_visual=nullptr;
 c3x_renderer_unit_draw_expanded_fn custom_renderer_unit_draw_expanded=nullptr;int custom_renderer_init_state=1;
 struct {bool enable_custom_rendering=true,enable_custom_rendering_zoom=false;int day_night_cycle_mode=0,seasonal_cycle_mode=0;} current_config;
 int custom_renderer_zoom_tile_width=128,custom_renderer_zoom_native_tile_width=128;
 long long custom_renderer_zoom_translate_x_fp=0,custom_renderer_zoom_translate_y_fp=0;
 int custom_renderer_zoom_unit_tick_delta_x=0,custom_renderer_zoom_unit_tick_delta_y=0;
 bool custom_renderer_zoom_unit_tick_translated=false;
 bool day_night_cycle_unstarted=false,seasonal_cycle_unstarted=false;int current_day_night_cycle=12,current_seasonal_cycle=0;
 c3x_renderer_visual_clock_fn custom_renderer_visual_clock=nullptr;
 LARGE_INTEGER custom_renderer_qpc_frequency={1000000},custom_renderer_animation_timestamp={},custom_renderer_animation_sample_at={};};
constexpr int AT_DEFAULT=1,AT_PLANT=18,DNCM_OFF=0,SCM_OFF=0,CS_SUMMER=0,CS_SPRING=3,IS_OK=1,UTA_Army=1;
struct Screen {int Player_CivID=1;Unit* Current_Unit=nullptr;} screen;Screen* p_main_screen_form=&screen;
unsigned playback_flags=0;
State state;State* is=&state;Bic bic;Bic* p_bic_data=&bic;PCX_Color_Table fixture_palette;
std::vector<int> calls;c3x_renderer_unit_v1 captured;bool success=true,fixture_reduced=false;int dc_count=0;PCX_Image* fixture_background=nullptr;JGL_Image* denied_dc=nullptr;
c3x_renderer_unit_visual_v1 captured_visual{};int visual_calls=0;
int capture_visual(c3x_renderer_unit_visual_v1 const* value){assert(value&&value->struct_size==sizeof(*value));captured_visual=*value;++visual_calls;return 1;}
int Unit_get_max_hp(Unit*){return 4;}
int clamp(int a,int b,int v){return v<a?a:(v>b?b:v);}
long long qpc=1000000;
bool QueryPerformanceCounter(LARGE_INTEGER* value){value->QuadPart=qpc;qpc+=66000;return true;}
bool Unit_has_ability(Unit* u,int,int){return u->army;}
bool custom_renderer_zoom_enabled(){return state.current_config.enable_custom_rendering && state.current_config.enable_custom_rendering_zoom;}
void sync_custom_renderer_zoom_to_native(){}
bool custom_renderer_zoom_transform_active(){
 return custom_renderer_zoom_enabled() &&
  (state.custom_renderer_zoom_tile_width!=state.custom_renderer_zoom_native_tile_width ||
   state.custom_renderer_zoom_translate_x_fp!=0 || state.custom_renderer_zoom_translate_y_fp!=0);
}
void custom_renderer_zoom_transform_point(int* x,int* y){
 if(!custom_renderer_zoom_transform_active())return;
 auto transform=[](int value,long long translation){
  long long transformed=(long long)value*state.custom_renderer_zoom_tile_width*65536/
   state.custom_renderer_zoom_native_tile_width+translation;
  return (int)(transformed>=0?(transformed+32768)/65536:(transformed-32768)/65536);
 };
 *x=transform(*x,state.custom_renderer_zoom_translate_x_fp);
 *y=transform(*y,state.custom_renderer_zoom_translate_y_fp);
}
Unit* army_member=nullptr;Unit* get_unit_ptr(int id){return army_member && army_member->Body.ID==id?army_member:nullptr;}
HDC __fastcall acquire(JGL_Image* p){assert(state.custom_renderer_native_operation==C3X_NATIVE_SPRITE);if(p==denied_dc)return nullptr;++dc_count;return p;}
void __fastcall release_dc(JGL_Image*,int,int){assert(state.custom_renderer_native_operation==C3X_NATIVE_SPRITE);--dc_count;}
int __fastcall colors(JGL_Color_Table*,int,unsigned char* out,int first,int count){assert(first==6 && count==1);out[0]=12;out[1]=34;out[2]=56;return 0;}
int capture(c3x_renderer_unit_v1 const* value,void* destination,void* background){assert(destination && background && destination!=background && dc_count==2);captured=*value;calls.push_back(20);return success?1:0;}
int capture_expanded(c3x_renderer_unit_v1 const* value,void* destination,void* background,int* bounds){
 int result=capture(value,destination,background);
 if(result){bounds[0]=-40;bounds[1]=-60;bounds[2]=300;bounds[3]=280;}
 return result;
}
int capture_playback(c3x_renderer_unit_v1 const* value,void* destination,void* background,int* bounds,unsigned flags){
 playback_flags=flags;return capture_expanded(value,destination,background,bounds);
}
int __fastcall Sprite_draw_unit_body_normal(Sprite*,int,PCX_Image*,PCX_Image*,int x,int y,char* path,PCX_Color_Table* p){assert(x==11 && y==23 && path[0]=='p' && p==&fixture_palette);calls.push_back(30);return 77;}
int __fastcall original_reduced(Sprite*,int,PCX_Image*,PCX_Image*,int x,int y,int sx,int sy,int divisor,char* path,PCX_Color_Table* p){assert(x==11 && y==23 && sx==1 && sy==1 && divisor==2 && path[0]=='p' && p==&fixture_palette);calls.push_back(30);return 77;}
// Match the corrected nine-stack-argument native declaration.
auto Sprite_draw_unit_body_reduced=original_reduced;
void __fastcall Unit_tick_anim(Unit*,int,PCX_Image*,int,int,bool);

int translate_custom_renderer_native(int,JGL_Image*,void*,RECT*,RECT*,unsigned){return 0;}

#define this self
struct Tile{};Tile visibility_tile;bool tile_visible=true;
Tile* tile_at(int,int){return &visibility_tile;}
unsigned capture_custom_renderer_visibility(Tile*,int,int,int){return C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED|(tile_visible?C3X_RENDERER_TILE_VISIBLE:0);}
#include "build/unit_bridge_capture.h"
#undef this
void __fastcall Unit_tick_anim(Unit* u,int,PCX_Image* canvas,int x,int y,bool status){
 if(state.custom_renderer_zoom_unit_tick_translated){
  int width=state.custom_renderer_zoom_tile_width;
  assert(x==640-(539*width+64)/128 && y==480-(278*width+64)/128);
 }
 else {assert(x==101 && y==202);}
 assert(status);if(!u->visible)return;
 u->Body.Rect={};
 calls.push_back(10);
 for(int child=0;child<(u->army && army_member?2:1);++child){
  Unit* body=child?army_member:u;
  int body_x=11+state.custom_renderer_zoom_unit_tick_delta_x;
  int body_y=23+state.custom_renderer_zoom_unit_tick_delta_y;
  if(fixture_reduced)patch_Sprite_draw_unit_body_reduced(&body->Body.Animation.Frame_1.sprite,0,fixture_background,canvas,body_x,body_y,1,1,2,const_cast<char*>("palette"),&fixture_palette);
  else patch_Sprite_draw_unit_body_normal(&body->Body.Animation.Frame_1.sprite,0,fixture_background,canvas,body_x,body_y,const_cast<char*>("palette"),&fixture_palette);
 }
 calls.push_back(40);
}
int main(){
 UnitType type;bic.UnitTypes=&type;int counts[19];for(auto & c:counts)c=16;Animation_Info info={counts};
 Unit unit;unit.Body.Animation.Animation_Info=&info;unit.Body.Animation.Frame_1.Flic_Info=&info;
 JGLV table_v={reinterpret_cast<std::uintptr_t>(colors)};JGL_Color_Table table={&table_v};fixture_palette.JGL_Color_Table=&table;
 PCXV canvas_v={acquire,release_dc};JGL_Image image={&canvas_v};PCX_Image canvas={{&image}},other={{&image}};
 JGL_Image background_image={&canvas_v};PCX_Image background={{&background_image}};fixture_background=&background;
 state.custom_renderer_unit_draw=capture;
 state.custom_renderer_unit_visual=capture_visual;
 auto invoke=[&](){calls.clear();patch_Unit_tick_anim(&unit,0,&canvas,101,202,true);assert(state.custom_renderer_native_operation==123 && dc_count==0 && !state.custom_renderer_unit_context && !state.custom_renderer_unit_canvas);};
 success=true;invoke();assert(captured.presentation_time_ticks==0);
 assert(visual_calls==1 && captured_visual.unit_id==42 && captured_visual.action==2);
 assert(captured_visual.pixel_x==640 && captured_visual.pixel_y==480 && captured_visual.target_x==720 && captured_visual.target_y==520);
 assert(captured_visual.damage==2 && captured_visual.max_hp==4 && captured_visual.body_x==captured.body_x && captured_visual.body_y==captured.body_y);
 assert(captured_visual.presentation_time_ticks==captured.presentation_time_ticks);
 invoke();assert(captured.presentation_time_ticks==66000);
 qpc+=1000000;invoke();assert(captured.presentation_time_ticks==66000); // interturn wall time is frozen
 for(int zoom=0;zoom<2;++zoom){
  fixture_reduced=zoom!=0;success=true;invoke();assert((calls==std::vector<int>{10,20,40}));
  assert(unit.Body.Rect.left==11 && unit.Body.Rect.top==23);
  assert(unit.Body.Rect.right==11+191/(zoom?2:1) && unit.Body.Rect.bottom==23+191/(zoom?2:1));
  assert(captured.unit_id==42 && captured.action==2 && captured.action_cursor==7 && captured.frame_count==16);
  assert(captured.body_x==11 && captured.body_y==23 && captured.direction==3 && captured.reduced==zoom);
  assert(captured.projection_scale_milli==(zoom?500:1000));
  assert(captured.presentation_frequency==1000000 && captured.presentation_time_ticks>=0);
  assert(captured.display_color_rgb==0x0c2238 && !std::strcmp(captured.unit_key,"PRTO_Archer"));
  success=false;invoke();assert(unit.Body.Rect.left==20 && unit.Body.Rect.right==45);assert((calls==std::vector<int>{10,20,40}));
  denied_dc=&image;invoke();assert((calls==std::vector<int>{10,40}));
  denied_dc=&background_image;invoke();assert((calls==std::vector<int>{10,40}));denied_dc=nullptr;
  state.current_config.enable_custom_rendering=false;invoke();assert((calls==std::vector<int>{10,30,40}));
  state.current_config.enable_custom_rendering=true;unit.army=true;success=true;
  invoke();assert((calls==std::vector<int>{10,20,40}));
  Unit member=unit;member.army=false;member.Body.ID=84;member.Body.Animation.field_FC=4;
  member.Body.Rect={200,200,210,210};army_member=&member;unit.Body.army_top_defender_id=84;invoke();assert((calls==std::vector<int>{10,20,20,40}));
  assert(captured.unit_id==84 && captured.action_cursor==4);
  assert(member.Body.Rect.left==200 && member.Body.Rect.right==210); // Only parent's native dirty bounds own redraw.
  unit.Body.army_top_defender_id=-1;invoke();assert((calls==std::vector<int>{10,20,40}));
  army_member=nullptr;unit.army=false;
  unit.visible=false;invoke();assert(calls.empty());unit.visible=true;
 }
 state.custom_renderer_unit_draw_expanded=capture_expanded;
 success=true;invoke();assert(unit.Body.Rect.left==-40 && unit.Body.Rect.top==-60 && unit.Body.Rect.right==300 && unit.Body.Rect.bottom==280);
 success=false;invoke();assert(unit.Body.Rect.left==20 && unit.Body.Rect.right==45);
 state.custom_renderer_unit_draw_expanded=nullptr;
 state.custom_renderer_unit_draw_playback=capture_playback;success=true;
 invoke();assert(playback_flags==C3X_RENDERER_UNIT_STATE_CAPTURED);
 tile_visible=false;playback_flags=0;invoke();assert(playback_flags==0);tile_visible=true;
 screen.Current_Unit=&unit;invoke();assert(playback_flags==(C3X_RENDERER_UNIT_STATE_CAPTURED|C3X_RENDERER_UNIT_SELECTED));
 unit.army=true;Unit member=unit;member.army=false;member.Body.ID=84;member.Body.Rect={};army_member=&member;unit.Body.army_top_defender_id=84;
 invoke();assert(captured.unit_id==84 && playback_flags==(C3X_RENDERER_UNIT_STATE_CAPTURED|C3X_RENDERER_UNIT_SELECTED));
 assert(unit.Body.Rect.left==-40 && member.Body.Rect.left==20);
 army_member=nullptr;unit.army=false;screen.Current_Unit=nullptr;
 state.custom_renderer_unit_draw_playback=nullptr;
 // UI portraits use these same native hooks, outside the map tick's canvas.
 // They must preserve native arguments/return values at every custom zoom.
 state.current_config.enable_custom_rendering_zoom=true;
 for(int width:{64,96,128,160,192})for(bool custom_units:{false,true}){
  state.custom_renderer_zoom_tile_width=width;
  state.current_config.enable_custom_rendering=custom_units;
  for(bool nested:{false,true}){
   state.custom_renderer_unit_context=nested?&unit:nullptr;
   state.custom_renderer_unit_canvas=nested?&canvas:nullptr;
   PCX_Image* portrait=nested?&other:&canvas;
   calls.clear();
   assert(patch_Sprite_draw_unit_body_normal(&unit.Body.Animation.Frame_1.sprite,0,&background,portrait,11,23,const_cast<char*>("palette"),&fixture_palette)==77);
   assert(patch_Sprite_draw_unit_body_reduced(&unit.Body.Animation.Frame_1.sprite,0,&background,portrait,11,23,1,1,2,const_cast<char*>("palette"),&fixture_palette)==77);
   assert((calls==std::vector<int>{30,30}) && dc_count==0);
  }
 }
 state.custom_renderer_unit_context=nullptr;state.custom_renderer_unit_canvas=nullptr;
 state.current_config.enable_custom_rendering=true;
 fixture_reduced=false;state.current_config.enable_custom_rendering_zoom=true;
 state.custom_renderer_zoom_tile_width=80;state.custom_renderer_zoom_native_tile_width=128;
 success=true;invoke();assert((calls==std::vector<int>{10,20,40}));
 assert(captured.body_x==7 && captured.body_y==14 && captured.projection_scale_milli==625);
 assert(unit.Body.Rect.left==7 && unit.Body.Rect.top==14);
 assert(unit.Body.Rect.right==7+191*625/1000 && unit.Body.Rect.bottom==14+191*625/1000);
 success=false;invoke();assert((calls==std::vector<int>{10,20,40}));
 state.current_config.enable_custom_rendering=false;invoke();assert((calls==std::vector<int>{10,30,40}));
 state.current_config.enable_custom_rendering=true;
 for(int width:{64,96,128,160,192})for(bool reduced:{false,true}){
  fixture_reduced=reduced;
  state.custom_renderer_zoom_tile_width=width;success=true;invoke();
  int scale=width*1000/128;
  assert(captured.projection_scale_milli==scale);
  assert(captured.body_x==(11*width+64)/128 && captured.body_y==(23*width+64)/128);
  assert(unit.Body.Rect.left<=captured.body_x && unit.Body.Rect.top<=captured.body_y);
  assert(unit.Body.Rect.right>=captured.body_x+191*scale/1000);
  assert(unit.Body.Rect.bottom>=captured.body_y+191*scale/1000);
  success=false;invoke();assert((calls==std::vector<int>{10,20,40}));
  state.current_config.enable_custom_rendering=false;invoke();assert((calls==std::vector<int>{10,30,40}));
  state.current_config.enable_custom_rendering=true;
  state.custom_renderer_unit_draw=nullptr;invoke();assert((calls==std::vector<int>{10,40}));
  state.custom_renderer_unit_draw=capture;
  state.custom_renderer_init_state=0;invoke();assert((calls==std::vector<int>{10,40}));
  state.custom_renderer_init_state=IS_OK;
 }
 state.current_config.enable_custom_rendering_zoom=false;
 state.custom_renderer_unit_context=&unit;state.custom_renderer_unit_canvas=&canvas;
 calls.clear();Sprite unrelated;assert(!forward_custom_unit_body(&unrelated,&canvas,&canvas,0,0,0,&fixture_palette));
 assert(!forward_custom_unit_body(&unit.Body.Animation.Frame_1.sprite,&canvas,&other,0,0,0,&fixture_palette));assert(calls.empty());
 // Scoped context restores an outer callback's exact previous values.
 success=true;patch_Unit_tick_anim(&unit,0,&other,101,202,true);
 assert(state.custom_renderer_unit_context==&unit && state.custom_renderer_unit_canvas==&canvas);
 // A later native capture takes the same clock as independent GPU frames.
 state.custom_renderer_visual_clock=+[]()->c3x_renderer_i64{return 9000000;};
 state.custom_renderer_unit_context=nullptr;state.custom_renderer_unit_canvas=nullptr;
 invoke();assert(captured.presentation_time_ticks==9000000);

}
