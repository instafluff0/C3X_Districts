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
struct Summary {int current_anim_type=2,queued_anim_type=0,direction_2=3;};
struct Animation {struct {void* Flic_Info;Sprite sprite;} Frame_1;Animation_Info* Animation_Info;Summary summary;int field_FC=7;};
struct Unit {struct {int ID=42,UnitTypeID=0;Animation Animation;} Body;bool army=false,visible=true;};
struct UnitType {char Civilipedia_Entry[32]="PRTO_Archer";};
struct Bic {int UnitTypeCount=1;UnitType* UnitTypes;};
struct State {Unit* custom_renderer_unit_context=nullptr;PCX_Image* custom_renderer_unit_canvas=nullptr;
 c3x_renderer_unit_draw_fn custom_renderer_unit_draw=nullptr;int custom_renderer_init_state=1;
 struct {bool enable_custom_rendering=true,enable_custom_rendered_units=true;int day_night_cycle_mode=0,seasonal_cycle_mode=0;} current_config;
 bool day_night_cycle_unstarted=false,seasonal_cycle_unstarted=false;int current_day_night_cycle=12,current_seasonal_cycle=0;};
constexpr int AT_DEFAULT=1,AT_PLANT=18,DNCM_OFF=0,SCM_OFF=0,CS_SUMMER=0,CS_SPRING=3,IS_OK=1,UTA_Army=1;
State state;State* is=&state;Bic bic;Bic* p_bic_data=&bic;PCX_Color_Table fixture_palette;
std::vector<int> calls;c3x_renderer_unit_v1 captured;bool success=true,fixture_reduced=false;int dc_count=0;
int clamp(int a,int b,int v){return v<a?a:(v>b?b:v);}
bool Unit_has_ability(Unit* u,int,int){return u->army;}
HDC __fastcall acquire(JGL_Image* p){++dc_count;return p;}
void __fastcall release_dc(JGL_Image*,int,int){--dc_count;}
int __fastcall colors(JGL_Color_Table*,int,unsigned char* out,int first,int count){assert(first==6 && count==1);out[0]=12;out[1]=34;out[2]=56;return 0;}
int capture(c3x_renderer_unit_v1 const* value,void*){captured=*value;calls.push_back(20);return success?1:0;}
int __fastcall Sprite_draw_unit_body_normal(Sprite*,int,PCX_Image*,PCX_Image*,int x,int y,char* path,PCX_Color_Table* p){assert(x==11 && y==23 && path[0]=='p' && p==&fixture_palette);calls.push_back(30);return 77;}
int __fastcall original_reduced(Sprite*,int,PCX_Image*,PCX_Image*,int x,int y,int sx,int sy,int divisor,char* path,PCX_Color_Table* p){assert(x==11 && y==23 && sx==1 && sy==1 && divisor==2 && path[0]=='p' && p==&fixture_palette);calls.push_back(30);return 77;}
// Mirror the currently incorrect callable declaration, then exercise the audited cast.
using OldReduced=int (__fastcall *)(Sprite*,int,PCX_Image*,PCX_Image*,int,int);
OldReduced Sprite_draw_unit_body_reduced=reinterpret_cast<OldReduced>(original_reduced);
void __fastcall Unit_tick_anim(Unit*,int,PCX_Image*,int,int,bool);

#define this self
#include "build/unit_bridge_capture.h"
#undef this
void __fastcall Unit_tick_anim(Unit* u,int,PCX_Image* canvas,int x,int y,bool status){
 assert(x==101 && y==202 && status);if(!u->visible)return;
 calls.push_back(10);
 for(int child=0;child<(u->army?2:1);++child){
  if(fixture_reduced)patch_Sprite_draw_unit_body_reduced(&u->Body.Animation.Frame_1.sprite,0,canvas,canvas,11,23,1,1,2,const_cast<char*>("palette"),&fixture_palette);
  else patch_Sprite_draw_unit_body_normal(&u->Body.Animation.Frame_1.sprite,0,canvas,canvas,11,23,const_cast<char*>("palette"),&fixture_palette);
 }
 calls.push_back(40);
}
int main(){
 UnitType type;bic.UnitTypes=&type;int counts[19];for(auto & c:counts)c=16;Animation_Info info={counts};
 Unit unit;unit.Body.Animation.Animation_Info=&info;unit.Body.Animation.Frame_1.Flic_Info=&info;
 JGLV table_v={reinterpret_cast<std::uintptr_t>(colors)};JGL_Color_Table table={&table_v};fixture_palette.JGL_Color_Table=&table;
 PCXV canvas_v={acquire,release_dc};JGL_Image image={&canvas_v};PCX_Image canvas={{&image}},other={{&image}};
 state.custom_renderer_unit_draw=capture;
 auto invoke=[&](){calls.clear();patch_Unit_tick_anim(&unit,0,&canvas,101,202,true);assert(dc_count==0 && !state.custom_renderer_unit_context && !state.custom_renderer_unit_canvas);};
 for(int zoom=0;zoom<2;++zoom){
  fixture_reduced=zoom!=0;success=true;invoke();assert((calls==std::vector<int>{10,20,40}));
  assert(captured.unit_id==42 && captured.action==2 && captured.action_cursor==7 && captured.frame_count==16);
  assert(captured.body_x==11 && captured.body_y==23 && captured.direction==3 && captured.reduced==zoom);
  assert(captured.display_color_rgb==0x0c2238 && !std::strcmp(captured.unit_key,"PRTO_Archer"));
  success=false;invoke();assert((calls==std::vector<int>{10,20,30,40}));
  state.current_config.enable_custom_rendered_units=false;invoke();assert((calls==std::vector<int>{10,30,40}));
  state.current_config.enable_custom_rendered_units=true;unit.army=true;invoke();assert((calls==std::vector<int>{10,30,30,40}));unit.army=false;
  unit.visible=false;invoke();assert(calls.empty());unit.visible=true;
 }
 state.custom_renderer_unit_context=&unit;state.custom_renderer_unit_canvas=&canvas;
 calls.clear();Sprite unrelated;assert(!forward_custom_unit_body(&unrelated,&canvas,0,0,0,&fixture_palette));
 assert(!forward_custom_unit_body(&unit.Body.Animation.Frame_1.sprite,&other,0,0,0,&fixture_palette));assert(calls.empty());
 // Scoped context restores an outer callback's exact previous values.
 success=true;patch_Unit_tick_anim(&unit,0,&other,101,202,true);
 assert(state.custom_renderer_unit_context==&unit && state.custom_renderer_unit_canvas==&canvas);
}
