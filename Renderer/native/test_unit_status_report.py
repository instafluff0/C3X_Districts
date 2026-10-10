"""Execute the injected unit status report: draw_status's rules as renderer facts.

Stage 4.3 (performance review, section 47): the renderer draws a unit's map
status itself, at native pixel size, from facts the injected hook reports.
These cases mirror Unit::draw_status (0x5BA750) and its movement LED choice.
"""
import unittest
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.test_zoom_integration import function


class UnitStatusReportTests(unittest.TestCase):
    def test_draw_status_rules_become_renderer_facts(self):
        source = (ROOT / 'injected_code.c').read_text()
        report = 'bool ' + function(source, 'report_custom_renderer_unit_status')
        run_cpp(r'''
#pragma clang diagnostic ignored "-Wc++11-narrowing"
#include <cassert>
#include <cstring>
#include "Renderer/native/c3x_renderer_api.h"
constexpr int __=0;
enum {UTC_Land=0,UTC_Sea=1,UnitState_Fortifying=1,UTA_Army=7};
struct PCX_Image;struct PCX_Vtable{void(*clear_JGL)(PCX_Image*);void(*destruct)(PCX_Image*,int,int);};
struct JGL{void* Image=nullptr;};
struct PCX_Image{PCX_Vtable* vtable;JGL JGL;};
struct Sprite{void* vtable=nullptr;void* jgl_sprite=nullptr;int sliced_x=-1;};
struct Tile;struct TileVtable{bool(*m35_Check_Is_Water)(Tile*);};
struct Tile{TileVtable* vtable;bool water=false;};
struct Body{int ID=1,X=4,Y=5,CivID=1,UnitTypeID=0,Damage=0,Moves=0,Container_Unit=-1,UnitState=0;short Active=0;};
struct Unit{Body Body;int attack=1,defense=1,max_hp=3,max_moves=3;bool army=false,visible=true;};
struct UnitType{int Unit_Class=UTC_Land;};
struct{UnitType UnitTypes[2];struct{int DefenceBonus_Fortification=25;}General;} bic,*p_bic_data=&bic;
struct{int Player_CivID=1;} screen,*p_main_screen_form=&screen;
Unit units[12];int stacked=0;Tile land{nullptr,false};
bool tile_water(Tile* t){return t->water;}TileVtable tile_vtable{tile_water};
Tile* tile_at(int,int){land.vtable=&tile_vtable;return &land;}
Unit* get_unit_ptr(int id){return id>=0&&id<12?&units[id]:nullptr;}
bool Unit_has_ability(Unit* u,int,int ability){return ability==UTA_Army&&u->army;}
int Unit_get_max_hp(Unit* u){return u->max_hp;}int Unit_get_max_move_points(Unit* u){return u->max_moves;}
int Unit_get_attack_strength(Unit* u){return u->attack;}int Unit_get_defense_strength(Unit* u){return u->defense;}
bool patch_Unit_is_visible_to_civ(Unit* u,int,int civ,int){assert(civ==screen.Player_CivID);return u->visible;}
struct unit_tile_iter{int id;Unit* unit;};
unit_tile_iter uti_init(Tile*){return {stacked>0?2:-1,stacked>0?&units[2]:nullptr};}
void uti_next(unit_tile_iter* it){it->id=it->id+1<2+stacked?it->id+1:-1;it->unit=it->id>=0?&units[it->id]:nullptr;}
#define FOR_UNITS_ON(uti_name, tile) for (unit_tile_iter uti_name = uti_init (tile); uti_name.id != -1; uti_next (&uti_name))
int clamp(int lo,int hi,int v){return v<lo?lo:v>hi?hi:v;}int not_above(int lim,int v){return v>lim?lim:v;}
int loads=0,clears=0,destructs=0;bool file_ok=true;
void clear(PCX_Image*){++clears;}void destruct(PCX_Image*,int,int){++destructs;}PCX_Vtable pcx_vtable{clear,destruct};
void PCX_Image_construct(PCX_Image* p){p->vtable=&pcx_vtable;p->JGL.Image=nullptr;}
char asset[]="art/interface/MovementLED.pcx";
char* BIC_get_asset_path(void*,int,char const* name,bool){assert(!std::strcmp(name,"art\\interface\\MovementLED.pcx"));return asset;}
int PCX_Image_read_file(PCX_Image* p,int,char*,void*,int,int,int){++loads;if(!file_ok)return 1;static int pixels;p->JGL.Image=&pixels;return 0;}
void Sprite_construct(Sprite* s){*s={};}
int Sprite_slice_pcx(Sprite* s,int,PCX_Image*,int x,int y,int w,int h,int,int){assert(y==1&&w==6&&h==6);s->sliced_x=x;s->jgl_sprite=&s->sliced_x;return 1;}
c3x_renderer_unit_status_v1 sent;void const* sent_led=nullptr;int calls=0,answer=1;
int native_image(int op,void* image,void*,void const* from,void const* to,unsigned){
 assert(op==C3X_NATIVE_UNIT_STATUS&&image==&land);++calls;sent=*static_cast<c3x_renderer_unit_status_v1 const*>(from);
 sent_led=to;return answer;}
struct State{int(*custom_renderer_native_image)(int,void*,void*,void const*,void const*,unsigned)=native_image;
 Sprite custom_renderer_movement_leds[3];int custom_renderer_movement_leds_state=0;} state,*is=&state;
''' + report.replace('canvas->JGL.Image', '(void*)&land') + r'''
int main(){
 PCX_Image canvas{&pcx_vtable,{}};auto& u=units[0];u.Body.Moves=0;
 auto led=[&](int n){return state.custom_renderer_movement_leds[n].jgl_sprite;};
 auto status=[&](bool stack=false){calls=0;bool drawn=report_custom_renderer_unit_status(&u,&canvas,100,200,stack);
  assert(drawn==(answer==1));return calls;};
 // Fresh unit: full bar, green LED (the JGL sprite each Sprite wraps), loaded once like Civ III's.
 assert(status()==1&&sent.x==100&&sent.y==200&&sent.max_hp==3&&sent.damage==0);
 assert(sent.flags==C3X_RENDERER_UNIT_STATUS_BAR&&sent.stack==0&&sent_led==led(0));
 assert(loads==1&&clears==1&&destructs==1&&state.custom_renderer_movement_leds[2].sliced_x==29);
 status();assert(loads==1);
 // LED: partial moves yellow, none red, none while moving (Active high byte) yellow; other civs none.
 u.Body.Moves=1;status();assert(sent_led==led(1));
 u.Body.Moves=3;status();assert(sent_led==led(2));
 u.Body.Moves=5;status();assert(sent_led==led(2));
 u.Body.Moves=3;u.Body.Active=0x100;status();assert(sent_led==led(1));
 u.Body.Active=0;u.Body.CivID=2;status();assert(!sent_led);u.Body.CivID=1;u.Body.Moves=0;
 // Fortified outline: land, dry tile, fortified, a move left, a fortification bonus.
 u.Body.UnitState=UnitState_Fortifying;status();assert(sent.flags==(C3X_RENDERER_UNIT_STATUS_BAR|C3X_RENDERER_UNIT_STATUS_FORTIFIED));
 u.Body.Moves=3;status();assert(sent.flags==C3X_RENDERER_UNIT_STATUS_BAR);u.Body.Moves=0;
 land.water=true;status();assert(sent.flags==C3X_RENDERER_UNIT_STATUS_BAR);land.water=false;
 bic.UnitTypes[0].Unit_Class=UTC_Sea;status();assert(sent.flags==C3X_RENDERER_UNIT_STATUS_BAR);bic.UnitTypes[0].Unit_Class=UTC_Land;
 bic.General.DefenceBonus_Fortification=0;status();assert(sent.flags==C3X_RENDERER_UNIT_STATUS_BAR);bic.General.DefenceBonus_Fortification=25;
 // No attack or defence: no bar and no outline (the LED remains).
 u.attack=u.defense=0;status();assert(sent.flags==0&&sent_led);u.attack=u.defense=1;u.Body.UnitState=0;
 // Stack marks: other visible units on the tile, only when asked, capped at 8.
 stacked=1;status(true);assert(sent.stack==2);status(false);assert(sent.stack==0);
 units[2].visible=false;status(true);assert(sent.stack==0);units[2].visible=true;
 stacked=10;status(true);assert(sent.stack==8);stacked=0;
 // Army members draw nothing; the renderer may decline and native draws.
 u.Body.Container_Unit=5;units[5].army=true;calls=0;assert(report_custom_renderer_unit_status(&u,&canvas,0,0,false)&&calls==0);
 u.Body.Container_Unit=-1;answer=0;status();answer=1;
 // An unreadable LED file: statuses still report, without an LED.
 state.custom_renderer_movement_leds_state=0;file_ok=false;status();assert(state.custom_renderer_movement_leds_state==-1&&!sent_led);
}
''')


if __name__ == '__main__':
    unittest.main()
