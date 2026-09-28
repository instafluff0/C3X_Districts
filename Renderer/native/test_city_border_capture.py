"""Execute the native border capture and config-off delegate with fake game state."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp
ROOT=Path(__file__).resolve().parents[2]

class CityBorderCaptureTests(unittest.TestCase):
    def test_native_edge_palette_wrap_and_visibility_rules(self):
        source=(ROOT/'injected_code.c').read_text()
        start=source.index('\t\t\t// Match Map_Renderer::m17 exactly:')
        stop=source.index('\t\t\tif (record->real_terrain_type',start)
        body=source[start:stop]
        run_cpp(r'''
#include <cassert>
#include <cstddef>
#define __fastcall
const int __=0;
struct Map_Renderer;struct JGL_Color_Table;struct Tile;
struct RenderVT{char (*m05_Check_dword_9AFD34)(Map_Renderer*,int);};
struct Map_Renderer{RenderVT* vtable;unsigned Flags=0;};
struct PaletteVT{int (*m04_Get_Palette_Colors)(JGL_Color_Table*,int,unsigned char*,int,int);};
struct JGL_Color_Table{PaletteVT* vtable;};
struct PCX_Color_Table{struct JGL_Color_Table* JGL_Color_Table;};
struct Units_Image_Data{PCX_Color_Table* Color_Tables[32]={};};
struct State{Units_Image_Data* custom_renderer_unit_images=nullptr;}state,*is=&state;
struct Leader{int Color_Table_ID=0;}leaders[32];
struct Map{Map_Renderer Renderer;int Flags=0,Width=8,Height=8;};
struct Bic{struct Map Map;}bic,*p_bic_data=&bic;
struct TileVT{int (*m38_Get_Territory_OwnerID)(Tile*);};
struct Tile{TileVT* vtable;int owner=1;}tiles[8][8];Tile* p_null_tile=nullptr;
bool Map_in_range(Map*m,int,int x,int y){return x>=0&&y>=0&&x<m->Width&&y<m->Height;}
void wrap_tile_coords(Map*m,int*x,int*y){if(m->Flags&1)*x=(*x+m->Width)%m->Width;if(m->Flags&2)*y=(*y+m->Height)%m->Height;}
Tile* tile_at(int x,int y){return Map_in_range(&bic.Map,0,x,y)?&tiles[y][x]:nullptr;}
struct Record{int territory_owner_id=1;unsigned territory_edge_mask=0,territory_color_rgb=0;};
bool hidden=false;char check(Map_Renderer*,int){return hidden;}
int color(JGL_Color_Table*,int,unsigned char*p,int start,int count){assert(start==64&&count==1);p[0]=23;p[1]=145;p[2]=231;return 1;}
int owner(Tile*t){return t->owner;}
void capture(Record*record,int tile_x,int tile_y){
'''+body+r'''
}
int main(){
 TileVT tv{owner};for(auto&row:tiles)for(auto&t:row)t.vtable=&tv;
 RenderVT rv{check};bic.Map.Renderer.vtable=&rv;
 PaletteVT pv{color};JGL_Color_Table table{&pv};PCX_Color_Table palette{&table};
 Units_Image_Data images;images.Color_Tables[3]=&palette;is->custom_renderer_unit_images=&images;leaders[1].Color_Table_ID=3;
 for(unsigned mask=0;mask<16;++mask){
  for(unsigned edge=0;edge<4;++edge)tiles[3+(edge/2)*2][3+(edge&1)*2].owner=(mask&(1<<edge))?2:1;
  Record r;capture(&r,4,4);assert(r.territory_edge_mask==mask&&r.territory_color_rgb==0x1791e7);
 }
 for(auto&row:tiles)for(auto&t:row)t.owner=2;
 Record r;capture(&r,0,0);assert(r.territory_edge_mask==8); // Invalid neighbors never form borders.
 bic.Map.Flags=3;r={};capture(&r,0,0);assert(r.territory_edge_mask==15);
 hidden=true;bic.Map.Renderer.Flags=0x200;r={};capture(&r,0,0);assert(!r.territory_edge_mask);
 hidden=false;r={};capture(&r,0,0);assert(r.territory_edge_mask==15);
 r={};r.territory_owner_id=0;capture(&r,0,0);assert(!r.territory_edge_mask);
 is->custom_renderer_unit_images=nullptr;r={};capture(&r,0,0);assert(!r.territory_edge_mask);
}
''')

    def test_native_animation_delegate_keeps_all_arguments_when_disabled(self):
        source=(ROOT/'injected_code.c').read_text()
        start=source.index('void __fastcall\npatch_Units_Image_Data_advance_animations')
        body=source[start:source.index('\n#endif',start)].replace('this','self')
        run_cpp(r'''
#include <cassert>
#define __fastcall
struct Unit{};struct Units_Image_Data{};
struct State{struct{bool enable_custom_rendering=false;}current_config;Units_Image_Data*custom_renderer_unit_images=nullptr;}state,*is=&state;
Units_Image_Data original;Unit unit,*units[]={&unit};int effect=7,calls=0;
void Units_Image_Data_advance_animations(Units_Image_Data*t,int edx,float elapsed,Unit** u,int n,void*e){
 assert(t==&original&&edx==19&&elapsed==.25f&&u==units&&n==1&&e==&effect);++calls;}
'''+body+r'''
int main(){patch_Units_Image_Data_advance_animations(&original,19,.25f,units,1,&effect);
 assert(calls==1&&is->custom_renderer_unit_images==nullptr);
 state.current_config.enable_custom_rendering=true;
 patch_Units_Image_Data_advance_animations(&original,19,.25f,units,1,&effect);
 assert(calls==2&&is->custom_renderer_unit_images==&original);}
''')

if __name__=='__main__':unittest.main()
