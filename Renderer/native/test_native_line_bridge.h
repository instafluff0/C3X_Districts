#pragma once
// Extracted production OpenGL patches. Only original EXE/GDI+ endpoints are
// test stand-ins; DC acquisition executes the actual hooked, hash-pinned JGL.
struct OpenGLRenderer {unsigned initialized=0,drawn=0,style_calls=0;};
int __fastcall OpenGLRenderer_initialize(OpenGLRenderer* context,int,PCX_Image* target){
    ++context->initialized;auto image=target->JGL.Image;
    auto dc=reinterpret_cast<HDC(__thiscall*)(JGL_Image*)>(image->vtable[10])(image);
    if(!dc)return 2;reinterpret_cast<Release>(image->vtable[11])(image,1);return 0;
}
void __fastcall OpenGLRenderer_set_color(OpenGLRenderer* context,int,unsigned){++context->style_calls;}
void __fastcall OpenGLRenderer_set_opacity(OpenGLRenderer* context,int,unsigned){++context->style_calls;}
void __fastcall OpenGLRenderer_set_line_width(OpenGLRenderer* context,int,int){++context->style_calls;}
void __fastcall OpenGLRenderer_enable_line_dashing(OpenGLRenderer* context){++context->style_calls;}
void __fastcall OpenGLRenderer_disable_line_dashing(OpenGLRenderer* context){++context->style_calls;}
void __fastcall OpenGLRenderer_draw_line(OpenGLRenderer* context,int,int,int,int,int){++context->drawn;}
unsigned native_gdi_initializations=0,native_gdi_draws=0;float native_gdi_width=0;unsigned native_gdi_argb=0;
int __stdcall line_create_graphics(HDC dc,void** out){++native_gdi_initializations;*out=dc;return 0;}
int __stdcall line_delete(void*){return 0;}
int __stdcall line_set(void*,int){return 0;}
int __stdcall line_pen(unsigned argb,float width,int,void** out){native_gdi_argb=argb;native_gdi_width=width;*out=&native_gdi_argb;return 0;}
int __stdcall line_draw(void*,void*,int,int,int,int){++native_gdi_draws;return 0;}
bool set_up_gdi_plus(){auto& g=state.gdi_plus;g.CreateFromHDC=line_create_graphics;g.DeleteGraphics=line_delete;
    g.SetSmoothingMode=g.SetPenDashStyle=line_set;g.CreatePen1=line_pen;g.DeletePen=line_delete;g.DrawLineI=line_draw;return true;}
#include "build/native_line_hooks.h"
