#pragma once
#include <fstream>
#include <memory>
// Real local HUD art through JGL's allocator and palette representation. The
// ignored pack is prepared by native_ui_fixture.py and hashed in each receipt.
struct NativeUiAssets {
    struct Pair {JGLSprite color={},alpha={};void* palette=nullptr;};
    std::vector<std::unique_ptr<Pair>> pairs;char* module;
    NativeUiAssets(void* graph,char* base):module(base){
        char path[2048]={};if(!GetEnvironmentVariableA("C3X_RENDERER_NATIVE_UI_PACK",path,sizeof(path)))return;
        std::ifstream input(path,std::ios::binary);verify(bool(input),"open local native HUD pack");
        auto read=[&](void* p,std::size_t bytes){input.read(static_cast<char*>(p),bytes);verify(bool(input),"complete local HUD input");};
        unsigned count=0;read(&count,4);verify(count==6,"all local HUD pairs present");
        for(unsigned n=0;n<count;++n){
            unsigned w=0,h=0;read(&w,4);read(&h,4);verify(w&&w<=1024&&h&&h<=1024,"bounded local HUD extent");
            auto pair=std::make_unique<Pair>();auto gt=*static_cast<void***>(graph);
            pair->palette=reinterpret_cast<void*(__thiscall*)(void*,void*)>(gt[30])(graph,nullptr);verify(pair->palette!=nullptr,"HUD palette allocation");
            for(auto sprite:{&pair->color,&pair->alpha}){
                unsigned char rgb[768];std::vector<unsigned char> pixels(w*h);read(rgb,sizeof(rgb));read(pixels.data(),pixels.size());
                if(sprite==&pair->color){auto table=*static_cast<void***>(pair->palette);
                    for(unsigned format=0;format<2;++format){auto words=reinterpret_cast<unsigned short*(__thiscall*)(void*)>(table[6+format])(pair->palette);
                        for(unsigned i=0;i<256;++i)words[i]=unsigned((rgb[i*3]>>3)<<(format?11:10))|((rgb[i*3+1]>>(format?2:3))<<5)|(rgb[i*3+2]>>3);}
                    for(unsigned i=0;i<256;++i){auto dest=reinterpret_cast<unsigned char*(__thiscall*)(void*,unsigned)>(table[8])(pair->palette,i);std::memcpy(dest,rgb+i*3,3);}
                }
                reinterpret_cast<JGLSprite*(__thiscall*)(JGLSprite*,void*)>(module+0x7e80)(sprite,nullptr);
                using SpriteInit=int(__thiscall*)(JGLSprite*,void*,int,int,int,int,void*);
                verify(reinterpret_cast<SpriteInit>(sprite->vtable[1])(sprite,pixels.data(),w,h,8,0,pair->palette)==0,"native HUD sprite construction");
            }
            pairs.push_back(std::move(pair));
        }
    }
    ~NativeUiAssets(){for(auto& p:pairs){for(auto sprite:{&p->color,&p->alpha})reinterpret_cast<void(__thiscall*)(JGLSprite*)>(module+0x7ed0)(sprite);
        reinterpret_cast<void*(__thiscall*)(void*,unsigned)>(module+0x3cf10)(p->palette,1);}}
    int draw(unsigned index,JGL_Image* background,JGL_Image* destination,int x,int y,bool original=false){
        auto& p=*pairs.at(index);using Draw=int(__thiscall*)(JGLSprite*,JGLSprite*,JGL_Image*,JGL_Image*,int,int,void*);
        return reinterpret_cast<Draw>(original?state.custom_renderer_jgl_blend_original[0]:p.color.vtable[20])(&p.color,&p.alpha,background,destination,x,y,p.palette);
    }
};
