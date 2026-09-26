#pragma once
// Opt-in Lab capture of the exact projected terrain triangles fed to the GPU.
// The regular ground and joined mountain surface are exported before packing;
// their positions and indices are unchanged by the natural-mesh packer.
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif
#include "../../shared/natural/vertex.h"

namespace c3x_renderer { namespace fidelity {
inline bool export_border_ground_mesh(int tile_x,int tile_y,
        std::array<std::vector<MapVertex>,3> const& vertices,
        std::array<std::vector<unsigned>,2> const& indices) {
#ifndef _WIN32
    (void)tile_x;(void)tile_y;(void)vertices;(void)indices;
    return true;
#else
    char prefix[1024]={},selected[64]={};
    unsigned prefix_length=GetEnvironmentVariableA("C3X_RENDERER_BORDER_MESH_PREFIX",prefix,sizeof(prefix));
    if(!prefix_length)return true;
    if(prefix_length>=sizeof(prefix))return false;
    int site_x=0,site_y=0;
    if(!GetEnvironmentVariableA("C3X_RENDERER_BORDER_MESH_SITE",selected,sizeof(selected)) ||
       sscanf_s(selected,"%d,%d",&site_x,&site_y)!=2)return false;
    if(std::abs(tile_x-site_x)>5 || std::abs(tile_y-site_y)>5)return true;
    std::string path=std::string(prefix)+"."+std::to_string(tile_x)+"_"+std::to_string(tile_y)+".bin";
    std::FILE* file=nullptr;
    if(fopen_s(&file,path.c_str(),"wb"))return false;
    if(!file)return false;
    bool ok=std::fwrite("C3XBRD1",1,7,file)==7;
    auto write=[&](void const* data,std::size_t size){ok=ok && std::fwrite(data,1,size,file)==size;};
    char zero=0;write(&zero,1);
    std::int32_t tile[]={tile_x,tile_y};write(tile,sizeof(tile));
    for(unsigned layer:{0u,2u}){
        auto const& source=vertices[layer];
        auto const& topology=indices[layer==0?0:1];
        std::uint32_t counts[]={std::uint32_t(source.size()),
            std::uint32_t(topology.empty()?source.size():topology.size())};
        write(counts,sizeof(counts));
        for(auto const& vertex:source){
            float values[]={vertex.world_x,vertex.world_y,vertex.world_z};
            write(values,sizeof(values));
        }
        if(topology.empty())for(std::uint32_t i=0;i<counts[1];++i)write(&i,sizeof(i));
        else for(unsigned index:topology){std::uint32_t value=index;write(&value,sizeof(value));}
    }
    if(std::fclose(file)!=0)ok=false;
    return ok;
#endif
}
}}
