// Small transport witness for production-style texture arrays and b0-b7.
#include "../contracts/packet_v1.h"
#include <cstring>
#include <iostream>
using namespace labv2;

template<class T> std::vector<uint8_t> bytes(T const& value) {
    std::vector<uint8_t> result(sizeof(value));std::memcpy(result.data(),&value,sizeof(value));return result;
}

Packet fixture() {
    Packet p;p.width=p.height=64;p.color_branch=1;p.binding_contract=3;p.geometry_contract=1;
    p.valid_rect={0,0,64,64};
    Texture t;t.width=t.height=4;t.format=61;t.array_layers=2;
    t.mips={{4,std::vector<uint8_t>(16,64)},{2,std::vector<uint8_t>(4,32)},
            {4,std::vector<uint8_t>(16,192)},{2,std::vector<uint8_t>(4,128)}};
    p.textures.push_back(t);
    float vertices[]={-1,-1,0, 1,-1,0, -1,1,0, -1,1,0, 1,-1,0, 1,1,0};
    p.buffers.push_back(bytes(vertices));
    p.buffers.push_back(bytes(std::array<float,4>{1,0,0,0}));
    p.buffers.push_back(bytes(std::array<float,4>{.25f,0,0,0}));
    p.buffers.push_back(bytes(std::array<float,4>{0,.25f,0,0}));
    p.buffers.push_back(bytes(std::array<float,4>{0,0,.75f,0}));
    Draw d;d.vertex_buffer=0;d.constant_buffer=1;d.frame_buffer=2;d.extra_constants[0]=3;d.extra_constants[5]=4;
    d.count=6;d.stride=12;d.depth_mode=0;d.attributes={{3,0}};d.textures[0]=1;
    p.draws.push_back(d);return p;
}

int main(int argc,char**argv) {
    try {
    bool reading=argc==3 && std::string(argv[1])=="--read";
    if(argc!=2 && !reading)return 2;
    auto p=fixture();if(!reading && !write_packet(argv[1],p))return 1;
    auto read=read_packet(argv[reading?2:1]);
    if(read.binding_contract!=3 || read.textures[0].array_layers!=2 || read.textures[0].mip_levels()!=2 ||
       read.draws[0].constant(7)!=4 || read.draws[0].constant(2)!=3)return 1;
    auto invalid=[&](Packet bad) {
        FILE* file=std::tmpfile();if(!file)throw std::runtime_error("temporary stream unavailable");
        bool rejected=false;
        try {Stream stream{file,false,{}};transfer(stream,bad);}catch(std::runtime_error const&){rejected=true;}
        std::fclose(file);return rejected;
    };
    auto bad=p;bad.textures[0].mips.pop_back();if(!invalid(bad))return 1;
    bad=p;bad.textures[0].array_layers=257;if(!invalid(bad))return 1;
    bad=p;bad.draws[0].extra_constants[5]=99;if(!invalid(bad))return 1;
    bad=p;bad.buffers[4].resize(15);if(!invalid(bad))return 1;
    bad=p;bad.binding_contract=2;if(!invalid(bad))return 1;
    std::cout<<"PASS array mip layout, b2/b7 transport and invalid input rejection\n";
    } catch(std::exception const& error) {std::cerr<<error.what()<<"\n";return 1;}
}
