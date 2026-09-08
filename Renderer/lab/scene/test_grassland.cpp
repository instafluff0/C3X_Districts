// Exercise the pilot's input boundary without local art, a VM or a GPU.
#define main grassland_scene_main
#include "grassland.cpp"
#undef main
#include <cassert>

template<class F> void rejects(F operation) {
    bool rejected=false;
    try {operation();}catch(std::runtime_error const&){rejected=true;}
    assert(rejected);
}
int main() {
    std::string csv="C3X_BIQ_TERRAIN_V3,32,32,512\n";
    for(int y=0;y<32;y++)for(int x=y%2;x<32;x+=2)
        csv+=std::to_string(x)+","+std::to_string(y)+",2,2,0,0,0\n";
    std::istringstream valid(csv);auto world=read_detail(valid);
    assert(world.tiles.size()==512 && world.bits.size()==512);
    for(auto bits:world.bits)assert(bits==(2|(2<<8)));
    auto bad_scene=[&](std::string value){rejects([&](){std::istringstream input(value);read_detail(input);});};
    bad_scene(csv+"0,0,2,2,0,0,0\n"); // duplicate
    bad_scene(csv.substr(0,csv.rfind('\n',csv.size()-2)+1)); // missing tile
    for(std::string first:{"32,0,2,2,0,0,0", "-2,0,2,2,0,0,0", "1,0,2,2,0,0,0",
        "0,0,2,5,0,0,0", "0,0,1,1,0,0,0", "0,0,2,2,1,0,0", "0,0,2,2,0,1,0",
        "0,0,2,2,0,0,2", "0,0,2,2,0,0,0,1", "0,0,2,2,0,0,0 garbage", "0,0,2,,2,0,0,0"}) {
        auto start=csv.find('\n')+1,end=csv.find('\n',start);
        auto changed=csv;changed.replace(start,end-start,first);bad_scene(changed);
    }
    bad_scene("C3X_BIQ_TERRAIN_V3,100,100,5000\n"+csv.substr(csv.find('\n')+1));
    std::string context="C3X_BIQ_TERRAIN_V3,32,32,512\n";
    for(int y=0;y<32;y++)for(int x=y%2;x<32;x+=2){
        int base=x<12?1:2,real=base;
        if(x==18 && y==14)real=5;
        if(x==14 && y==12)real=7;
        if(x==20 && y==10)real=6;
        context+=std::to_string(x)+","+std::to_string(y)+","+std::to_string(base)+","+std::to_string(real)+",0,0,0\n";
    }
    std::istringstream context_input(context);auto context_world=read_detail(context_input,true);
    assert(context_world.bits[(14*32+18)/2]==(2|(5<<8)));
    assert(context_world.bits[(12*32+14)/2]==(2|(7<<8)));
    assert(context_world.bits[(10*32+20)/2]==(2|(6<<8)));
    assert(context_world.bits[0]==(1|(1<<8)));
    rejects([&](){std::istringstream input(csv);read_detail(input,true);});
    bad_scene(context); // Context cannot be mislabeled as the approved detail pilot.
    std::vector<std::uint8_t> dds(148+8+8+8,0);
    std::memcpy(dds.data(),"DDS ",4);
    auto set=[&](unsigned at,std::uint32_t value){std::memcpy(dds.data()+at,&value,4);};
    set(4,124);set(76,32);set(84,0x30315844u);set(12,4);set(16,4);set(28,3);
    set(128,71);set(132,3);set(140,1);
    auto original=dds;auto loaded=texture(dds);
    assert(loaded.width==4 && loaded.height==4 && loaded.mips.size()==3);
    for(auto const&mip:loaded.mips)assert(mip.pitch==8);
    for(auto change:{std::array<unsigned,2>{4,123},{76,31},{84,0},{12,0},{16,16385},
                     {28,4},{128,999},{132,4},{140,2},{136,4}}) {
        dds=original;set(change[0],change[1]);rejects([&](){texture(dds);});
    }
    dds=original;dds.pop_back();rejects([&](){texture(dds);});
    dds=original;dds.push_back(0);rejects([&](){texture(dds);});
    dds.resize(100);rejects([&](){texture(dds);});
    std::cout<<"PASS grassland pilot fixture and DDS boundaries\n";
}
