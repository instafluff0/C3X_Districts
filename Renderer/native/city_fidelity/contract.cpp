#include <fstream>
#include <iostream>
#include <iterator>
#include "runtime.h"
int main(int argc,char**argv) {
    using namespace c3x_renderer::city_fidelity;
    if(argc!=2)return 2;
    std::ifstream stream(argv[1],std::ios::binary);
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(stream)),{});
    Library library;
    if(!library.decode(bytes)){std::cerr<<"invalid city composition payload\n";return 1;}
    std::size_t selected=0,lights=0,vertices=0,triangles=0;
    for(auto const&m:library.models)for(auto const&p:m.parts){vertices+=p.vertices.size();triangles+=p.indices.size()/3;}
    for(auto const&t:library.compositions){
        if(t.authority=="selected-r111" || t.authority=="selected-r112"){
            selected++;unsigned count=0;
            for(auto const&i:t.instances)if(i.capital){
                count++;
                if(std::abs(i.yaw-.5235987755983f)>1e-5f || std::abs(i.offset[0])+std::abs(i.offset[1])>1e-5f)return 1;
            }
            if(count!=1 || t.paving.vertices.empty())return 1;
        }
        for(auto const&i:t.instances){
            lights+=i.lights.size();
            auto a=place(i,10,20,2.5f),b=place(i,27,13,2.5f);
            for(auto const&p:library.models[i.model].parts)for(auto const&v:p.vertices){
                float wa[3],wb[3];a.position(v.position,wa);b.position(v.position,wb);
                if(std::abs(wb[0]-wa[0]-17)>5e-6f || std::abs(wb[1]-wa[1]+7)>5e-6f || wa[2]!=wb[2])return 1;
                // Translation changes neither source frame nor height. Native
                // zoom is applied once after this common world transform.
                float rise=(wa[2]-2.5f/112)*112*.468571424f;
                if(std::abs(rise-v.position[2]*i.scale*80.9543f)>.0001f)return 1;
            }
        }
    }
    if(selected!=4)return 1;
    // Every truncation must fail transactionally; retain the usable library.
    auto old_size=library.byte_count;unsigned rejected=0;
    for(std::size_t length=0;length<bytes.size();length+=7919){
        std::vector<std::uint8_t> short_file(bytes.begin(),bytes.begin()+length);
        if(library.decode(short_file) || library.byte_count!=old_size)return 1;rejected++;
    }
    auto altered=bytes;altered[8]=altered[9]=altered[10]=altered[11]=255;
    if(library.decode(altered) || library.byte_count!=old_size)return 1;
    altered=bytes;altered.push_back(0);if(library.decode(altered))return 1;
    std::cout<<"PASS city composition: models="<<library.models.size()<<" materials="<<library.materials.size()
        <<" templates="<<library.compositions.size()<<" vertices="<<vertices<<" triangles="<<triangles
        <<" lights="<<lights<<" selected="<<selected<<" rejected_truncations="<<rejected<<" bytes="<<library.byte_count<<"\n";
    return 0;
}
