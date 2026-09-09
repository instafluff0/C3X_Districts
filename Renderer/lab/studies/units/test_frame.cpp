#include "animation_runtime.h"
#include <fstream>
#include <iterator>
#include <iostream>
namespace c3x_renderer {
struct FrameStudy {
#include "sample_frame.h"
};
}
int main(int argc,char**argv) {
    using namespace c3x_renderer;
    if(argc!=3)return 1;
    auto read=[](char const*path){std::ifstream in(path,std::ios::binary);return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(in),{});};
    auto a=read(argv[1]),b=read(argv[2]);AnimationMesh old,newer;
    if(!decode_animation_mesh(a,old)||!decode_animation_mesh(b,newer)||old.vertices.size()!=newer.vertices.size()||old.indices!=newer.indices||old.palettes!=newer.palettes)return 2;
    for(unsigned i=0;i<old.vertices.size();++i) {
        if(std::memcmp(&old.vertices[i].source,&newer.vertices[i].source,sizeof(FeatureSourceVertex)) ||
           old.vertices[i].weights!=newer.vertices[i].weights || old.vertices[i].joints!=newer.vertices[i].joints)return 3;
    }
    auto rejected=[&](std::vector<std::uint8_t> bytes){AnimationMesh sentinel;sentinel.duration=123;
        return !decode_animation_mesh(bytes,sentinel)&&sentinel.duration==123;};
    auto corrupt=b;corrupt.pop_back();if(!rejected(corrupt))return 4;
    corrupt=b;corrupt.push_back(0);if(!rejected(corrupt))return 5;
    corrupt=b;corrupt[8]=1;if(!rejected(corrupt))return 6;
    corrupt=b;float nan=std::nanf("");std::memcpy(corrupt.data()+32+64,&nan,4);if(!rejected(corrupt))return 7;
    // Analytic 90-degree Z rotation, interpolation and non-unit uniform scale.
    AnimationMesh mesh;mesh.frames=2;mesh.bones=1;mesh.duration=1;mesh.vertices.resize(1);mesh.palettes.resize(32);
    auto &v=mesh.vertices[0];v.weights={1,0,0,0};v.joints={0,0,0,0};v.tangent={1,0,0};v.bitangent={0,-1,0};
    mesh.palettes={2,0,0,0,0,2,0,0,0,0,2,0,0,0,0,1, 0,2,0,0,-2,0,0,0,0,0,2,0,0,0,0,1};
    FrameStudy study;auto start=study.study_frames(mesh,0),half=study.study_frames(mesh,.5),end=study.study_frames(mesh,1);
    auto close=[](float a,float b){return std::abs(a-b)<1e-6f;};
    if(!close(start[0][0][0],1)||!close(start[0][1][1],-1)||!close(end[0][0][1],1)||!close(end[0][1][0],1)||
       !close(half[0][0][0],std::sqrt(.5f))||!close(half[0][0][1],std::sqrt(.5f)))return 8;
    for(double phase:{0.,.31,1.}) {
        std::vector<FeatureSourceVertex> x,y;
        if(!sample_animation_mesh(old,phase*old.duration,false,x)||!sample_animation_mesh(newer,phase*newer.duration,false,y)||x.size()!=y.size()||
           std::memcmp(x.data(),y.data(),x.size()*sizeof(FeatureSourceVertex)))return 9;
    }
    std::cout<<"PASS frame payload geometry/pose parity, malformed-input rejection and analytic tangent rotation\n";
}
