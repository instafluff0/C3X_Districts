// Standalone compiled-pack oracle: recover skin palettes through local joints.
#include "render_core/unit_pose_transition.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
using namespace c3x_renderer;
using namespace c3x_renderer::render_core;
int main(int argc,char** argv){
    if(argc!=2)return 2;
    std::ifstream list(argv[1]);std::string path;std::size_t files=0,samples=0;double error=0;
    while(std::getline(list,path)){
        std::ifstream stream(path,std::ios::binary);
        std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(stream)),{});
        AnimationMesh mesh;
        if(!decode_animation_mesh(bytes,mesh)||mesh.rig.parents.empty()){
            std::cerr<<"Invalid or missing rig: "<<path<<'\n';return 1;}
        auto const& rig=mesh.rig;
        std::array<bool,256> used{};
        for(auto const& vertex:mesh.vertices)for(unsigned i=0;i<4;++i)
            if(vertex.weights[i]>0)used[vertex.joints[i]]=true;
        for(unsigned frame:{0u,mesh.frames/2,mesh.frames-1}){
            std::array<JointMatrix,256> world{};
            for(std::size_t bone=0;bone<rig.parents.size();++bone){
                world[bone]=joint_matrix(rig.poses[frame*rig.parents.size()+bone]);
                if(rig.parents[bone]>=0)world[bone]=joint_multiply(world[bone],world[rig.parents[bone]]);
            }
            for(unsigned bone=0;bone<mesh.bones;++bone){
                if(!used[bone])continue;
                auto palette=joint_multiply(rig.inverse_bind[bone],world[rig.skin_joints[bone]]);
                for(unsigned j=0;j<16;++j){
                    error=std::max(error,double(std::abs(palette[j]-mesh.palettes[(frame*mesh.bones+bone)*16+j])));
                    if(error>2e-4){std::cerr<<"Authored pose mismatch frame="<<frame<<" bone="<<bone<<" component="<<j<<" expected="<<mesh.palettes[(frame*mesh.bones+bone)*16+j]<<" actual="<<palette[j]<<" max_error="<<error<<": "<<path<<'\n';return 1;}
                }
                ++samples;
            }
        }
        ++files;
    }
    if(!files)return 1;
    std::cout<<"PASS unit rig pack files="<<files<<" joint_samples="<<samples<<" max_error="<<error<<'\n';
}
