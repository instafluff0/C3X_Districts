    // Study-only tangent sampling. Source vertex directions use the linear
    // skin matrix, independently normalized as in the inspected object VS.
    std::vector<std::array<std::array<float,3>,2>> study_frames(AnimationMesh const& mesh,double phase) {
        double frame=std::clamp(phase,0.0,1.0)*(mesh.frames-1);
        unsigned first=std::min(mesh.frames-1,unsigned(frame));
        unsigned second=std::min(mesh.frames-1,first+1);float fraction=float(frame-first);
        std::vector<std::array<std::array<float,3>,2>> out(mesh.vertices.size());
        for(std::size_t i=0;i<mesh.vertices.size();++i) {
            auto const& v=mesh.vertices[i];
            for(unsigned basis=0;basis<2;++basis) {
                auto const& source=basis?v.bitangent:v.tangent;auto & result=out[i][basis];
                for(unsigned influence=0;influence<4;++influence) {
                    if(v.weights[influence]==0)continue;
                    auto a=mesh.palettes.data()+(std::size_t(first)*mesh.bones+v.joints[influence])*16;
                    auto b=mesh.palettes.data()+(std::size_t(second)*mesh.bones+v.joints[influence])*16;
                    for(unsigned axis=0;axis<3;++axis)for(unsigned c=0;c<3;++c)
                        result[axis]+=v.weights[influence]*source[c]*(a[c*4+axis]+(b[c*4+axis]-a[c*4+axis])*fraction);
                }
                float length=std::sqrt(result[0]*result[0]+result[1]*result[1]+result[2]*result[2]);
                for(unsigned axis=0;axis<3;++axis)result[axis]=length>1e-12f?result[axis]/length:source[axis];
            }
        }
        return out;
    }
