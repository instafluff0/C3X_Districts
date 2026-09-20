#pragma once
#include "source_fidelity/ground_preparation.h"
#include "object_preparation.h"
namespace c3x_renderer {
// One selected tile is the scheduling and immutable GPU allocation unit.
// Borrowed assets/world remain under the frame lease; only copied inputs enter jobs.
struct WorldPreparationInput {
    fidelity::GroundPreparationInput ground;
    fidelity::TerrainCompileInput terrain;
    objects::PreparationInput objects;
};
struct PreparedWorld {
    std::unique_ptr<fidelity::PreparedGround> ground;
    std::unique_ptr<fidelity::TerrainSurfaces> terrain;
    std::unique_ptr<objects::PreparedObjects> objects;
    std::shared_ptr<void> buffer;
    std::array<unsigned,6> ground_vertices{},ground_indices{};
    std::array<unsigned,3> terrain_vertices{},terrain_indices{};
    std::size_t gpu_bytes=0;
    double ground_ms=0,terrain_ms=0,object_ms=0,upload_ms=0;
    std::size_t bytes()const{return sizeof(*this)+gpu_bytes+
        (ground?ground->bytes():0)+(terrain?terrain->bytes():0)+(objects?objects->bytes():0);}
};
// Match the existing retained-content context, with conservative zoom/extent
// isolation for prepared (not yet adopted) compiler output. Camera anchors and
// animation time are not content identity. Dependency proofs remain mandatory.
using WorldPreparationKey=std::array<std::uint64_t,24>;
inline WorldPreparationKey world_preparation_key(std::array<std::uint64_t,20> const& context,
        c3x_renderer_frame_v1 const& frame){
    WorldPreparationKey key{};std::copy(context.begin(),context.end(),key.begin());
    key[20]=frame.tile_width;key[21]=frame.tile_height;key[22]=frame.target_width;key[23]=frame.target_height;return key;
}
using WorldPreparation=render_core::ContentPreparation<WorldPreparationKey,WorldPreparationInput,PreparedWorld>;
}
