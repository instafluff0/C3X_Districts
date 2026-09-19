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
using WorldPreparation=render_core::ContentPreparation<unsigned,WorldPreparationInput,PreparedWorld>;
// Queue callbacks borrow renderer-owned private scratch under the frame lease.
// The caller clears/joins this queue before mutating those owners.
struct WorldPreparationLease {WorldPreparation queue;};
}
