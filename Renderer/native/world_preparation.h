#pragma once
#include "source_fidelity/ground_preparation.h"
#include "object_preparation.h"
#include "render_core/captured_scene.h"
namespace c3x_renderer {
// One selected tile is the scheduling and immutable GPU allocation unit.
// Assets live until the device/content reset barrier; mutable world inputs are owned.
struct WorldPreparationMemory {
    static constexpr std::size_t limit=64u*1024u*1024u;
    std::atomic<std::size_t> bytes{0},peak{0};
    void add(std::size_t size){auto next=bytes.fetch_add(size)+size;auto old=peak.load();
        while(old<next && !peak.compare_exchange_weak(old,next)){} }
};
struct WorldPreparationTopology {
    render_core::WorldCoast coast;
    std::shared_ptr<WorldPreparationMemory> memory;
    std::size_t size;
    WorldPreparationTopology(render_core::WorldCoast const& world,std::shared_ptr<WorldPreparationMemory> account)
        :coast(world),memory(std::move(account)),size(coast.bytes()){memory->add(size);}
    WorldPreparationTopology(WorldPreparationTopology const&)=delete;
    ~WorldPreparationTopology(){memory->bytes-=size;}
};
struct WorldPreparationSources {
    std::shared_ptr<WorldPreparationTopology const> topology;
    render_core::CapturedScene::ObservationSnapshot observations;
    std::size_t size;
    WorldPreparationSources(std::shared_ptr<WorldPreparationTopology const> world,render_core::CapturedScene const& scene)
        :topology(std::move(world)),observations(scene),size(observations.bytes()+sizeof(*this)){topology->memory->add(size);}
    WorldPreparationSources(WorldPreparationSources const&)=delete;
    ~WorldPreparationSources(){topology->memory->bytes-=size;}
};
using WorldPreparationKey=std::array<std::uint64_t,24>;
struct WorldPreparationInput {
    fidelity::GroundPreparationInput ground;
    fidelity::TerrainCompileInput terrain;
    objects::PreparationInput objects;
    WorldPreparationKey key{};
    bool backing_only=false;
    std::shared_ptr<WorldPreparationSources const> sources;
};
struct PreparedWorld {
    std::unique_ptr<fidelity::PreparedGround> ground;
    std::unique_ptr<fidelity::TerrainSurfaces> terrain;
    std::unique_ptr<objects::PreparedObjects> objects;
    std::shared_ptr<void> buffer;
    std::array<unsigned,6> ground_vertices{},ground_indices{};
    std::array<unsigned,3> terrain_vertices{},terrain_indices{};
    std::size_t gpu_bytes=0;
    bool from_backing=false;
    bool backing_saved=false;
    bool upload_ready=false;
    double ground_ms=0,terrain_ms=0,object_ms=0,upload_ms=0;
    std::size_t bytes()const{return sizeof(*this)+gpu_bytes+
        (ground?ground->bytes():0)+(terrain?terrain->bytes():0)+(objects?objects->bytes():0);}
};
// Match the existing retained-content context, with conservative zoom/extent
// isolation for prepared (not yet adopted) compiler output. Camera anchors and
// animation time are not content identity. Dependency proofs remain mandatory.
inline WorldPreparationKey world_preparation_key(std::array<std::uint64_t,20> const& context,
        c3x_renderer_frame_v1 const& frame){
    WorldPreparationKey key{};std::copy(context.begin(),context.end(),key.begin());
    key[20]=frame.tile_width;key[21]=frame.tile_height;key[22]=frame.target_width;key[23]=frame.target_height;return key;
}
using WorldPreparation=render_core::ContentPreparation<WorldPreparationKey,WorldPreparationInput,PreparedWorld>;
}
