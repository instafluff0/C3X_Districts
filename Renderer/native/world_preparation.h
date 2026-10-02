#pragma once
#include "source_fidelity/ground_preparation.h"
#include "object_preparation.h"
#include "render_core/captured_scene.h"
#include "render_core/prepared_world_validity.h"
namespace c3x_renderer {
// One selected tile component is the scheduling and immutable GPU allocation unit.
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
    std::uint64_t scope;
    WorldPreparationTopology(render_core::WorldCoast const& world,std::shared_ptr<WorldPreparationMemory> account,std::uint64_t generation)
        :coast(world),memory(std::move(account)),size(coast.bytes()),scope(generation){memory->add(size);}
    WorldPreparationTopology(WorldPreparationTopology const&)=delete;
    ~WorldPreparationTopology(){memory->bytes-=size;}
};
struct WorldPreparationSources {
    std::shared_ptr<WorldPreparationTopology const> topology;
    std::shared_ptr<render_core::CapturedScene::WorldSnapshot const> observations;
    std::size_t size;
    WorldPreparationSources(std::shared_ptr<WorldPreparationTopology const> world,render_core::CapturedScene const& scene)
        :topology(std::move(world)),observations(scene.world_snapshot()),size(sizeof(*this)){topology->memory->add(size);}
    WorldPreparationSources(WorldPreparationSources const&)=delete;
    ~WorldPreparationSources(){topology->memory->bytes-=size;}
};
// Lookup hashes never prove equality. Keep scalar bit patterns and the ordered
// river-node values beside the hash, including exact lifetime/compiler inputs.
struct GroundRecipeKey {
    std::vector<std::uint64_t> words;
    bool operator==(GroundRecipeKey const& other)const{return words==other.words;}
    bool operator!=(GroundRecipeKey const& other)const{return !(*this==other);}
    bool operator<(GroundRecipeKey const& other)const{return words<other.words;}
    std::uint64_t lookup_hash()const{
        std::uint64_t value=1469598103934665603ull;
        for(auto word:words)for(unsigned byte=0;byte<8;++byte)value=(value^((word>>(byte*8))&255u))*1099511628211ull;
        return value;
    }
};
struct WorldPreparationKey {
    std::array<std::uint64_t,25> identity{};
    GroundRecipeKey recipe;
    bool operator==(WorldPreparationKey const& other)const{return identity==other.identity && recipe==other.recipe;}
    bool operator!=(WorldPreparationKey const& other)const{return !(*this==other);}
    bool operator<(WorldPreparationKey const& other)const{
        return identity<other.identity || (identity==other.identity && recipe<other.recipe);
    }
    WorldPreparationKind kind()const{return WorldPreparationKind(identity[24]);}
};
inline GroundRecipeKey ground_recipe_key(fidelity::GroundPreparationInput const& ground,
        fidelity::TerrainCompileInput const& terrain,std::array<std::uint64_t,4> const& lifetime){
    GroundRecipeKey key;key.words.reserve(80+ground.nodes.size()*4);
    auto add=[&](auto value){
        static_assert(sizeof(value)<=sizeof(std::uint64_t),"recipe scalar");
        std::uint64_t word=0;std::memcpy(&word,&value,sizeof(value));key.words.push_back(word);
    };
    for(auto value:lifetime)add(value); // scope, assets, device, compiler revision
    for(auto value:terrain.key)add(value);
    add(terrain.tile_x);add(terrain.tile_y);add(terrain.real_terrain_type);add(terrain.ground);
    add(terrain.tile_width);add(terrain.tile_height);add(terrain.target_height);add(terrain.detail.identity());
    add(terrain.river_ready);add(terrain.skip_flat_shore);add(terrain.separate_relief);add(terrain.indexed);add(terrain.retain_height);
    auto const& input=ground.compile;auto const& tile=input.tile;
    add(tile.tile_x);add(tile.tile_y);add(tile.terrain_type);add(tile.real_terrain_type);add(tile.river_code);add(tile.has_effect!=0);
    add(input.world_ground);add(input.pickup_profile);add(input.fidelity_profile);add(input.draw_marsh);add(input.river_assets_ready);
    add(input.ground);add(input.uv_scale);add(input.flat_grid);add(input.tile_ground_grid);add(input.shadow_grid);
    // Canonical world vertices overwrite screen XY/depth/normals; pickup has
    // no light-dependent terrain-shadow output. Legacy keys keep these inputs.
    if(!input.world_ground){add(input.half_w);add(input.half_h);add(input.left);add(input.top);add(input.relief_projection_scale);
        add(input.retain_ground_grids);add(input.reuse_nested_ground_grids);add(input.prewarming);}
    if(!input.pickup_profile)for(auto value:input.key_light)add(value);
    add(ground.tile_width);add(ground.tile_height);add(ground.world_width);add(ground.world_height);
    add(ground.wrap_x);add(ground.wrap_y);add(ground.skip_flat_shore);add(ground.separate_natural_relief);
    add(ground.center.distance);add(ground.center.beach_width);add(ground.center.rocky);add(ground.center.depth);
    add(std::uint64_t(ground.nodes.size()));
    for(auto const& node:ground.nodes){add(node.lattice_x);add(node.lattice_y);add(node.degree);add(node.touches_water);}
    // Global topology revisions select query-cache freshness, not content.
    // Actual sampled world/coast/river dependencies must validate every hit.
    return key;
}
inline WorldPreparationKey ground_world_preparation_key(GroundRecipeKey const& recipe){
    WorldPreparationKey key;key.identity[0]=recipe.lookup_hash();key.identity[24]=std::uint64_t(WorldPreparationKind::ground);key.recipe=recipe;return key;
}
// Per-compile proxy preserves the immutable observation lease while recording
// only the observation value the ground compiler reads. It is never shared
// between lanes, and it neither changes city/forest observations nor authority.
template<class Observations> class GroundPreparationObservations {
    Observations const& source;
    struct Record {int ground=-1,relief=-1;std::uint64_t semantic=0;};
    mutable Record record;
public:
    explicit GroundPreparationObservations(Observations const& observations):source(observations){}
    auto key(int x,int y)const{return source.key(x,y);}
    Record const* current(std::uint64_t key)const{
        auto value=source.current(key);if(!value)return nullptr;
        record={value->ground,value->relief,render_core::ground_topology_value(value)};return &record;
    }
};
template<class Observations> GroundPreparationObservations<Observations> ground_preparation_observations(Observations const& observations){
    return GroundPreparationObservations<Observations>(observations);
}
struct WorldPreparationInput {
    fidelity::GroundPreparationInput ground;
    fidelity::TerrainCompileInput terrain;
    objects::PreparationInput objects;
    WorldPreparationKey key{};
    bool backing_only=false;
    std::shared_ptr<WorldPreparationSources const> sources;
    WorldPreparationKind kind=WorldPreparationKind::combined;
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
    WorldPreparationKind kind=WorldPreparationKind::combined;
    bool complete()const{return world_preparation_kind_valid(kind) &&
        (!world_preparation_needs_ground(kind) || (ground && terrain)) &&
        (!world_preparation_needs_objects(kind) || bool(objects));}
    std::size_t bytes()const{return sizeof(*this)+gpu_bytes+
        (ground?ground->bytes():0)+(terrain?terrain->bytes():0)+(objects?objects->bytes():0);}
};
// Canonical full-detail world output has projection-independent inputs. Legacy
// output keeps its exact zoom/extent identity; real detail lives in context. Camera anchors and
// animation time are not content identity. Dependency proofs remain mandatory.
inline WorldPreparationKey world_preparation_key(std::array<std::uint64_t,20> const& context,
        c3x_renderer_frame_v1 const& frame,bool canonical=false,WorldPreparationKind kind=WorldPreparationKind::combined){
    WorldPreparationKey key{};std::copy(context.begin(),context.end(),key.identity.begin());
    key.identity[20]=canonical?0:frame.tile_width;key.identity[21]=canonical?0:frame.tile_height;
    key.identity[22]=canonical?0:frame.target_width;key.identity[23]=canonical?0:frame.target_height;
    key.identity[24]=std::uint64_t(kind);return key;
}
using WorldPreparation=render_core::ContentPreparation<WorldPreparationKey,WorldPreparationInput,PreparedWorld>;
}
