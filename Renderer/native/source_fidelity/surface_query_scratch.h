#pragma once
#include "../../lab/shared/natural/world.h"
#include "../../lab/shared/natural/queries.h"
#include "../../lab/shared/natural/relief.h"
namespace c3x_renderer { namespace fidelity {

// A private instance of every mutable, per-tile object the height/shore
// query pipeline (SurfaceQueries + ReliefSurface + NaturalWorld river
// sampling) touches. The renderer's own per-tile `queries`/`pickup_surface`
// share one instance of each across every generator that runs for a tile
// (ground, cliffs, city); a second generator can only run concurrently with
// that one if it queries through its own copy instead. Mirrors the isolation
// TerrainCompileScratch already established for the terrain-worker pool:
// `rivers` borrows the renderer's immutable NaturalData payload but tracks
// its own dependency-consumer state, so two DependencyScopes never race.
struct SurfaceQueryScratch {
    render_core::ExactPointCache<render_core::ShoreSample> shore_samples;
    render_core::ExactPointCache<render_core::GroundSample> pickup_ground_samples;
    render_core::ExactPointCache<std::array<float,2>> height_samples;
    NaturalWorld rivers;
    std::size_t pickup_height_queries=0;
    render_core::World dimensions{};
    SurfaceQueryScratch(){rivers.river_page_limit=2;}
    void bind(NaturalData const& natural,render_core::WorldTopology const& world,std::int64_t revision){
        auto next=world.dimensions();
        if(next.width!=dimensions.width || next.height!=dimensions.height ||
           next.wrap_x!=dimensions.wrap_x || next.wrap_y!=dimensions.wrap_y)rivers.reset_world();
        dimensions=next;
        rivers.borrowed_data=&natural;rivers.update_rivers(world,revision);
    }
    void reset_tile(){pickup_ground_samples.clear();height_samples.clear();}
};

}}
