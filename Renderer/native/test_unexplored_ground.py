"""Unexplored ground is drawn; the fog overlay alone hides it (instant reveals).

The user (October 17): like Civ III, the map is drawn as it really is and
fog of war and unexplored black are laid over it. A unit's move must reveal
the tiles around it instantly, after every step, not after a long rebuild.

Previously exploration changed the terrain inputs (ViewerTopology marked
unexplored tiles), so a reveal recompiled the explored ground around it, and
the fog stayed until a new view with the revealed tiles' geometry was
published; a further move cancelled that job. Now:

- unexplored tiles within two tiles of explored land (the hidden ring) are
  compiled and drawn as ordinary terrain from their object-free world inputs
  (two: a unit on a mountain sees two tiles, and a quick second step can
  arrive before the first step's view is drawn);
- a reveal changes no terrain input, so it invalidates nothing;
- the fog lifts on the view already drawn, since it holds the ring, when that
  view draws the whole reveal: every newly explored tile within one tile of
  explored land (its neighbours drawn), with its contents and no unit.
  Otherwise the whole reveal waits for its view (the user, October 27: tiles
  appeared before their contents and with sharp edges).
"""
import unittest

from Renderer.native.native_cpp_test import run_cpp


class UnexploredGroundTests(unittest.TestCase):
    def test_reveal_keeps_compiled_ground_and_extends_the_ring(self):
        run_cpp(r'''
#include "Renderer/native/render_core/captured_scene.h"
#include <cassert>
#include <cstdio>
using namespace c3x_renderer::render_core;
constexpr unsigned known=C3X_RENDERER_TILE_VISIBILITY_KNOWN,explored=C3X_RENDERER_TILE_EXPLORED;
int main(){
 c3x_renderer_frame_v1 f{};f.world_width_tiles=f.world_height_tiles=20;f.tile_width=128;f.tile_height=64;
 CapturedScene scene;scene.publication_scope(f,{1,1,1,1},1);bool changed=false;
 auto tile=[](int x,int y,unsigned flags){c3x_renderer_tile_v1 t{};t.tile_x=x;t.tile_y=y;
  t.terrain_type=2;t.real_terrain_type=x==6?5:2;t.barbarian_tribe_id=-1;
  // As Civ III's capture reports absent cities and resources.
  t.city_id=t.city_owner_id=t.city_size=t.city_culture_group=t.city_era=-1;t.resource_id=t.resource_class=-1;
  t.tile_flags=flags|C3X_RENDERER_TILE_TOPOLOGY_HALO;return t;};
 // One explored tile, (4,4), with full appearance; the rest unexplored.
 for(int y=0;y<20;++y)for(int x=y&1;x<20;x+=2)
  assert(scene.publish(tile(x,y,x==4&&y==4?known|explored|C3X_RENDERER_TILE_PREFETCH:known),changed));
 auto input=[&](int x,int y){return scene.world_snapshot()->current(scene.key(x,y));};
 auto ring=[&](int x,int y){auto in=input(x,y);return in && (in->occurrence.tile_flags&CapturedScene::hidden_ring);};
 // The 24 tiles within two tiles are the hidden ring: drawable, explored
 // for drawing. Raw offsets: a neighbour is (2,0), (1,1) or (0,2).
 unsigned members=0;
 for(int y=0;y<20;++y)for(int x=y&1;x<20;x+=2)if(ring(x,y)){++members;assert(input(x,y)->occurrence.tile_flags&explored);}
 assert(members==24);
 assert(ring(6,4) && ring(5,5) && ring(8,4) && ring(6,6) && ring(4,8) && ring(2,2));
 assert(!ring(4,4) && !ring(10,4) && !ring(4,10) && !ring(7,7));
 // Native visibility stays unexplored for them.
 assert(!(scene.retained(scene.key(6,4))->visibility_flags&explored));
 auto before=input(6,4)->content_revision;auto neighbour=input(4,4)->content_revision;
 // Reveal (6,4): same terrain, nothing hidden there. Its content and the
 // explored neighbour's are unchanged, so nothing recompiles.
 assert(scene.publish(tile(6,4,known|explored|C3X_RENDERER_TILE_VISIBLE|C3X_RENDERER_TILE_PREFETCH),changed));
 assert(input(6,4)->content_revision==before && input(4,4)->content_revision==neighbour);
 assert(!ring(6,4) && (input(6,4)->occurrence.tile_flags&explored));
 // Unexplored tiles within two of it join the ring.
 assert(ring(10,4) && ring(9,5) && ring(8,6) && !ring(12,4));
 std::printf("PASS hidden ring: drawn=1 reveal_keeps_content=1 ring_extends=1\n");
}
''')

    def test_reveal_beside_explored_land_lifts_on_the_drawn_view(self):
        run_cpp(r'''
#include "Renderer/native/render_core/unit_arrival_visibility.h"
#include <cassert>
#include <cstdio>
#include <vector>
using namespace c3x_renderer::render_core;
constexpr unsigned known=C3X_RENDERER_TILE_VISIBILITY_KNOWN,explored=C3X_RENDERER_TILE_EXPLORED,
 visible=C3X_RENDERER_TILE_VISIBLE;
int main(){
 // The drawn view: (4,4) explored; (6,4) one tile away, drawn as ring
 // terrain with the ring all around it.
 std::vector<c3x_renderer_tile_v1> drawn(2);
 // As Civ III's capture reports a tile without a city or unit.
 for(int i=0;i<2;++i){drawn[i].tile_x=4+2*i;drawn[i].tile_y=4;drawn[i].tile_flags=known;
  drawn[i].city_id=drawn[i].city_owner_id=drawn[i].city_size=drawn[i].city_culture_group=drawn[i].city_era=-1;
  drawn[i].unit_type_id=drawn[i].unit_owner_id=-1;}
 drawn[0].tile_flags|=explored|visible;
 c3x_renderer_frame_v1 view{};view.tiles=drawn.data();view.tile_count=2;view.world_width_tiles=view.world_height_tiles=20;
 UnitArrivalVisibility arrival;arrival.capture(view,1);arrival.sample(view,1);
 // Civ III now reports it revealed (no unit travel is pending).
 auto native=drawn;native[1].tile_flags=known|explored|visible;
 c3x_renderer_frame_v1 now=view;now.tiles=native.data();arrival.capture(now,1);
 auto shown=arrival.sample(view,1);
 assert((shown.tiles[1].tile_flags&(explored|visible))==(explored|visible));
 assert(arrival.revealed==1 && arrival.held_undrawn==0);
 std::printf("PASS instant reveal: ring_shown=1\n");
}
''')

    def test_reveal_two_tiles_out_waits_whole_for_its_view(self):
        # A unit on a hill or mountain sees two tiles. The drawn view draws
        # the ring two tiles around explored land, so a tile two away met
        # undrawn ground: its outer edge showed a sharp cut in the fade until
        # the next view, and its neighbour one tile away showed before it
        # (the user, October 27). The whole reveal now waits for its view.
        run_cpp(r'''
#include "Renderer/native/render_core/unit_arrival_visibility.h"
#include <cassert>
#include <cstdio>
#include <vector>
using namespace c3x_renderer::render_core;
constexpr unsigned known=C3X_RENDERER_TILE_VISIBILITY_KNOWN,explored=C3X_RENDERER_TILE_EXPLORED,
 visible=C3X_RENDERER_TILE_VISIBLE;
int main(){
 // The drawn view: (4,4) explored; (6,4) and (8,4) one and two tiles away.
 std::vector<c3x_renderer_tile_v1> drawn(3);
 for(int i=0;i<3;++i){drawn[i].tile_x=4+2*i;drawn[i].tile_y=4;drawn[i].tile_flags=known;
  drawn[i].city_id=drawn[i].city_owner_id=drawn[i].city_size=drawn[i].city_culture_group=drawn[i].city_era=-1;
  drawn[i].unit_type_id=drawn[i].unit_owner_id=-1;}
 drawn[0].tile_flags|=explored|visible;
 c3x_renderer_frame_v1 view{};view.tiles=drawn.data();view.tile_count=3;view.world_width_tiles=view.world_height_tiles=20;
 UnitArrivalVisibility arrival;arrival.capture(view,1);arrival.sample(view,1);
 auto native=drawn;for(int i=1;i<3;++i)native[i].tile_flags=known|explored|visible;
 c3x_renderer_frame_v1 now=view;now.tiles=native.data();arrival.capture(now,1);
 // Both wait on the drawn view, the near one too.
 auto shown=arrival.sample(view,1);
 for(int i=1;i<3;++i)assert(!(shown.tiles[i].tile_flags&explored));
 assert(arrival.revealed==0 && arrival.held_undrawn==2);
 // The view built with the reveal shows both at once.
 shown=arrival.sample(now,1);
 for(int i=1;i<3;++i)assert((shown.tiles[i].tile_flags&(explored|visible))==(explored|visible));
 assert(arrival.revealed==2 && arrival.held_undrawn==0);
 std::printf("PASS whole reveal: two_out_waits=1 near_waits=1 shown_together=2\n");
}
''')

    def test_reveal_with_a_unit_waits_for_its_view(self):
        # Unit bodies pose only on tiles the drawn view sees, so a unit on a
        # revealed tile appeared a moment after its tile (the user, October
        # 27). Such a reveal waits for the view that sees it.
        run_cpp(r'''
#include "Renderer/native/render_core/unit_arrival_visibility.h"
#include <cassert>
#include <cstdio>
#include <vector>
using namespace c3x_renderer::render_core;
constexpr unsigned known=C3X_RENDERER_TILE_VISIBILITY_KNOWN,explored=C3X_RENDERER_TILE_EXPLORED,
 visible=C3X_RENDERER_TILE_VISIBLE;
int main(){
 std::vector<c3x_renderer_tile_v1> drawn(2);
 for(int i=0;i<2;++i){drawn[i].tile_x=4+2*i;drawn[i].tile_y=4;drawn[i].tile_flags=known;
  drawn[i].city_id=drawn[i].city_owner_id=drawn[i].city_size=drawn[i].city_culture_group=drawn[i].city_era=-1;
  drawn[i].unit_type_id=drawn[i].unit_owner_id=-1;}
 drawn[0].tile_flags|=explored|visible;
 c3x_renderer_frame_v1 view{};view.tiles=drawn.data();view.tile_count=2;view.world_width_tiles=view.world_height_tiles=20;
 UnitArrivalVisibility arrival;arrival.capture(view,1);arrival.sample(view,1);
 // Civ III reveals (6,4), one tile away, with a foreign warrior on it.
 auto native=drawn;native[1].tile_flags=known|explored|visible;native[1].unit_type_id=3;native[1].unit_owner_id=2;
 c3x_renderer_frame_v1 now=view;now.tiles=native.data();arrival.capture(now,1);
 assert(!(arrival.sample(view,1).tiles[1].tile_flags&explored) && arrival.held_undrawn==1);
 assert(arrival.sample(now,1).tiles[1].tile_flags&visible);
 std::printf("PASS unit reveal waits for its view\n");
}
''')

    def test_fog_of_war_lifts_on_the_drawn_view(self):
        # A unit's step brings fogged (explored) ground back into sight. The
        # view already draws it, so the fog lifts at once; it waited for the
        # next completed view (about 0.8 s after arrival in the unit-motion
        # game test, October 17).
        run_cpp(r'''
#include "Renderer/native/render_core/unit_arrival_visibility.h"
#include <cassert>
#include <cstdio>
using namespace c3x_renderer::render_core;
constexpr unsigned known=C3X_RENDERER_TILE_VISIBILITY_KNOWN,explored=C3X_RENDERER_TILE_EXPLORED,
 visible=C3X_RENDERER_TILE_VISIBLE;
int main(){
 c3x_renderer_tile_v1 drawn{};drawn.tile_x=4;drawn.tile_y=4;drawn.tile_flags=known|explored;
 c3x_renderer_frame_v1 view{};view.tiles=&drawn;view.tile_count=1;view.world_width_tiles=view.world_height_tiles=20;
 UnitArrivalVisibility arrival;arrival.capture(view,1);arrival.sample(view,1);
 auto native=drawn;native.tile_flags|=visible;
 c3x_renderer_frame_v1 now=view;now.tiles=&native;arrival.capture(now,1);
 assert(arrival.sample(view,1).tiles[0].tile_flags&visible);
 assert(arrival.revealed==1 && arrival.held_undrawn==0);
 // Loss of sight still applies at once to the same drawn view.
 arrival.capture(view,1);assert(!(arrival.sample(now,1).tiles[0].tile_flags&visible));
 std::printf("PASS fog lifts on the drawn view\n");
}
''')

    def test_ring_tiles_keep_their_compiled_ground(self):
        # A ring tile Civ III captures only in its halo has no native
        # appearance record, so its compiled ground was never attached and
        # every content job rebuilt it.
        run_cpp(r'''
#include "Renderer/native/render_core/captured_scene.h"
#include <cassert>
#include <cstdio>
using namespace c3x_renderer::render_core;
constexpr unsigned known=C3X_RENDERER_TILE_VISIBILITY_KNOWN,explored=C3X_RENDERER_TILE_EXPLORED;
int main(){
 c3x_renderer_frame_v1 f{};f.world_width_tiles=f.world_height_tiles=20;f.tile_width=128;f.tile_height=64;
 CapturedScene scene;scene.publication_scope(f,{1,1,1,1},1);bool changed=false;
 auto tile=[](int x,int y,unsigned flags){c3x_renderer_tile_v1 t{};t.tile_x=x;t.tile_y=y;
  t.terrain_type=2;t.real_terrain_type=2;t.barbarian_tribe_id=-1;
  t.city_id=t.city_owner_id=t.city_size=t.city_culture_group=t.city_era=-1;t.resource_id=t.resource_class=-1;
  t.tile_flags=flags|C3X_RENDERER_TILE_TOPOLOGY_HALO;return t;};
 for(int y=0;y<20;++y)for(int x=y&1;x<20;x+=2)
  assert(scene.publish(tile(x,y,x==4&&y==4?known|explored|C3X_RENDERER_TILE_PREFETCH:known),changed));
 auto id=scene.key(6,4);
 // The frame observes the ring tile in its halo, without appearance; the
 // window draws its input.
 auto native=tile(6,4,known);
 f.tiles=&native;f.tile_count=1;assert(scene.begin(f));
 assert(scene.update(native,2,-1,2,CapturedScene::topology(native)));scene.finish();
 auto drawn=scene.world_input(id)->occurrence;drawn.tile_flags|=C3X_RENDERER_TILE_RENDER;
 assert(drawn.tile_flags&CapturedScene::hidden_ring);
 scene.attach(drawn,{1,44});
 assert(scene.retained(id)->compiled_views[0].generation==44);
 // Other content is not the ring's compile.
 auto other=drawn;other.real_terrain_type=other.terrain_type=5;scene.attach(other,{1,45});
 assert(scene.retained(id)->compiled_views[0].generation==44);
 std::printf("PASS ring keeps compiled ground\n");
}
''')

    def test_ring_ground_survives_native_capture(self):
        # Civ III captures an unexplored tile on screen with its real borders,
        # overlays and objects, which the ring's object-free input omits. That
        # capture replaced the record's appearance, clearing the ground the
        # loading phase prepared and changing its identity: the first view
        # after loading rebuilt 145 ring tiles (game test, October 17).
        run_cpp(r'''
#include "Renderer/native/render_core/captured_scene.h"
#include <cassert>
#include <cstdio>
using namespace c3x_renderer::render_core;
constexpr unsigned known=C3X_RENDERER_TILE_VISIBILITY_KNOWN,explored=C3X_RENDERER_TILE_EXPLORED;
int main(){
 c3x_renderer_frame_v1 f{};f.world_width_tiles=f.world_height_tiles=20;f.tile_width=128;f.tile_height=64;
 CapturedScene scene;scene.publication_scope(f,{1,1,1,1},1);bool changed=false;
 auto tile=[](int x,int y,unsigned flags){c3x_renderer_tile_v1 t{};t.tile_x=x;t.tile_y=y;
  t.terrain_type=2;t.real_terrain_type=2;t.barbarian_tribe_id=-1;
  t.city_id=t.city_owner_id=t.city_size=t.city_culture_group=t.city_era=-1;t.resource_id=t.resource_class=-1;
  t.tile_flags=flags|C3X_RENDERER_TILE_TOPOLOGY_HALO;t.territory_owner_id=3;return t;};
 for(int y=0;y<20;++y)for(int x=y&1;x<20;x+=2)
  assert(scene.publish(tile(x,y,x==4&&y==4?known|explored|C3X_RENDERER_TILE_PREFETCH:known),changed));
 // Loading prepares the ring tile from its input.
 auto id=scene.key(6,4);auto recipe=scene.world_input(id)->occurrence;
 assert(recipe.tile_flags&CapturedScene::hidden_ring);
 scene.attach(recipe,{1,44},true);assert(scene.retained(id)->compiled_views[0].generation==44);
 auto identity=scene.world_appearance_revision(id);assert(identity);
 // Civ III's capture of the unexplored tile on screen: borders (its halo
 // capture has the owner only) and a tile effect.
 auto native=tile(6,4,known|C3X_RENDERER_TILE_RENDER);
 native.territory_owner_id=3;native.territory_edge_mask=5;native.territory_color_rgb=0x1dd8da;native.has_effect=1;
 native.tile_flags&=~C3X_RENDERER_TILE_TOPOLOGY_HALO;
 assert(scene.publish(native,changed));
 f.tiles=&native;f.tile_count=1;assert(scene.begin(f));
 assert(scene.update(native,2,-1,2,CapturedScene::topology(native)));scene.finish();
 std::printf("compiled=%llu identity=%llu->%llu\n",(unsigned long long)scene.retained(id)->compiled_views[0].generation,
  (unsigned long long)identity,(unsigned long long)scene.world_appearance_revision(id));
 assert(scene.retained(id)->compiled_views[0].generation==44);
 assert(scene.world_appearance_revision(id)==scene.world_input(id)->content_revision);
 // Each frame observes the world window, which draws the ring tile from its
 // input, while publications carry Civ III's capture: the record must not
 // alternate between the two (that cleared the ground and invalidated the
 // shadow atlas every frame).
 auto record=scene.retained(id)->revision;auto window=scene.world_input(id)->occurrence;
 window.tile_flags=(window.tile_flags&~C3X_RENDERER_TILE_TOPOLOGY_HALO)|C3X_RENDERER_TILE_PREFETCH|C3X_RENDERER_TILE_TOPOLOGY_HALO;
 for(int frame=0;frame<3;++frame){
  f.tiles=&window;f.tile_count=1;assert(scene.begin(f));
  assert(scene.update(window,2,-1,2,CapturedScene::topology(window)));scene.finish();
  assert(scene.publish(native,changed));
 }
 assert(scene.retained(id)->revision==record && scene.retained(id)->compiled_views[0].generation==44);
 // Its identity is the same whether or not the frame observes it (shadow
 // and raster proofs compare it every frame).
 auto observed=scene.world_appearance_revision(id);
 f.tiles=nullptr;f.tile_count=0;assert(scene.begin(f));scene.finish();
 assert(scene.world_appearance_revision(id)==observed);
}
''')

    def test_ring_draws_captured_contents_and_keeps_them_on_reveal(self):
        # The user (October 17): resources and forests appeared a moment
        # after their revealed tile. The ring drew object-free inputs, so a
        # reveal changed the tile's content and identity and its next view
        # rebuilt it and the canopy around it. Like Civ III, the ring now
        # draws what Civ III captured on screen; a reveal changes nothing.
        run_cpp(r'''
#include "Renderer/native/render_core/captured_scene.h"
#include <cassert>
#include <cstdio>
#include <cstring>
using namespace c3x_renderer::render_core;
constexpr unsigned known=C3X_RENDERER_TILE_VISIBILITY_KNOWN,explored=C3X_RENDERER_TILE_EXPLORED,
 visible=C3X_RENDERER_TILE_VISIBLE,render=C3X_RENDERER_TILE_RENDER;
int main(){
 c3x_renderer_frame_v1 f{};f.world_width_tiles=f.world_height_tiles=20;f.tile_width=128;f.tile_height=64;
 CapturedScene scene;scene.publication_scope(f,{1,1,1,1},1);bool changed=false;
 auto tile=[](int x,int y,unsigned flags){c3x_renderer_tile_v1 t{};t.tile_x=x;t.tile_y=y;
  t.terrain_type=2;t.real_terrain_type=2;t.barbarian_tribe_id=-1;
  t.city_id=t.city_owner_id=t.city_size=t.city_culture_group=t.city_era=-1;t.resource_id=t.resource_class=-1;
  t.tile_flags=flags|C3X_RENDERER_TILE_TOPOLOGY_HALO;return t;};
 for(int y=0;y<20;++y)for(int x=y&1;x<20;x+=2)
  assert(scene.publish(tile(x,y,x==4&&y==4?known|explored|C3X_RENDERER_TILE_PREFETCH:known),changed));
 auto input=[&](int x,int y){return scene.world_input(scene.key(x,y));};
 // Civ III draws the unexplored tiles on screen: a pine forest with game,
 // inside borders, and a city next to it.
 auto forest=tile(6,4,known|render);forest.real_terrain_type=7;
 forest.feature_flags=C3X_RENDERER_FEATURE_FOREST|C3X_RENDERER_FEATURE_PINE;
 forest.resource_id=5;forest.resource_class=1;std::strcpy(forest.resource_name,"Game");
 forest.territory_edge_mask=5;forest.territory_color_rgb=0x1dd8da;
 auto town=tile(5,5,known|render);town.city_id=9;town.city_owner_id=2;town.city_size=3;
 assert(scene.publish(forest,changed) && scene.publish(town,changed));
 auto ring=input(6,4)->occurrence;
 assert((ring.tile_flags&CapturedScene::hidden_ring) && ring.resource_id==5 && ring.territory_edge_mask==5);
 assert(ring.feature_flags==(C3X_RENDERER_FEATURE_FOREST|C3X_RENDERER_FEATURE_PINE));
 assert(input(5,5)->occurrence.city_id==-1); // city bodies are known once explored
 auto id=scene.key(6,4);auto revision=input(6,4)->content_revision,identity=scene.world_appearance_revision(id);
 auto recipe=ring;recipe.tile_flags|=C3X_RENDERER_TILE_PREFETCH;scene.attach(recipe,{1,44},true);
 assert(scene.retained(id)->compiled_views[0].generation==44);
 // The reveal shows the same tile: its content, identity and compiled
 // ground are unchanged, so nothing on it appears later.
 forest.tile_flags=known|explored|visible|render;assert(scene.publish(forest,changed));
 assert(!(input(6,4)->occurrence.tile_flags&CapturedScene::hidden_ring));
 assert(input(6,4)->content_revision==revision && scene.world_appearance_revision(id)==identity);
 assert(scene.retained(id)->compiled_views[0].generation==44);
 // Revealing the city changes what is drawn there.
 auto withheld=input(5,5)->content_revision;
 town.tile_flags=known|explored|visible|render;assert(scene.publish(town,changed));
 assert(input(5,5)->occurrence.city_id==9 && input(5,5)->content_revision!=withheld);
 std::printf("PASS ring contents: resource=1 forest=1 reveal_unchanged=1 city_withheld=1\n");
}
''')

    def test_reveal_with_new_contents_waits_for_its_view(self):
        # A reveal can show what the drawn ring tile lacks (a city, or
        # improvements learned with it). Lifting the fog then showed the
        # contents a moment later; the tile now waits for a view drawing it,
        # and so does the rest of its reveal, which appears with it (the
        # tiles around a city appeared first, the user, October 27).
        run_cpp(r'''
#include "Renderer/native/render_core/unit_arrival_visibility.h"
#include <cassert>
#include <cstdio>
#include <vector>
using namespace c3x_renderer::render_core;
constexpr unsigned known=C3X_RENDERER_TILE_VISIBILITY_KNOWN,explored=C3X_RENDERER_TILE_EXPLORED,
 visible=C3X_RENDERER_TILE_VISIBLE;
int main(){
 // The drawn view: (4,4) explored, (6,4) and (5,5) beside it, drawn as
 // ring terrain.
 std::vector<c3x_renderer_tile_v1> drawn(3);
 int const at[3][2]={{4,4},{6,4},{5,5}};
 for(int i=0;i<3;++i){drawn[i].tile_x=at[i][0];drawn[i].tile_y=at[i][1];drawn[i].tile_flags=known;drawn[i].resource_id=-1;
  drawn[i].city_id=drawn[i].city_owner_id=drawn[i].city_size=drawn[i].city_culture_group=drawn[i].city_era=-1;
  drawn[i].unit_type_id=drawn[i].unit_owner_id=-1;}
 drawn[0].tile_flags|=explored|visible;
 c3x_renderer_frame_v1 view{};view.tiles=drawn.data();view.tile_count=3;view.world_width_tiles=view.world_height_tiles=20;
 UnitArrivalVisibility arrival;arrival.capture(view,1);arrival.sample(view,1);
 // Civ III reveals both: (6,4) as drawn, (5,5) with a city it had captured
 // but the ring withheld (city bodies are known once explored).
 drawn[2].city_id=9;drawn[2].city_owner_id=2;view.tiles=drawn.data();
 auto native=drawn;for(int i=1;i<3;++i)native[i].tile_flags=known|explored|visible;
 c3x_renderer_frame_v1 now=view;now.tiles=native.data();arrival.capture(now,1);
 auto shown=arrival.sample(view,1);
 for(int i=1;i<3;++i)assert(!(shown.tiles[i].tile_flags&explored));
 assert(arrival.held_undrawn==2);
 // A view drawing the city shows the reveal, whole.
 auto next=native;c3x_renderer_frame_v1 later=view;later.tiles=next.data();
 shown=arrival.sample(later,1);
 for(int i=1;i<3;++i)assert(shown.tiles[i].tile_flags&visible);
 std::printf("PASS whole reveal: city_waits=1 beside_waits=1\n");
}
''')

    def test_window_draws_the_ring_on_screen(self):
        run_cpp(r'''
#include "Renderer/native/render_core/world_window.h"
#include <cassert>
#include <cstdio>
#include <map>
#include <vector>
using namespace c3x_renderer::render_core;
constexpr unsigned known=C3X_RENDERER_TILE_VISIBILITY_KNOWN,explored=C3X_RENDERER_TILE_EXPLORED;
int main(){
 c3x_renderer_frame_v1 f{};f.world_width_tiles=f.world_height_tiles=40;f.tile_width=128;f.tile_height=64;
 f.target_width=1280;f.target_height=640;
 std::vector<c3x_renderer_tile_v1> native;
 for(int y=0;y<20;++y)for(int x=y&1;x<20;x+=2){c3x_renderer_tile_v1 t{};t.tile_x=x;t.tile_y=y;
  t.anchor_x=x*64-64;t.anchor_y=y*32-32;
  t.tile_flags=C3X_RENDERER_TILE_RENDER|known|(x==10&&y==10?explored:0u);native.push_back(t);}
 f.tiles=native.data();f.tile_count=unsigned(native.size());
 // Retained world: the explored tile, and its ring drawn as explored terrain
 // (window_tile takes the ring from its object-free world input).
 auto ring=[](int x,int y){int dx=x-10,dy=y-10;return (dx||dy) && std::abs(dx+dy)<=4 && std::abs(dy-dx)<=4;};
 WorldWindow window;
 assert(window.build(f,[&](int x,int y,c3x_renderer_tile_v1& tile){
  if(!ring(x,y))return false;tile={};tile.tile_x=x;tile.tile_y=y;tile.tile_flags=known|explored;return true;}));
 unsigned drawn=0,rings=0;
 for(unsigned i=0;i<window.window_count;++i){auto const& t=window.tiles[i];
  if(!(t.tile_flags&explored))continue;++drawn;rings+=ring(t.tile_x,t.tile_y);}
 std::printf("window drawn=%u ring=%u\n",drawn,rings);
 assert(drawn==25 && rings==24);
}
''')


if __name__ == '__main__':
    unittest.main()
