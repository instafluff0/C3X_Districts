"""Bounded opt-in object-key observation without changing content admission."""
import unittest
from Renderer.native.native_cpp_test import ROOT, run_cpp


class WorldKeyDiagnosticTests(unittest.TestCase):
    def test_exact_keys_complete_counts_bounded_concurrent_witnesses(self):
        run_cpp(r'''
using UINT=unsigned;
#include "Renderer/native/world_preparation.h"
#include "Renderer/native/render_core/world_key_diagnostic.h"
#include <algorithm>
#include <cassert>
#include <thread>
using namespace c3x_renderer;
using render_core::WorldKeyDiagnostic;
int main(){
 WorldPreparationKey current{};current.identity[0]=12;current.identity[1]=8;
 current.identity[24]=std::uint64_t(WorldPreparationKind::objects);current.identity[17]=900;
 auto exact=current,revision=current,other=current,wrong_kind=current,wrong_tile=current;
 revision.identity[17]=100;other.identity[10]=3;wrong_kind.identity[24]=0;wrong_tile.identity[0]=14;
 std::vector<WorldPreparationKey> keys={wrong_kind,wrong_tile,revision,other};
 WorldKeyDiagnostic diagnostic;
 auto enumerate=[&](auto visit){for(auto const& key:keys)visit(key);};
 auto found=diagnostic.inspect(current,enumerate);
 assert(found.available && found.related && found.revision_only && found.nearest_recipe_empty);
 assert(found.nearest==other.identity && found.mask==(1ull<<10));
 assert(found.all_masks==((1ull<<10)|(1ull<<17)));
 assert(current==exact && keys[2]==revision);
 std::reverse(keys.begin(),keys.end());
 auto reversed=diagnostic.inspect(current,enumerate);
 assert(reversed.nearest==found.nearest && reversed.mask==found.mask);
 keys={revision};found=diagnostic.inspect(current,enumerate);
 assert(found.mask==(1ull<<17) && found.nearest==revision.identity);
 auto recipe=current;recipe.recipe.words={7};keys={recipe};found=diagnostic.inspect(current,enumerate);
 assert(found.related && !found.revision_only && found.mask==(1ull<<25) && !found.nearest_recipe_empty);
 keys={wrong_tile,wrong_kind};found=diagnostic.inspect(current,enumerate);
 assert(found.available && !found.related);
 auto failed=diagnostic.inspect(current,[](auto){throw 1;});assert(!failed.available);
 assert(diagnostic.counts[0]==6 && diagnostic.counts[1]==3 && diagnostic.counts[2]==1 &&
        diagnostic.counts[3]==1 && diagnostic.counts[4]==1);
 auto words=WorldKeyDiagnostic::words(current.identity);
 assert(words.size()==25*16+24 && std::count(words.begin(),words.end(),':')==24);
 assert(words.substr(0,16)=="000000000000000c");
 auto parts=WorldKeyDiagnostic::fact_parts(std::string(1400,'f'));std::string reconstructed;
 assert(parts.size()==3);for(auto const& part:parts){assert(part.size()<=480);reconstructed+=part;}
 assert(reconstructed==std::string(1400,'f'));
 diagnostic.reset();keys={revision};
 std::array<std::thread,4> lanes;
 std::atomic<unsigned> samples{0};
 for(auto& lane:lanes)lane=std::thread([&]{for(unsigned n=0;n<75;++n){
    auto match=diagnostic.inspect(current,enumerate);assert(match.available && match.revision_only);
    if(match.witness<WorldKeyDiagnostic::witness_limit)++samples;
 }});
 for(auto& lane:lanes)lane.join();
 assert(diagnostic.counts[0]==300 && diagnostic.counts[1]==300 && diagnostic.witnesses==16 && samples==16);
 assert(diagnostic.identity_mask==(1ull<<17));
 for(unsigned enabled=0;enabled<2;++enabled)for(unsigned component=0;component<2;++component)
  for(unsigned backing=0;backing<2;++backing)for(unsigned retain=0;retain<2;++retain)
   assert(WorldKeyDiagnostic::foreground(enabled!=0,component,backing!=0,retain!=0)==
          (enabled && component==1 && !backing && !retain));
}
''')

    def test_actual_publication_observes_before_replacement_without_scope_or_visibility_confusion(self):
        run_cpp(r'''
#include "Renderer/native/render_core/scene_publication.h"
#include "Renderer/native/render_core/world_key_diagnostic.h"
#include <cassert>
using namespace c3x_renderer::render_core;
int main(){
 CapturedScene measured,ordinary;
 ScenePublication journal,control;
 c3x_renderer_tile_v1 tile{};tile.tile_x=2;tile.tile_y=2;tile.terrain_type=2;tile.real_terrain_type=2;
 tile.city_id=-1;tile.resource_id=-1;tile.resource_class=-1;tile.barbarian_tribe_id=-1;
 tile.tile_flags=C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED|
    C3X_RENDERER_TILE_VISIBLE|C3X_RENDERER_TILE_PREFETCH;
 c3x_renderer_frame_v1 frame{};frame.world_width_tiles=8;frame.world_height_tiles=8;frame.tiles=&tile;frame.tile_count=1;
 c3x_renderer_camera_identity_v1 identity{};identity.map_epoch=identity.viewer_epoch=identity.scene_epoch=identity.visibility_epoch=1;
 unsigned calls=0;bool changed=false,control_changed=false;
 auto compare=[&]{
  auto a=measured.world_snapshot(),b=ordinary.world_snapshot();auto id=a->key(2,2);
  assert(a->scope_sequence()==b->scope_sequence());
  assert(a->current(id)->content_revision==b->current(id)->content_revision);
  assert(!std::memcmp(&a->current(id)->occurrence,&b->current(id)->occurrence,sizeof(tile)));
 };
 assert(journal.capture(frame,identity) && control.capture(frame,identity));
 assert(journal.apply(measured,changed,nullptr,[&](auto const&,auto const&){++calls;}));
 assert(control.apply(ordinary,control_changed));assert(calls==0);compare();
 auto prior=measured.world_snapshot();auto id=prior->key(2,2);auto revision=prior->current(id)->content_revision;
 tile.tile_flags=(tile.tile_flags&~C3X_RENDERER_TILE_PREFETCH)|C3X_RENDERER_TILE_RENDER;
 tile.anchor_x=500;tile.anchor_y=-30;tile.visibility_mask=9;tile.unit_type_id=77;
 assert(journal.capture(frame,identity) && control.capture(frame,identity));
 assert(journal.apply(measured,changed,nullptr,[&](auto const& scene,auto const& next){
   ++calls;auto before=CapturedScene::content(prior->current(id)->occurrence),after=CapturedScene::content(next);
   assert(WorldKeyDiagnostic::facts(before,after).empty());
   assert(scene.world_appearance_revision(id)==revision);
 }));
 assert(control.apply(ordinary,control_changed));assert(calls==1);compare();
 assert(measured.world_appearance_revision(id)==revision);prior.reset();
 prior=measured.world_snapshot();std::weak_ptr<CapturedScene::WorldSnapshot const> borrowed=prior;
 tile.resource_id=7;tile.territory_edge_mask=3;tile.territory_color_rgb=0x123456;tile.road_mask=1;
 std::strcpy(tile.resource_name,"source-label");
 assert(journal.capture(frame,identity) && control.capture(frame,identity));
 assert(journal.apply(measured,changed,nullptr,[&](auto const& scene,auto const& next){
   ++calls;auto before=CapturedScene::content(prior->current(id)->occurrence),after=CapturedScene::content(next);
   assert(before.resource_id==-1 && after.resource_id==7);
   assert(scene.world_appearance_revision(id)==revision);
   auto facts=WorldKeyDiagnostic::facts(before,after);
   assert(facts.find("resource_id:")!=std::string::npos && facts.find("road:")!=std::string::npos);
   assert(facts.find("territory_edges:")!=std::string::npos && facts.find("territory_rgb:")!=std::string::npos);
   assert(facts.find("resource_name_digest:")!=std::string::npos && facts.find("source-label")==std::string::npos);
   assert(facts.find("other_bytes:")==std::string::npos);
 }));
 assert(control.apply(ordinary,control_changed));assert(calls==2);compare();
 assert(measured.world_appearance_revision(id)!=revision);prior.reset();assert(borrowed.expired());
 ++identity.viewer_epoch;
 assert(journal.capture(frame,identity) && control.capture(frame,identity));
 assert(journal.apply(measured,changed,nullptr,[&](auto const&,auto const&){++calls;}));
 assert(control.apply(ordinary,control_changed));assert(calls==2);compare();
 auto a=CapturedScene::content(tile),b=a;b.city_site_grade=2;
 assert(WorldKeyDiagnostic::facts(a,b).find("other_bytes:")!=std::string::npos);
}
''')

    def test_actual_cpp_diagnostic_remains_opt_in_benchmark_only(self):
        source = (ROOT / 'Renderer/native/c3x_renderer.cpp').read_text()
        enabled = source.split('bool world_key_diagnostic_enabled()const {', 1)[1].split('\n    }', 1)[0]
        self.assertIn('C3X_RENDERER_WORLD_KEY_DIAGNOSTIC', enabled)
        self.assertIn('==1', enabled)
        self.assertIn("option[0]=='1' && option[1]==0", enabled)
        miss = source.split('++world_misses[component][present?6:0];', 1)[1].split('char census[8]', 1)[0]
        self.assertIn('#ifdef C3X_RENDERER_BENCHMARK_ORACLE', miss)
        self.assertIn('if(!present', miss)
        self.assertIn('component,input.backing_only,input.retain_prepared', miss)
        self.assertIn('diagnose_world_key_miss(input)', miss)
        diagnostic = source.split('void diagnose_world_key_miss(', 1)[1].split('void diagnose_world_key_facts(', 1)[0]
        self.assertIn('char detail[700]', diagnostic)
        self.assertIn('"world-key-diagnostic-current"', diagnostic)
        self.assertIn('"world-key-diagnostic-nearest"', diagnostic)
        facts = source.split('void diagnose_world_key_facts(', 1)[1].split('\n#endif', 1)[0]
        self.assertIn('WorldKeyDiagnostic::fact_parts(fields)', facts)
        self.assertIn('field_parts=%zu', facts)
        self.assertIn('"world-key-fact-diagnostic-part"', facts)
        adopt = source.split('if(scene_changes.ready()){', 1)[1].split('world_authoritative=', 1)[0]
        self.assertIn('world_initialization_scope==scope', adopt)
        self.assertIn('diagnostic_prior.reset();', adopt)
        self.assertIn('world_key_diagnostic.reset();', adopt)
        self.assertIn('scene_changes.apply(renderer_state.topology_cache,changed,&changed_tiles);', adopt)


if __name__ == '__main__':
    unittest.main()
