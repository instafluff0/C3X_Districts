// Bounded exact shadow preparation work witness; no GPU or game required.

#include "Renderer/native/render_core/shadow_page_contents.h"
#include <cassert>
#include <iostream>
#include <chrono>
using namespace c3x_renderer::render_core;
using R=RasterDependencyRevisions;using Key=std::array<std::uint64_t,20>;using Pages=ShadowPageContents<Key>;
struct Proof {unsigned tile=0,value=7;};
struct Input {std::vector<unsigned> values;};
void broad_journal_validation(){
 struct Source {unsigned id=0;std::vector<R::Key> keys;};
 struct Body {unsigned tile=0,value=7;std::vector<std::weak_ptr<Source>> inputs;};
 using Proofs=ShadowCasterProofs<Body>;Proofs proofs;R revisions;
 constexpr unsigned count=128,sources_per_body=8;
 std::vector<std::shared_ptr<Body>> bodies;std::vector<std::shared_ptr<Source>> owners;
 std::vector<unsigned> content(count,7),visibility(count,3),source_content(count*sources_per_body,7);
 unsigned checks=0,source_checks=0;
 auto valid=[&](Body const& body){++checks;bool result=body.value==content[body.tile];
  for(auto const& weak:body.inputs){auto input=weak.lock();++source_checks;result=input&&source_content[input->id]==7&&result;}return result;};
 auto watch=[&](Body const& body,auto& target){
  if(!target.watch(R::Domain::semantic,body.tile)||!target.watch(R::Domain::visibility,body.tile))return false;
  for(auto const& weak:body.inputs){auto input=weak.lock();if(!input||!target.watch_source(input,[&](auto const& source){
   for(auto const& key:source.keys)if(!target.watch(key.domain,key.id))return false;return true;}))return false;}return true;};
 for(unsigned p=0;p<count;++p){auto body=std::make_shared<Body>();body->tile=p;
  for(unsigned s=0;s<sources_per_body;++s){auto input=std::make_shared<Source>();input->id=p*sources_per_body+s;
   for(unsigned n=0;n<285;++n){input->keys.push_back({R::Domain::world,n});input->keys.push_back({R::Domain::flow,n});}
   body->inputs.push_back(input);owners.push_back(input);}bodies.push_back(body);}
 auto bind=[&]{proofs.begin({1,2,3,1},revisions,[](auto bytes){return bytes<=Proofs::limit;});
  for(unsigned p=0;p<count;++p)assert(proofs.add(p+1,bodies[p],p,visibility[p],watch,valid));proofs.finish();};
 auto validate=[&]{return proofs.validate(revisions,valid,[&](auto tile){return visibility[tile];});};
 auto events=[&](unsigned n,unsigned offset=0){for(unsigned i=0;i<n;++i)revisions.touch(R::Domain::visibility,offset+i);};
 bind();assert(proofs.sources.size()==count*sources_per_body&&proofs.dependency_lists.size()==1);
 assert(proofs.dependency_lists.begin()->second.keys.size()==570);auto bytes=proofs.bytes();
 auto registrations=proofs.validation_counts.proof_registrations,watches=proofs.validation_counts.dependency_watch_calls;
 auto before=checks;assert(validate()&&checks==before); // no-change fast path
 events(16);assert(validate()&&checks==before+16); // sparse boundary
 before=checks;events(17);assert(validate()&&checks==before+count); // full boundary
 before=checks;events(625);assert(validate()&&checks==before+count);
 before=checks;assert(validate()&&checks==before);
 // Broad unrelated visibility events still require exact content checks, so
 // even a producer not selected by those keys cannot hide changed content.
 content.back()=8;before=checks;events(17,10000);assert(!validate()&&checks==before+count);
 content.back()=7;before=checks;assert(validate()&&checks==before+1);
 // A changed distinct source is rejected and recovers without registration.
 source_content.back()=8;revisions.touch(R::Domain::flow,42);before=checks;
 assert(!validate()&&checks==before+count);source_content.back()=7;revisions.touch(R::Domain::world,42);assert(validate());
 ++visibility.back();before=checks;events(625);assert(!validate()&&checks==before+count);
 bind();before=checks;assert(validate()&&checks==before+1);
 before=checks;events(16,10000);assert(validate()&&checks==before);
 before=checks;revisions.invalidate();assert(validate()&&checks==before+count);
 before=checks;for(unsigned n=0;n<R::capacity+1;++n)revisions.touch(R::Domain::world,999999);
 assert(validate()&&checks==before+count);
 R different;before=checks;assert(proofs.validate(different,valid,[&](auto tile){return visibility[tile];})&&checks==before+count);
 assert(validate()); // rebind the journal before testing the broad-event race
 // A callback racing a broad journal cannot advance the validated receipt.
 auto prior=proofs.checkpoint;events(17);bool changed=false;
 assert(!proofs.validate(revisions,[&](Body const& body){if(!changed){revisions.touch(R::Domain::visibility,999999);changed=true;}return valid(body);},[&](auto tile){return visibility[tile];}));
 assert(proofs.checkpoint.owner==prior.owner&&proofs.checkpoint.sequence==prior.sequence);assert(validate());
 before=checks;proofs.complete=false;assert(!validate()&&checks==before);proofs.complete=true;
 assert(proofs.bytes()==bytes&&proofs.validation_counts.proof_registrations==registrations&&proofs.validation_counts.dependency_watch_calls==watches);
 assert(source_checks>=checks*sources_per_body); // every checked body checks all immutable owners
 std::vector<std::weak_ptr<Source>> pinned;for(auto const& input:owners)pinned.push_back(input);owners.clear();
 for(auto const& weak:pinned)assert(!weak.expired());proofs.clear();for(auto const& weak:pinned)assert(weak.expired());
 assert(proofs.bytes()==sizeof(proofs));
 std::cout<<"SHADOW_BROAD_JOURNAL_WITNESS sparse16=1 full17=1 broad625=1 exact_content_and_visibility=1 source_owner_recovery=1 unchanged_fastpath=1 journal_race_rejected=1 ownership_and_caps_unchanged=1\n";
}
void current_visibility_rebinding(){
 using Proofs=ShadowCasterProofs<Proof>;Proofs proofs;R revisions;
 auto proof=std::make_shared<Proof>(Proof{1,7});auto source=std::make_shared<Input>();source->values={11,12};
 unsigned content=7,observed=3,checks=0;auto unlimited=[](auto){return true;};
 auto valid=[&](Proof const& body){++checks;return body.value==content;};
 auto watch=[&](Proof const& body,auto& target){return target.watch(R::Domain::semantic,body.tile)&&target.watch(R::Domain::visibility,body.tile)&&
  target.watch_source(source,[&](auto const& input){for(auto id:input.values)if(!target.watch(R::Domain::world,id))return false;return true;});};
 auto bind=[&](unsigned tile,unsigned visibility){proofs.begin({1,2,3,1},revisions,unlimited);proofs.mark(1);proofs.finish(false);
  assert(proofs.add(1,proof,tile,visibility,watch,valid));proofs.finish();};
 auto validate=[&]{return proofs.validate(revisions,valid,[&](auto){return observed;});};
 bind(1,observed);assert(validate());auto registrations=proofs.validation_counts.proof_registrations;
 auto watches=proofs.validation_counts.dependency_watch_calls,expansions=proofs.validation_counts.source_expansions;
 // A current selection supplies a new visibility snapshot for the same
 // immutable content. Rebinding must force exact content and visibility checks.
 for(unsigned next:{4u,9u,0u}){observed=next;auto before=checks;bind(1,observed);
  assert(proofs.producers.at(1).visibility==observed&&!proofs.producers.at(1).valid&&!proofs.valid_all);
  assert(validate()&&checks==before+1&&proofs.valid_all);
  auto checked=checks;bind(1,observed);assert(validate()&&checks==checked);}
 assert(proofs.validation_counts.proof_registrations==registrations&&proofs.validation_counts.dependency_watch_calls==watches&&proofs.validation_counts.source_expansions==expansions);
 // Neither accepting current visibility nor a missing journal entry can hide
 // changed content or a different visibility returned by validation.
 content=8;observed=5;bind(1,observed);assert(!validate()&&!proofs.valid_all);
 content=7;assert(validate());bind(1,6);assert(!validate());observed=6;assert(validate());
 observed=7;revisions.touch(R::Domain::visibility,1);assert(!validate());
 bind(1,observed);assert(validate()&&proofs.validation_counts.proof_registrations==registrations);
 // A supplied tile moving under the same immutable owner is inconsistent.
 // Reject it; a consistent new proof recovers with new visibility watch keys.
 proofs.begin({1,2,3,1},revisions,unlimited);proofs.mark(1);proofs.finish(false);
 assert(!proofs.add(1,proof,2,observed,watch,valid)&&!proofs.complete&&!validate());
 std::weak_ptr<Proof> old=proof;proof=std::make_shared<Proof>(Proof{2,7});bind(2,observed);
 assert(old.expired()&&validate());assert(proofs.producers.at(1).tile==2);
 assert(proofs.validation_counts.proof_registrations==registrations+1);
 auto const& dependencies=proofs.producers.at(1).dependencies;
 assert(std::find(dependencies.begin(),dependencies.end(),R::Key{R::Domain::visibility,1})==dependencies.end());
 assert(std::find(dependencies.begin(),dependencies.end(),R::Key{R::Domain::visibility,2})!=dependencies.end());
 auto before=checks;revisions.touch(R::Domain::visibility,1);assert(validate()&&checks==before);
 revisions.touch(R::Domain::visibility,2);assert(validate()&&checks==before+1);
 old=proof;proof=std::make_shared<Proof>(Proof{2,7});bind(2,observed);
 assert(old.expired()&&validate()&&proofs.validation_counts.proof_registrations==registrations+2);
 proofs.clear();assert(proofs.bytes()==sizeof(proofs));
 std::cout<<"SHADOW_VISIBILITY_REBINDING_WITNESS same_owner_visibility_rebind=1 exact_checks_required=1 unchanged_registration_and_watches=1 content_mismatch_rejected=1 inconsistent_tile_rejected=1 consistent_owner_recovers=1\n";
}
// Backing restore decodes distinct immutable owners even when river page
// footprints match. The old per-owner vectors cross16MiB in this busy shape.
void decoded_owner_dependencies(){
 struct Decoded {unsigned id=0;std::vector<R::Key> keys;};
 struct Body {unsigned tile=0;std::vector<std::weak_ptr<Decoded>> inputs;};
 using Proofs=ShadowCasterProofs<Body>;Proofs proofs;R revisions;
 std::vector<std::shared_ptr<Decoded>> owners;std::vector<std::shared_ptr<Body>> bodies;
 std::vector<unsigned> current(1050,7);unsigned body_checks=0,owner_checks=0;std::size_t peak=0;
 auto admit=[&](std::size_t bytes){peak=std::max(peak,bytes);return bytes<=Proofs::limit;};
 auto decoded=[&](unsigned id){auto value=std::make_shared<Decoded>();value->id=id;
  for(unsigned n=0;n<569;++n)value->keys.push_back({n%2?R::Domain::flow:R::Domain::world,n/2});
  if(id%2)std::reverse(value->keys.begin(),value->keys.end());return value;};
 for(unsigned n=0;n<107;++n)bodies.push_back(std::make_shared<Body>(Body{n,{}}));
 for(unsigned n=0;n<1049;++n){owners.push_back(decoded(n));bodies[n%107]->inputs.push_back(owners.back());}
 auto valid=[&](Body const& body){++body_checks;bool result=true;
  for(auto const& weak:body.inputs){auto owner=weak.lock();++owner_checks;result=owner&&current[owner->id]==7&&result;}return result;};
 auto watch=[&](Body const& body,auto& owner){
  if(!owner.watch(R::Domain::semantic,body.tile)||!owner.watch(R::Domain::visibility,body.tile))return false;
  for(auto const& weak:body.inputs){auto input=weak.lock();if(!input || !owner.watch_source(input,[&](auto const& source){
   for(auto const& key:source.keys)if(!owner.watch(key.domain,key.id))return false;return true;}))return false;}
  return true;};
 auto register_all=[&]{proofs.begin({1,2,3,1},revisions,admit);
  for(unsigned n=0;n<107;++n)proofs.mark(n+1);proofs.finish(false);
  for(unsigned n=0;n<107;++n)assert(proofs.add(n+1,bodies[n],n,3,watch,valid));proofs.finish();};
 register_all();assert(body_checks==107&&owner_checks==1049);
 assert(proofs.sources.size()==1049&&proofs.dependency_lists.size()==1);
 auto const& shared=proofs.dependency_lists.begin()->second;
 assert(shared.refs==1049&&shared.keys.size()==569&&shared.keys.capacity()==1024);
 assert(proofs.validation_counts.source_expansions==1049&&proofs.validation_counts.source_reuses==0);
 assert(std::size_t(1049)*1024*sizeof(R::Key)>Proofs::limit&&peak<1024u*1024u);
 for(auto const& input:owners){auto const& source=proofs.sources.at(input.get());
  assert(source.owner.get()==input.get()&&source.dependencies==&shared&&source.refs==1);}
 assert(proofs.validate(revisions,valid,[](auto){return 3;}));
 auto checked=body_checks,checked_owners=owner_checks;auto watches=proofs.validation_counts.dependency_watch_calls;
 for(unsigned n=0;n<100;++n){register_all();assert(proofs.validate(revisions,valid,[](auto){return 3;}));}
 assert(body_checks==checked&&owner_checks==checked_owners&&proofs.validation_counts.dependency_watch_calls==watches);
 revisions.touch(R::Domain::world,999999);assert(proofs.validate(revisions,valid,[](auto){return 3;}));assert(body_checks==checked);
 // A shared journal watch still validates every distinct proof/owner; one
 // owner's changed authoritative value cannot hide behind identical keys.
 current[731]=8;revisions.touch(R::Domain::world,42);assert(!proofs.validate(revisions,valid,[](auto){return 3;}));
 assert(body_checks==checked+107&&owner_checks==checked_owners+1049);
 current[731]=7;revisions.touch(R::Domain::world,42);assert(proofs.validate(revisions,valid,[](auto){return 3;}));
 checked=body_checks;revisions.touch(R::Domain::semantic,3);assert(proofs.validate(revisions,valid,[](auto){return 3;}));assert(body_checks==checked+1);
 // Replacement retains the old distinct owner only until the registration
 // transaction closes; the interned list remains owned by the others.
 std::weak_ptr<Decoded> old=owners[0];auto replacement=decoded(1049);owners[0]=replacement;
 auto next_body=std::make_shared<Body>(*bodies[0]);next_body->inputs[0]=replacement;bodies[0]=next_body;
 register_all();assert(old.expired()&&proofs.sources.size()==1049&&proofs.dependency_lists.size()==1&&shared.refs==1049);
 assert(proofs.validate(revisions,valid,[](auto){return 3;}));
 std::vector<std::weak_ptr<Decoded>> live;for(auto const& owner:owners)live.push_back(owner);
 owners.clear();replacement.reset();for(auto const& weak:live)assert(!weak.expired());
 // Pruning removes the last owner/list immediately, with no retained history.
 proofs.begin({1,2,3,1},revisions,admit);proofs.finish();for(auto const& weak:live)assert(weak.expired());
 assert(proofs.sources.empty()&&proofs.dependency_lists.empty()&&proofs.bytes()==sizeof(proofs));
 auto busy_peak=peak;
 // Deliberate digest collision: the domain/id vector must still compare in
 // full. Distinct world/flow watches remain distinct invalidation footprints.
 auto a=std::make_shared<Decoded>(),b=std::make_shared<Decoded>(),duplicate=std::make_shared<Decoded>();
 constexpr std::uint64_t offset=14695981039346656037ull,prime=1099511628211ull;
 std::uint64_t collision_id=11^((offset^std::uint64_t(R::Domain::world))*prime)^((offset^std::uint64_t(R::Domain::flow))*prime);
 a->keys={{R::Domain::world,11}};b->keys={{R::Domain::flow,collision_id}};duplicate->keys={a->keys[0],a->keys[0]};
 assert(Proofs::dependency_hash(a->keys)==Proofs::dependency_hash(b->keys));
 auto one_source=[&](std::shared_ptr<Decoded> const& input){return [&,input](auto const&,auto& target){return target.watch_source(input,[&](auto const& source){for(auto const& key:source.keys)if(!target.watch(key.domain,key.id))return false;return true;});};};
 auto proof=std::make_shared<Body>();proofs.begin({1,2,3,1},revisions,admit);
 assert(proofs.add(1,proof,1,3,one_source(a),[](auto const&){return true;}));
 assert(proofs.add(2,proof,2,3,one_source(b),[](auto const&){return true;}));
 assert(proofs.add(3,proof,3,3,one_source(duplicate),[](auto const&){return true;}));proofs.finish();
 assert(proofs.dependency_lists.size()==2&&proofs.sources.at(a.get()).dependencies==proofs.sources.at(duplicate.get()).dependencies);
 assert(proofs.sources.at(a.get()).dependencies!=proofs.sources.at(b.get()).dependencies);
 unsigned collision_checks=0;revisions.touch(R::Domain::world,11);
 assert(proofs.validate(revisions,[&](auto const&){++collision_checks;return true;},[](auto){return 3;}));assert(collision_checks==2);
 revisions.touch(R::Domain::flow,collision_id);assert(proofs.validate(revisions,[&](auto const&){++collision_checks;return true;},[](auto){return 3;}));assert(collision_checks==3);
 // Unequal lists sharing a domain also cannot be interned together.
 auto different=std::make_shared<Decoded>();different->keys={{R::Domain::world,12}};
 proofs.begin({1,2,3,1},revisions,admit);proofs.mark(1);proofs.mark(2);proofs.mark(3);
 assert(proofs.add(4,proof,4,3,one_source(different),[](auto const&){return true;}));proofs.finish();assert(proofs.dependency_lists.size()==3);
 // Context change destroys all exact owner pins and interned watch storage.
 std::weak_ptr<Decoded> context_owner=different;a.reset();b.reset();duplicate.reset();different.reset();
 proofs.begin({2,2,3,1},revisions,admit);assert(context_owner.expired()&&proofs.sources.empty()&&proofs.dependency_lists.empty()&&proofs.bytes()==sizeof(proofs));proofs.finish();
 // Required temporary growth and unique-list metadata must each pass the
 // unchanged admission rule. Failed registration releases owner/scratch/list.
 auto denied=std::make_shared<Decoded>();denied->keys={{R::Domain::world,1}};
 auto before_list=sizeof(proofs)+sizeof(Proofs::Producer)+sizeof(std::uint64_t)+64+
  sizeof(Proofs::Source)+sizeof(void const*)+64+4*sizeof(R::Key);
 auto list_node=sizeof(Proofs::DependencyList)+sizeof(std::uint64_t)+64;
 for(auto ceiling:{before_list+list_node-1,before_list+list_node}){
  proofs.begin({2,2,3,1},revisions,[=](std::size_t bytes){return bytes<=ceiling;});
  assert(!proofs.add(1,proof,1,3,one_source(denied),[](auto const&){return true;}));
  assert(!proofs.complete&&proofs.sources.empty()&&proofs.producers.empty()&&proofs.dependency_lists.empty()&&!proofs.source_scratch.capacity()&&proofs.bytes()==sizeof(proofs));
 }
 // A callback refusal also cannot leave a zero-reference immutable owner.
 proofs.begin({2,2,3,1},revisions,admit);
 assert(!proofs.add(1,proof,1,3,[&](auto const&,auto& target){return target.watch_source(denied,[&](auto const&){target.watch(R::Domain::world,1);return false;});},[](auto const&){return true;}));
 assert(proofs.sources.empty()&&proofs.dependency_lists.empty()&&proofs.bytes()==sizeof(proofs));
 proofs.begin({2,2,3,1},revisions,[&](std::size_t bytes){return bytes<=sizeof(proofs)+1024;});
 auto too_many=[&](auto const&,auto& target){return target.watch_source(denied,[&](auto const&){for(unsigned n=0;n<10000;++n)if(!target.watch(R::Domain::world,n))return false;return true;});};
 assert(!proofs.add(1,proof,1,3,too_many,[](auto const&){return true;}));
 assert(!proofs.complete&&proofs.sources.empty()&&proofs.producers.empty()&&proofs.dependency_lists.empty()&&!proofs.source_scratch.capacity()&&proofs.bytes()==sizeof(proofs));
 proofs.begin({2,2,3,1},revisions,[](auto){return true;});
 auto hard_limit=[&](auto const&,auto& target){return target.watch_source(denied,[&](auto const&){for(unsigned n=0;n<2u*1024u*1024u;++n)if(!target.watch(R::Domain::world,n))return false;return true;});};
 assert(!proofs.add(1,proof,1,3,hard_limit,[](auto const&){return true;}));
 assert(!proofs.complete&&proofs.sources.empty()&&proofs.producers.empty()&&proofs.dependency_lists.empty()&&proofs.bytes()==sizeof(proofs));
 proofs.clear();std::cout<<"SHADOW_DECODED_OWNER_WITNESS producers=107 owners=1049 unique_keys=569 key_capacity=1024 old_vector_bytes="<<std::size_t(1049)*1024*sizeof(R::Key)
  <<" interned_lists=1 peak_metadata="<<busy_peak<<" cap="<<Proofs::limit<<" journal_checks_all_owners=1 digest_collision_full_equality=1 rollback_release=1\n";
}
void compact_page_membership(){
 static_assert(sizeof(Pages::OccurrenceId)==4,"compact membership ID");
 static_assert(sizeof(Pages::Occurrence)==32,"ID uses existing occurrence padding");
 constexpr std::size_t proof_bytes=56u*1024u*1024u/10u,cap=16u*1024u*1024u;
 Pages pages;ShadowSamplingGrid grid;grid.valid=true;grid.low={0,0};grid.count={5,5};grid.quality_span={40,40};
 Pages::Context context{1,2,3,1};std::array<float,12> light{};std::size_t peak=0;
 auto admit=[&](std::size_t bytes){peak=std::max(peak,proof_bytes+bytes);return proof_bytes<=cap&&bytes<=cap-proof_bytes;};
 auto bounds=[&](unsigned n){float x=float(n%5)*10+9.8f,y=float(n/5%5)*10+9.8f;return std::array<float,4>{x,y,x+.4f,y+.4f};};
 Pages::Inputs exact;pages.begin_incremental(grid,context,light);
 for(unsigned n=0;n<13000;++n){Key key{n+1,7};auto b=bounds(n);
  assert(pages.update(key,grid,[&]{return b;},admit));
  for(unsigned slot=0;slot<grid.pages();++slot)if(Pages::intersects(b,grid.page_box(slot)))exact[slot].push_back(key);
 }
 assert(pages.finish_incremental(grid,true,admit));std::size_t memberships=0,capacity=0;
 for(unsigned slot=0;slot<grid.pages();++slot){auto physical=pages.slots[slot];
  assert(pages.exact_inputs(physical,exact[slot]));assert(pages.complete_incremental(slot));
  memberships+=pages.pages[physical].contributors.size();capacity+=pages.pages[physical].contributors.capacity();}
 // Independently reconstruct the previous full-key page allocation with the
 // identical map records/vector capacities. Combined pressure exceeds16MiB.
 auto old_page_bytes=pages.bytes()+capacity*(sizeof(Key)-sizeof(Pages::OccurrenceId));
 assert(proof_bytes+old_page_bytes>cap&&proof_bytes+pages.bytes()<=cap&&peak<=cap);
 auto projections=pages.projections,edits=pages.contributor_edits,sorts=pages.page_sorts;
 for(unsigned repeat=0;repeat<4;++repeat){pages.begin_incremental(grid,context,light);
  for(unsigned n=0;n<13000;++n)pages.mark(Key{n+1,7});assert(pages.retire_missing(admit));
  for(unsigned n=0;n<13000;++n)assert(pages.update(Key{n+1,7},grid,[&]{return bounds(n);},admit));
  assert(pages.finish_incremental(grid,true,admit));for(unsigned slot=0;slot<grid.pages();++slot)assert(pages.reused[slot]);}
 assert(pages.projections==projections&&pages.contributor_edits==edits&&pages.page_sorts==sorts);
 auto compact_bytes=pages.bytes();pages.clear();assert(pages.bytes()==sizeof(pages)&&pages.last_id==0);
 auto unlimited=[](auto){return true;};Key a{1,7},b=a;b.back()=1;
 auto wide=std::array<float,4>{9,9,11,11};
 auto prepare=[&](Key const& key){pages.begin_incremental(grid,context,light);pages.mark(key);assert(pages.retire_missing(unlimited));
  assert(pages.update(key,grid,[&]{return wide;},unlimited));assert(pages.finish_incremental(grid,true,unlimited));};
 prepare(a);auto first_id=pages.occurrences.at(a).id;
 for(unsigned slot=0;slot<grid.pages();++slot)pages.complete_incremental(slot);
 // Removing an occurrence erases its IDs before exact Key deletion. A later
 // equal Key gets a fresh ID; a different full20-word Key never aliases it.
 pages.begin_incremental(grid,context,light);assert(pages.retire_missing(unlimited));assert(pages.finish_incremental(grid,true,unlimited));
 assert(pages.occurrences.empty());for(auto const& page:pages.pages)assert(page.contributors.empty());
 prepare(a);assert(pages.occurrences.at(a).id>first_id);auto second_id=pages.occurrences.at(a).id;
 prepare(b);assert(!pages.occurrences.count(a)&&pages.occurrences.at(b).id>second_id);
 for(unsigned slot=0;slot<grid.pages();++slot){auto expected=Pages::intersects(wide,grid.page_box(slot))?std::vector<Key>{b}:std::vector<Key>{};
  assert(pages.exact_inputs(pages.slots[slot],expected));assert(!pages.exact_inputs(pages.slots[slot],std::vector<Key>{a}));pages.complete_incremental(slot);}
 // An interrupted multi-page append can retry with the same exact ID and no
 // duplicate membership; no incomplete page becomes a reused pixel proof.
 pages.clear();pages.begin_incremental(grid,context,light);unsigned admissions=0;
 assert(!pages.update(a,grid,[&]{return wide;},[&](auto){return ++admissions<=2;}));
 assert(pages.occurrences.count(a));auto partial_id=pages.occurrences.at(a).id;
 std::uint32_t partial_mask=0;for(unsigned physical=0;physical<Pages::Grid::max_pages;++physical)
  if(!pages.pages[physical].contributors.empty()){assert(pages.pages[physical].contributors==std::vector<Pages::OccurrenceId>{partial_id});partial_mask|=1u<<physical;}
 assert(partial_mask&&pages.occurrences.at(a).pages==partial_mask);
 // A key leaving after a failed append must remove every successful partial
 // insertion, even though no finished page/membership proof was published.
 pages.begin_incremental(grid,context,light);assert(pages.retire_missing(unlimited));assert(pages.finish_incremental(grid,true,unlimited));
 assert(pages.occurrences.empty());for(auto const& page:pages.pages)assert(page.contributors.empty());
 pages.clear();pages.begin_incremental(grid,context,light);admissions=0;
 assert(!pages.update(a,grid,[&]{return wide;},[&](auto){return ++admissions<=2;}));partial_id=pages.occurrences.at(a).id;
 assert(pages.update(a,grid,[&]{assert(false);return wide;},unlimited));
 for(unsigned slot=0;slot<grid.pages();++slot){auto const& ids=pages.pages[pages.slots[slot]].contributors;
  assert(ids.size()==unsigned(Pages::intersects(wide,grid.page_box(slot))));}
 pages.begin_incremental(grid,context,light);assert(pages.update(a,grid,[&]{return wide;},unlimited));assert(pages.finish_incremental(grid,true,unlimited));
 assert(pages.occurrences.at(a).id==partial_id);
 for(unsigned slot=0;slot<grid.pages();++slot){auto expected=Pages::intersects(wide,grid.page_box(slot))?std::vector<Key>{a}:std::vector<Key>{};
  assert(pages.exact_inputs(pages.slots[slot],expected));assert(!pages.reused[slot]);pages.complete_incremental(slot);}
 // A new key can fail after one insertion in an otherwise completed warm
 // grid. Its partial mask is not a complete shortcut for same-pass retry.
 pages.clear();pages.begin_incremental(grid,context,light);
 for(unsigned n=1;n<=4;++n)assert(pages.update(Key{n,7},grid,[&]{return wide;},unlimited));
 assert(pages.finish_incremental(grid,true,unlimited));for(unsigned slot=0;slot<grid.pages();++slot)pages.complete_incremental(slot);
 pages.begin_incremental(grid,context,light);assert(pages.same_page_grid);
 for(unsigned n=1;n<=4;++n)pages.mark(Key{n,7});assert(pages.retire_missing(unlimited));
 admissions=0;Key entering{5,7};assert(!pages.update(entering,grid,[&]{return wide;},[&](auto){return ++admissions<=2;}));
 assert(pages.update(entering,grid,[&]{assert(false);return wide;},unlimited));
 for(unsigned slot=0;slot<grid.pages();++slot){auto const& ids=pages.pages[pages.slots[slot]].contributors;
  assert(ids.size()==(Pages::intersects(wide,grid.page_box(slot))?5u:0u));}
 assert(pages.finish_incremental(grid,true,unlimited));
 for(unsigned slot=0;slot<grid.pages();++slot){std::vector<Key> expected;
  if(Pages::intersects(wide,grid.page_box(slot)))for(unsigned n=1;n<=5;++n)expected.push_back(Key{n,7});
  assert(pages.exact_inputs(pages.slots[slot],expected));pages.complete_incremental(slot);}
 // Exhaustion invalidates every completed page before any ID can restart.
 pages.last_id=UINT32_MAX;assert(!pages.update(b,grid,[&]{return wide;},unlimited));
 assert(pages.occurrences.empty()&&pages.last_id==0);for(auto const& page:pages.pages)assert(!page.valid&&page.contributors.empty());
 prepare(b);assert(pages.occurrences.at(b).id==1);for(unsigned slot=0;slot<grid.pages();++slot)assert(!pages.reused[slot]);
 for(unsigned slot=0;slot<grid.pages();++slot)pages.complete_incremental(slot);
 ++light[3];pages.begin_incremental(grid,context,light);assert(pages.occurrences.empty()&&pages.last_id==0);
 for(auto const& page:pages.pages)assert(!page.valid&&page.contributors.empty());
 pages.clear();
 // The legacy independent oracle also drops inactive page proofs/key owners.
 auto narrow=grid;narrow.count={1,1};Pages::Inputs inputs;inputs[0]={a};pages.select(narrow,context,inputs,true);
 assert(pages.complete(0,narrow,context,inputs[0]));narrow.low={1,0};inputs[0]={b};pages.select(narrow,context,inputs,true);
 assert(!pages.reused[0]&&pages.complete(0,narrow,context,inputs[0]));assert(!pages.occurrences.count(a)&&pages.occurrences.count(b));
 narrow.low={0,0};inputs[0]={a};pages.select(narrow,context,inputs,true);assert(!pages.reused[0]);
 pages.last_id=UINT32_MAX;assert(!pages.complete(0,narrow,context,inputs[0]));
 assert(pages.occurrences.empty());for(auto const& page:pages.pages)assert(!page.valid&&page.contributors.empty());
 // Forced oracle entries carry membership only, not a proved projection.
 // Switching back to incremental must recompute nonzero bounds and redraw.
 prepare(a);for(unsigned slot=0;slot<grid.pages();++slot)pages.complete_incremental(slot);
 Pages::Inputs legacy;legacy[18]={b};pages.select(grid,context,legacy,true);
 for(unsigned slot=0;slot<grid.pages();++slot)if(!pages.reused[slot])assert(pages.complete(slot,grid,context,legacy[slot]));
 auto actual_bounds=std::array<float,4>{35,35,36,36};unsigned computed=0;
 pages.begin_incremental(grid,context,light);pages.mark(b);assert(pages.retire_missing(unlimited));
 assert(pages.update(b,grid,[&]{++computed;return actual_bounds;},unlimited));assert(pages.finish_incremental(grid,true,unlimited));
 assert(computed==1&&*pages.projected(b)==actual_bounds);
 for(unsigned slot=0;slot<grid.pages();++slot){auto expected=Pages::intersects(actual_bounds,grid.page_box(slot))?std::vector<Key>{b}:std::vector<Key>{};
  assert(pages.exact_inputs(pages.slots[slot],expected)&&!pages.reused[slot]);}
 pages.clear();
 // Vector growth charges the new allocation while the old capacity is live.
 // A ceiling that admits only the final net delta must reject replacement.
 narrow=grid;narrow.count={1,1};pages.begin_incremental(narrow,context,light);
 for(unsigned n=0;n<4;++n)assert(pages.update(Key{n+1},narrow,[]{return std::array<float,4>{1,1,2,2};},unlimited));
 auto physical=pages.slots[0];assert(pages.pages[physical].contributors.capacity()==4);
 auto next_map_bytes=pages.bytes()+sizeof(Key)+sizeof(Pages::Occurrence)+64;
 auto final_only=next_map_bytes+4*sizeof(Pages::OccurrenceId);std::size_t requested=0;
 assert(!pages.update(Key{5},narrow,[]{return std::array<float,4>{1,1,2,2};},[&](auto bytes){requested=bytes;return bytes<=final_only;}));
 assert(requested==next_map_bytes+8*sizeof(Pages::OccurrenceId)&&pages.pages[physical].contributors.size()==4&&pages.pages[physical].contributors.capacity()==4);
 assert(pages.occurrences.at(Key{5}).pages==0);
 assert(pages.update(Key{5},narrow,[]{assert(false);return std::array<float,4>{};},unlimited));assert(pages.finish_incremental(narrow,true,unlimited));
 assert(pages.pages[physical].contributors.size()==5);pages.clear();
 std::cout<<"SHADOW_COMPACT_MEMBERSHIP_WITNESS occurrences=13000 memberships="<<memberships<<" capacity="<<capacity
  <<" proof_allowance="<<proof_bytes<<" old_page_bytes="<<old_page_bytes<<" compact_page_bytes="<<compact_bytes
  <<" peak_combined="<<peak<<" combined_cap="<<cap<<" unchanged_page_edits=0 exact_key_resolution=1 retirement_no_id_reuse=1 overflow_invalidates=1 replacement_overlap_charged=1\n";
}
struct Projection {struct Bounds {float low[3]{},high[3]{};};
    static std::array<float,4> project(Bounds const& b,float const* offset,std::array<float,12> const& projection) {
        std::array<float,4> out={1e9f,1e9f,-1e9f,-1e9f};
        for(unsigned mask=0;mask<8;++mask){float u=0,v=0;
            for(unsigned i=0;i<3;++i){float x=((mask>>i)&1?b.high[i]:b.low[i])+offset[i];u+=x*projection[i];v+=x*projection[4+i];}
            out[0]=std::min(out[0],u);out[1]=std::min(out[1],v);out[2]=std::max(out[2],u);out[3]=std::max(out[3],v);
        }return out;
    }
};
int main(){
 broad_journal_validation();
 current_visibility_rebinding();
 decoded_owner_dependencies();
 compact_page_membership();
 ShadowCasterProofs<Proof> proofs;R revisions;Pages pages;Pages::Context context{1,2,3,1};
 ShadowSamplingGrid grid;grid.valid=true;grid.low={0,0};grid.count={5,5};grid.quality_span={40,40};
 std::array<float,12> light={1,0,0,0,0,1,0,0,0,0,1,0};std::size_t peak=0,cap=ShadowCasterProofs<Proof>::limit;
 auto admit=[&](std::size_t n){peak=std::max(peak,n);return n<=cap;};
 auto source=std::make_shared<Input>();for(unsigned i=0;i<4096;++i)source->values.push_back(5000+i);
 std::vector<std::shared_ptr<Proof>> producers;std::vector<Key> keys;
 for(unsigned i=0;i<1000;++i){producers.push_back(std::make_shared<Proof>(Proof{i,7}));keys.push_back(Key{i+1,0});}
 unsigned checked=0;std::map<unsigned,unsigned> current;
 auto valid=[&](Proof const& p){++checked;auto found=current.find(p.tile);return p.value==(found==current.end()?7:found->second);};
 auto watch=[&](Proof const& p,auto& owner){return owner.watch(R::Domain::semantic,p.tile)&&owner.watch(R::Domain::visibility,p.tile)&&
  owner.watch_source(source,[&](auto const& input){for(auto id:input.values)if(!owner.watch(R::Domain::world,id)||!owner.watch(R::Domain::flow,id))return false;return true;});};
 auto register_all=[&]{proofs.begin({1,2,3,1},revisions,[&](auto bytes){return admit(bytes+pages.bytes());});
  for(auto const& key:keys)proofs.mark(key[0]);proofs.finish(false);
  for(auto const& key:keys)assert(proofs.add(key[0],producers[key[0]-1],key[0]-1,3,watch,valid));proofs.finish();
  return proofs.validate(revisions,valid,[](auto){return 3;});};
 auto bounds=[&](Key const& key){auto x=float((key[0]*7)%49),y=float((key[0]*11)%49);Projection::Bounds b{{x,y,0},{x+.3f,y+.4f,.8f}};float offset[3]{};return Projection::project(b,offset,light);};
 std::uint64_t forced_projections=0;
 auto update=[&]{pages.begin_incremental(grid,context,light);
  for(auto const& key:keys)pages.mark(key);assert(pages.retire_missing([&](auto n){return admit(n+proofs.bytes());}));
  for(auto const& key:keys)assert(pages.update(key,grid,[&]{return bounds(key);},[&](auto n){return admit(n+proofs.bytes());}));
  assert(pages.finish_incremental(grid,true,admit));Pages::Inputs exact;
  for(auto const& key:keys)for(unsigned i=0;i<grid.pages();++i){++forced_projections;if(Pages::intersects(bounds(key),grid.page_box(i)))exact[i].push_back(key);}
  for(unsigned i=0;i<grid.pages();++i){std::sort(exact[i].begin(),exact[i].end());assert(pages.exact_inputs(pages.slots[i],exact[i]));
   if(!pages.reused[i])assert(pages.complete_incremental(i));}
 };
 assert(register_all());assert(checked==1000 && proofs.validation_counts.source_expansions==1);update();auto cold_forced=forced_projections;
 auto projections=pages.projections,tests=pages.page_tests,edits=pages.contributor_edits,watches=proofs.validation_counts.dependency_watch_calls,sorts=pages.page_sorts;auto checks=checked;
 for(unsigned i=0;i<100;++i){assert(register_all());update();}
 auto warm_forced=forced_projections-cold_forced;
 assert(checked==checks && proofs.validation_counts.dependency_watch_calls==watches && pages.projections==projections && pages.page_tests==tests && pages.contributor_edits==edits && pages.page_sorts==sorts);
 // Only one new occurrence is projected; old and new affected pages match a
 // complete reconstruction while all other vectors remain untouched.
 keys.erase(keys.begin());producers.push_back(std::make_shared<Proof>(Proof{1000,7}));keys.push_back(Key{1001,0});
 assert(register_all());update();assert(pages.projections==projections+1 && pages.page_tests==tests+25 && checked==checks+1);
 assert(proofs.validation_counts.source_expansions==1 && proofs.producers.size()==1000);
 revisions.touch(R::Domain::world,999999);assert(proofs.validate(revisions,valid,[](auto){return 3;}));assert(checked==checks+1);
 revisions.touch(R::Domain::semantic,1);assert(proofs.validate(revisions,valid,[](auto){return 3;}));assert(checked==checks+2);
 current[1]=8;revisions.touch(R::Domain::semantic,1);assert(!proofs.validate(revisions,valid,[](auto){return 3;}));
 current[1]=7;revisions.touch(R::Domain::semantic,1);assert(proofs.validate(revisions,valid,[](auto){return 3;}));
 auto before=checked;for(unsigned i=0;i<R::capacity+1;++i)revisions.touch(R::Domain::world,999999);assert(proofs.validate(revisions,valid,[](auto){return 3;}));assert(checked==before+1000);
 before=checked;revisions.invalidate();assert(proofs.validate(revisions,valid,[](auto){return 3;}));assert(checked==before+1000);
 auto expansions=proofs.validation_counts.source_expansions;grid.low={1,-1};update();assert(pages.projections==projections+1);
 ++context[19];update();assert(pages.projections==projections+1); // wrap/page context, same exact light basis
 ++light[3];++context[7];update();assert(pages.projections==projections+1001);
 // Interrupted admission cannot publish an incomplete occurrence mask. Retry
 // reconstructs affected membership before completed pixels can be reused.
 Pages retry;retry.begin_incremental(grid,context,light);Key extra{90000};unsigned admits=0;
 assert(!retry.update(extra,grid,[]{return std::array<float,4>{10,10,10.2f,10.2f};},[&](auto){return ++admits==1;}));
 retry.begin_incremental(grid,context,light);assert(retry.update(extra,grid,[]{return std::array<float,4>{10,10,10.2f,10.2f};},admit));
 assert(retry.finish_incremental(grid,true,admit));for(unsigned i=0;i<grid.pages();++i)
  assert(retry.pages[retry.slots[i]].contributors.size()==unsigned(Pages::intersects({10,10,10.2f,10.2f},grid.page_box(i))));
 retry.clear();update();
 // Leaving slices are not maintained while outside the logical grid. An old
 // contributor removed there can never authorize reuse when that slice returns.
 Pages edge;auto small=grid;small.low={0,0};small.count={2,1};Key disappearing{1};
 edge.begin_incremental(small,context,light);assert(edge.update(disappearing,small,[]{return std::array<float,4>{1,1,2,2};},admit));
 assert(edge.finish_incremental(small,true,admit));for(unsigned i=0;i<small.pages();++i)edge.complete_incremental(i);
 small.low={1,0};edge.begin_incremental(small,context,light);assert(edge.finish_incremental(small,true,admit));for(unsigned i=0;i<small.pages();++i)edge.complete_incremental(i);
 small.low={0,0};edge.begin_incremental(small,context,light);assert(edge.finish_incremental(small,true,admit));
 assert(!edge.reused[0]&&edge.pages[edge.slots[0]].contributors.empty());edge.clear();
 // Optional metadata rejection never creates uncharged owner storage.
 ShadowCasterProofs<Proof> denied;denied.begin({1,2,3,1},revisions,[](auto){return false;});
 assert(!denied.add(1,producers[1],1,3,watch,valid));assert(denied.producers.empty()&&denied.sources.empty());denied.clear();
 auto projected_before=pages.projections;auto checked_before=checked;keys.back()[18]=41;
 assert(register_all());update();assert(pages.projections==projected_before+1&&checked==checked_before);
 assert(proofs.validation_counts.source_expansions==expansions);
 // Completely disjoint producer generations can still share an immutable
 // input owner. Its zero-reference registration survives only this transaction.
 keys.clear();while(producers.size()<3000)producers.push_back(std::make_shared<Proof>(Proof{unsigned(producers.size()),7}));
 for(unsigned i=0;i<1000;++i)keys.push_back(Key{2001+i});checked_before=checked;
 assert(register_all());update();assert(checked==checked_before+1000&&proofs.validation_counts.source_expansions==expansions);
 assert(proofs.producers.size()==1000&&proofs.sources.size()==1);
 // Fixed identical-work timing includes registration/validation and page
 // preparation, and excludes the independent equality oracle below. No timing
 // threshold is asserted; native route measurements remain the speed authority.
 constexpr unsigned timing_iterations=16;auto start=std::chrono::steady_clock::now();
 for(unsigned iteration=0;iteration<timing_iterations;++iteration){
  assert(register_all());pages.begin_incremental(grid,context,light);
  for(auto const& key:keys)pages.mark(key);assert(pages.retire_missing(admit));
  for(auto const& key:keys)assert(pages.update(key,grid,[&]{return bounds(key);},admit));
  assert(pages.finish_incremental(grid,true,admit));
 }
 auto retained_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
 RasterContributors<Proof,20> forced;Pages::Inputs forced_inputs;unsigned forced_content_checks=0;
 start=std::chrono::steady_clock::now();
 for(unsigned iteration=0;iteration<timing_iterations;++iteration){
  forced.clear();for(auto const& key:keys){bool new_proof=false;
   assert(forced.add(key,producers[key[0]-1],key[0]-1,3,&new_proof));if(new_proof)assert(watch(*producers[key[0]-1],forced));}
  forced.finish_dependencies();assert(forced.valid([&](Proof const& p){++forced_content_checks;auto found=current.find(p.tile);return p.value==(found==current.end()?7:found->second);},[](auto){return 3;}));
  std::array<std::size_t,ShadowSamplingGrid::max_pages> counts{};
  for(auto const& key:keys)for(unsigned i=0;i<grid.pages();++i)if(Pages::intersects(bounds(key),grid.page_box(i)))++counts[i];
  forced_inputs={};for(unsigned i=0;i<grid.pages();++i)forced_inputs[i].reserve(counts[i]);
  for(auto const& key:keys)for(unsigned i=0;i<grid.pages();++i)if(Pages::intersects(bounds(key),grid.page_box(i)))forced_inputs[i].push_back(key);
  for(unsigned i=0;i<grid.pages();++i)std::sort(forced_inputs[i].begin(),forced_inputs[i].end());
 }
 auto forced_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
 for(unsigned i=0;i<grid.pages();++i)assert(pages.exact_inputs(pages.slots[i],forced_inputs[i]));
 std::cout<<"SHADOW_PREPARATION_CPU iterations="<<timing_iterations<<" producers="<<keys.size()<<" pages="<<grid.pages()
  <<" retained_ms="<<retained_ms<<" forced_ms="<<forced_ms<<" forced_content_checks="<<forced_content_checks
  <<" forced_source_expansions="<<forced.validation_counts.source_expansions<<" equality_oracle_timing=excluded\n";
 forced.clear();auto total_projections=pages.projections;
 std::weak_ptr<Input> weak=source;source.reset();keys.clear();proofs.begin({1,2,3,1},revisions,admit);proofs.finish();assert(weak.expired()&&proofs.sources.empty()&&proofs.producers.empty());
 proofs.clear();pages.clear();assert(proofs.bytes()==sizeof(proofs)&&pages.bytes()==sizeof(pages)&&peak<cap);
 std::cout<<"SHADOW_PREPARATION_WITNESS initial_retained_projections="<<projections<<" initial_forced_projections="<<cold_forced
  <<" unchanged_retained_projections=0 unchanged_forced_projections="<<warm_forced<<" entering_retained_projections=1 entering_forced_projections=25000"
  <<" total_retained_projections="<<total_projections<<" total_forced_projections="<<forced_projections
  <<" unchanged_page_edits=0 unchanged_page_sorts=0 source_expansions=1 peak_metadata="<<peak<<"\n";
}
