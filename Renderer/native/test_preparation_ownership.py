"""Queue-owned recipe/input allocations stay visible across dispatch and retirement."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class PreparationOwnershipTests(unittest.TestCase):
    def test_recipe_copies_active_inputs_and_full_ready_capacity_remain_distinct(self):
        run_cpp(r'''
#include "Renderer/native/render_core/content_preparation.h"
#include <cassert>
using namespace c3x_renderer::render_core;
struct Key {
 unsigned id=0;std::vector<std::uint64_t> words;
 bool operator<(Key const& other)const{return id<other.id;}
 bool operator==(Key const& other)const{return id==other.id;}
};
struct Input {Key key;std::vector<std::uint64_t> nodes;};
struct Result {unsigned id=0;std::size_t bytes()const{return 16u*1024u*1024u;}};
using Queue=ContentPreparation<Key,Input,Result>;
int main(){
 Queue queue;std::deque<Queue::Job> jobs;std::vector<Key> required;
 for(unsigned n=0;n<6;++n){Key key{n,std::vector<std::uint64_t>(64,n)};
  jobs.push_back({key,{key,std::vector<std::uint64_t>(17,n)}});required.push_back(key);}
 auto key_size=[](Key const& key){return key.words.capacity()*sizeof(std::uint64_t);};
 auto input_size=[&](Input const& input){return key_size(input.key)+input.nodes.capacity()*sizeof(std::uint64_t);};
 std::atomic<unsigned> entered{0};std::atomic<bool> release{false};std::array<std::atomic<unsigned>,6> calls{};
 auto compile=[&](Input const& input,std::atomic<bool> const& stop,unsigned){
  ++entered;while(!release&&!stop)std::this_thread::yield();
  if(stop)return std::unique_ptr<Result>{};
  assert(input.nodes==std::vector<std::uint64_t>(17,input.key.id));++calls[input.key.id];
  return std::make_unique<Result>(Result{input.key.id});};
 queue.configure(std::move(jobs),compile,4,required,{},64u*1024u*1024u,true);
 auto bare=queue.statistics();queue.set_owned_bytes(key_size,input_size);auto owned=queue.statistics();
 // Six required keys and six independent Job.key/Input.key copies, plus nodes.
 assert(owned.owner_metadata_bytes-bare.owner_metadata_bytes==18*512+6*17*8);
 assert(owned.bytes==0&&owned.reserved_bytes==0&&owned.capacity==64u*1024u*1024u);
 queue.resume();while(entered<4)std::this_thread::yield();
 auto active=queue.statistics();assert(active.active==4&&active.pending==2);
 assert(active.reserved_bytes==64u*1024u*1024u&&active.bytes==0);
 assert(active.owner_metadata_bytes>owned.owner_metadata_bytes);
 bool refused=false;try{queue.set_owned_bytes({},{});}catch(std::logic_error const&){refused=true;}assert(refused);
 release=true;
 auto limit=std::chrono::steady_clock::now()+std::chrono::seconds(2);
 while(queue.statistics().bytes!=64u*1024u*1024u&&std::chrono::steady_clock::now()<limit)std::this_thread::yield();
 auto full=queue.statistics();assert(full.bytes==full.capacity&&full.pending==2&&full.needed_result_evictions==0);
 assert(full.owner_metadata_bytes>bare.owner_metadata_bytes&&full.peak_observed_owner_metadata_bytes>=active.owner_metadata_bytes);
 auto first=queue.take(required[0]);assert(first&&first->id==0);
 auto missing=queue.take(required[4]);assert(missing&&missing->id==4);
 // Retarget replaces pending work; reoffer the required job as the caller does.
 // An already active or ready copy is pruned instead of dispatched twice.
 std::deque<Queue::Job> retarget;retarget.push_back({required[5],{required[5],std::vector<std::uint64_t>(17,5)}});
 queue.schedule(std::move(retarget),compile,4,std::vector<Key>{required[5]},{required[5]},64u*1024u*1024u,true);
 auto last=queue.take(required[5]);assert(last&&last->id==5);
 queue.pause();auto final=queue.statistics();assert(final.needed_result_evictions==0&&final.retired_ready>0);
 for(auto const& count:calls)assert(count<=1);
 queue.clear();auto cleared=queue.statistics();assert(!cleared.bytes&&!cleared.reserved_bytes);
 // Retained key/vector storage stays reported even after logical results retire.
 assert(cleared.owner_metadata_bytes>=bare.owner_metadata_bytes);
}
''')

    def test_pause_releases_active_input_storage_before_reporting_quiescence(self):
        run_cpp(r'''
#include "Renderer/native/render_core/content_preparation.h"
#include <cassert>
using namespace c3x_renderer::render_core;
struct Input {
 std::vector<std::uint64_t> nodes;std::unique_ptr<unsigned> owner;
 Input():nodes(8192,42),owner(std::make_unique<unsigned>(17)){}
 Input(Input&&)=default;Input& operator=(Input&&)=default;
};
struct Result {std::size_t bytes()const{return 1;}};
using Queue=ContentPreparation<unsigned,Input,Result>;
int main(){
 Queue queue;queue.set_owned_bytes({},[](Input const& input){return input.nodes.capacity()*8+sizeof(*input.owner);});
 auto baseline=queue.statistics().owner_metadata_bytes;std::atomic<bool> entered{false};
 std::deque<Queue::Job> jobs;jobs.push_back({1,Input{}});
 queue.configure(std::move(jobs),[&](Input const& input,std::atomic<bool> const& stop,unsigned){
  assert(*input.owner==17&&input.nodes.size()==8192);entered=true;
  while(!stop)std::this_thread::yield();return std::unique_ptr<Result>{};});
 queue.resume();while(!entered)std::this_thread::yield();auto active=queue.statistics();
 assert(active.active==1&&active.owner_metadata_bytes>=baseline+8192*8+4);
 queue.pause();auto paused=queue.statistics();assert(!paused.active&&!paused.reserved_bytes);
 // Cancellation returns the still-owned input to pending; clear retires it.
 assert(paused.pending==1&&paused.owner_metadata_bytes>=baseline+8192*8+4);
 queue.clear();auto cleared=queue.statistics();assert(!cleared.pending&&!cleared.bytes);
 assert(active.owner_metadata_bytes-cleared.owner_metadata_bytes>=8192*8+4);
}
''')


if __name__ == '__main__':
    unittest.main()
