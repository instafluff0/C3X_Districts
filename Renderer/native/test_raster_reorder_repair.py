"""Retained-raster order proofs report which surviving draws moved.

A camera step can change the enumeration order of unchanged contributors.
Strict validation still rejects it; local repair asks for the draws now
visited before a recorded predecessor and redraws only their regions.
"""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class RasterReorderRepairTests(unittest.TestCase):
    def test_forget_drops_stale_draws_constraints_and_unused_proofs(self):
        run_cpp(r'''
#include "Renderer/native/render_core/raster_contributors.h"
#include <cassert>
#include <cstdio>
#include <memory>
#include <vector>
struct Proof {};
using Inputs=c3x_renderer::render_core::RasterContributors<Proof,14>;
int main(){
 Inputs inputs;auto shared=std::make_shared<Proof const>(),own=std::make_shared<Proof const>();
 // Two draws share proof 7; draw 9 owns proof 9.
 auto key=[](std::uint64_t proof,std::uint64_t ordinal){Inputs::Key k{};k[0]=proof;k[1]=ordinal;k[2]=ordinal;return k;};
 inputs.begin_append();
 assert(inputs.add(key(7,1),shared,1,1)&&inputs.add(key(9,2),own,2,1)&&inputs.add(key(7,3),shared,3,1));
 inputs.finish_dependencies();
 assert(inputs.proofs.size()==2&&inputs.order_edges.size()==2);
 // The repaired region held draws 2 and 3; the tile rebuilt draw 2 as proof 11.
 inputs.forget({key(9,2),key(7,3)});
 assert(inputs.draws.size()==1&&inputs.order_edges.empty());
 assert(inputs.proofs.count(7)&&!inputs.proofs.count(9));
 inputs.begin_append();
 assert(inputs.add(key(11,2),own,2,1)&&inputs.add(key(7,3),shared,3,1));inputs.finish_dependencies();
 inputs.begin_membership();
 for(auto k:{key(7,1),key(11,2),key(7,3)})assert(inputs.visit_membership(k));
 assert(inputs.exact_membership()&&inputs.remaining_order_preserved());
 std::printf("PASS raster forget: stale_dropped=1 unused_proof_dropped=1 exact_after_append=1\n");
}
''')

    def test_reordered_survivors_are_reported_without_rejecting(self):
        run_cpp(r'''
#include "Renderer/native/render_core/raster_contributors.h"
#include <cassert>
#include <cstdio>
#include <memory>
#include <vector>
struct Proof {};
using Inputs=c3x_renderer::render_core::RasterContributors<Proof,14>;
int main(){
 Inputs inputs;auto proof=std::make_shared<Proof const>();
 auto key=[](std::uint64_t id){Inputs::Key k{};k[0]=id;k[2]=id;return k;};
 // One strip draws A, B, C, D in that order.
 inputs.begin_append();
 for(std::uint64_t id:{1,2,3,4})assert(inputs.add(key(id),proof,id,1));
 inputs.finish_dependencies();
 auto visit=[&](std::vector<std::uint64_t> order){inputs.begin_membership();for(auto id:order)assert(inputs.visit_membership(key(id)));};
 // Unchanged order.
 visit({1,2,3,4});assert(inputs.exact_membership());
 std::vector<std::uint32_t> reordered;assert(inputs.remaining_order_preserved(&reordered)&&reordered.empty());
 // D moved before C: strict validation rejects; repair learns only D.
 visit({1,2,4,3});assert(!inputs.exact_membership()&&!inputs.remaining_order_preserved());
 reordered.clear();assert(inputs.remaining_order_preserved(&reordered));
 assert(reordered.size()==1&&reordered[0]==inputs.draws.at(key(4)).index);
 // A removed survivor's constraints still pass through it: C before A.
 visit({3,2,4});reordered.clear();assert(inputs.remaining_order_preserved(&reordered));
 assert(!reordered.empty());
 std::printf("PASS raster reorder repair: strict_rejects=1 reports_moved=%zu\n",reordered.size());
}
''')


if __name__ == '__main__':
    unittest.main()
