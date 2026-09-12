"""Ordinary volcano fixtures preserve terrain identity and bounded context."""
from pathlib import Path
import tempfile
import unittest

from Renderer import renderer


class VolcanoFixtureTests(unittest.TestCase):
    def rows(self, case):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'scene.csv'
            renderer.scene('volcanoes', case, target)
            return target.read_text()

    def test_activity_control_changes_no_geometry(self):
        self.assertEqual(self.rows('detail'), self.rows('active'))

    def test_each_case_contains_one_ordinary_volcano_and_ground(self):
        import csv
        import io
        for case in renderer.standard('volcanoes')['recipe']['cases']:
            rows = list(csv.reader(io.StringIO(self.rows(case))))[1:]
            # V3 fields: x, y, base, real, river, bonus, flags.
            self.assertEqual(sum(int(row[3]) == 10 for row in rows), 1, case)
            terrain = {(int(row[0]), int(row[1])): int(row[3]) for row in rows}
            pair = ((13, 13), (12, 12)) if case == "isolated" else ((15, 15), (14, 14))
            self.assertEqual([terrain[xy] for xy in pair], [6, 6], case)
            self.assertGreater(sum(int(row[3]) in (0, 1, 2, 3) for row in rows), 100, case)

    def test_production_ownership_follows_tiles_and_wraps(self):
        import subprocess
        source = (renderer.ROOT / 'Renderer/native/source_fidelity/geometry.h').read_text()
        body = source[source.index('    std::vector<std::array<float,2>> volcano_centers;'):
                      source.index('    record_natural_phase(2);')]
        program = r'''#include <array>
#include <vector>
#include <cassert>
#include <cmath>
#include "Renderer/lab/shared/natural/vertex.h"
using Vertex=c3x_renderer::fidelity::MapVertex;
struct Tile {int real;};
void check(int nc,int nr,int owner_c,int owner_r,bool present,bool wrapped) {
    std::vector<Vertex> natural_vertices[3];
    for(unsigned layer:{0u,2u})for(int y=0;y<=16;y++)for(int x=0;x<=16;x++) {
        Vertex v{};v.world_x=nc+x/16.f;v.world_y=nr+y/16.f;v.world_z=.25;
        v.normal_x=.3;v.normal_z=.8;natural_vertices[layer].push_back(v);
    }
    auto original=natural_vertices[0];unsigned observed=0;
    auto lookup_natural=[&](int c,int r) {
        ++observed;
        auto wrap=[&](int n){return wrapped?(n%32+32)%32:n;};
        return Tile{present && wrap(c)==wrap(owner_c) && r==owner_r?10:6};
    };
''' + body + r'''
    assert(observed==9);
    for(unsigned layer:{0u,2u})for(unsigned i=0;i<original.size();i++) {
        auto a=original[i],b=natural_vertices[layer][i];
        assert(a.world_x==b.world_x && a.world_y==b.world_y && a.world_z==b.world_z);
        assert(a.normal_x==b.normal_x && a.normal_z==b.normal_z);
        assert(b.relief_owner_coverage==float(present));
        if(present) {
            int local_c=owner_c;
            if(wrapped)while(local_c<nc-1)local_c+=32;
            assert(b.relief_owner_u==b.world_x-(local_c+.5f));
            assert(b.relief_owner_v==b.world_y-(owner_r+.5f));
        }
    }
}
int main() {
    for(int c:{-9,0,16,42})for(int r:{-4,0,11})for(int dc=-1;dc<=1;dc++)for(int dr=-1;dr<=1;dr++) {
        check(c,r,c+dc,r+dr,true,false);
        check(c,r,c+dc,r+dr,false,false);
    }
    check(31,2,0,2,true,true);
    check(32,2,0,2,true,true);
}
'''
        with tempfile.TemporaryDirectory() as directory:
            cpp=Path(directory)/'ownership.cpp';binary=Path(directory)/'ownership'
            cpp.write_text(program)
            result=subprocess.run(['c++','-std=c++17','-O2','-Wall','-Wextra','-Werror',
                                   '-I',str(renderer.ROOT),str(cpp),'-o',str(binary)],
                                  capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            subprocess.run([str(binary)],check=True,capture_output=True)

    def test_shared_changes_select_volcanoes_and_terrain_witness(self):
        for category in ('day-night', 'shadows', 'transitions'):
            self.assertIn('volcanoes', renderer.affected(category))
        for witness in ('terrain-edit', 'volcano-lifecycle'):
            self.assertIn(witness, [case[0] for case in renderer.integration_replay_cases('volcanoes')])


if __name__ == '__main__':
    unittest.main()
