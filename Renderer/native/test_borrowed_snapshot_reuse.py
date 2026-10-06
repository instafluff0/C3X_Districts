"""Ambient frames keep the completed view's snapshot work during camera jobs.

After a unit move, a camera job publishes newer topology while ambient frames
still draw the borrowed completed view (an immutable snapshot). Validating that
snapshot's shadow atlas and static raster against the live topology failed on
every frame: all shadow pages were redrawn and the same static region was
repaired, so water and units skipped until the job finished (performance review
4v). Both are reused for exactly the drawn snapshot; live views still validate.
"""
import importlib.util
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


def method(source, header):
    start = source.index(header)
    depth = 0
    for i in range(source.index('{', start), len(source)):
        if source[i] == '{':
            depth += 1
        elif source[i] == '}':
            depth -= 1
            if depth == 0:
                return source[start:i + 1]
    raise AssertionError(header)


class BorrowedShadowReuseTests(unittest.TestCase):
    def test_borrowed_snapshot_reuses_its_atlas_and_live_views_still_validate(self):
        source = (ROOT / 'Renderer/sandbox/fresh_pipeline.h').read_text()
        reusable = method(source, '    bool atlas_reusable(unsigned reuse_failures){')
        run_cpp(r'''
#include <cassert>
#include <cstdio>
struct Harness{
 struct{bool borrowed_scene_frame=false;}renderer;
 bool live_valid=true;unsigned validations=0;
 bool atlas_dependencies(bool append){assert(!append);++validations;return live_valid;}
''' + reusable + r'''
};
int main(){Harness h;
 // A live view reuses only while its dependencies are current.
 assert(h.atlas_reusable(0)&&h.validations==1);
 h.live_valid=false;assert(!h.atlas_reusable(0)&&h.validations==2);
 // The borrowed snapshot keeps its exact atlas although the camera job has
 // applied newer topology, without consulting that topology.
 h.renderer.borrowed_scene_frame=true;assert(h.atlas_reusable(0)&&h.validations==2);
 // A different scene, caster set, light or coverage still rebuilds.
 for(unsigned bit=0;bit<7;++bit)assert(!h.atlas_reusable(1u<<bit));
 h.renderer.borrowed_scene_frame=false;for(unsigned bit=0;bit<7;++bit)assert(!h.atlas_reusable(1u<<bit));
 std::puts("PASS borrowed shadow atlas: snapshot reuse without live validation; live views and changed inputs rebuild");
}
''')

    def test_flag_covers_exactly_the_borrowed_ambient_render(self):
        source = (ROOT / 'Renderer/native/c3x_renderer.cpp').read_text()
        borrow = method(source, '    auto borrow_completed_scene(){')
        self.assertIn('completed_scene_borrowable()?&completed_scene->view:nullptr', borrow)
        prepare = source[source.index('auto completed_view=borrow_completed_scene();'):]
        prepare = prepare[:prepare.index('c3x_renderer64_render_fresh(')]
        # Restored on every exit of the sampler, and set from the same
        # condition that decided whether the completed view was borrowed.
        self.assertIn('struct BorrowedFrame {bool& flag;bool prior;~BorrowedFrame(){flag=prior;}}', prepare)
        self.assertIn('renderer_state.borrowed_scene_frame=completed_scene_borrowable();', prepare)
        # Once the job's own render has replaced the shared shadow and static
        # state, the remaining borrowed frames hold their image instead of
        # redrawing the old snapshot (and the adopted scene redrawing again).
        self.assertIn('if(renderer_state.borrowed_scene_frame && renderer_state.borrowed_scene_stale)return;', prepare)
        self.assertLess(prepare.index('borrowed_scene_frame=completed_scene_borrowable();'),
                        prepare.index('borrowed_scene_stale)return;'))
        job_start = source[source.index('if(gpu_publication.matches_projection(job_frame,job_camera_identity))retain_completed_scene();'):]
        self.assertIn('camera_scene_complete=false;renderer_state.borrowed_scene_stale=false;', job_start[:600])
        pipeline = (ROOT / 'Renderer/sandbox/fresh_pipeline.h').read_text()
        self.assertIn('if (atlas_reusable(reuse_failures))', pipeline)
        render = pipeline[pipeline.index('template<class BodyInputs,class RetireCompletedPlans> bool render('):]
        self.assertLess(render.index('if(!renderer.borrowed_scene_frame)renderer.borrowed_scene_stale=true;'),
                        render.index('refresh_casters(membership)'))
        # The displayed static slot is proven for the snapshot it was drawn
        # from; only live views run the dependency proof that leads to repair.
        static = pipeline[pipeline.index('// Exact content proofs (journal fast path when nothing changed).'):]
        static = static[:static.index('repair_front(front_index,displayed,settings,ring)')]
        self.assertIn('bool proven=renderer.borrowed_scene_frame;', static)
        self.assertIn('if(!proven){', static)
        self.assertLess(static.index('if(!proven){'), static.index('proven=raster_dependencies('))

    def test_capture_checker_flags_borrowed_redraws(self):
        spec = importlib.util.spec_from_file_location(
            'checker', ROOT / 'Renderer/tools/check_borrowed_snapshot.py')
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        line = '[C3X renderer] qpc={} ms=1 process=1 thread=1 sequence=1 stage={}'
        build = 'fresh-shadow-build proofs_ms=16.00 pages={} draws=683 proved=0 reuse_failures={} proof_stage=1'
        old_trace = [line.format(100, 'render-begin'), line.format(110, build.format(25, 128)),
                     line.format(120, build.format(25, 4)), line.format(130, 'camera-complete'),
                     line.format(140, build.format(25, 128))]
        found, builds, jobs, diagnosed = checker.violations(old_trace)
        # Only the dependency-only redraw inside the job: the job's own new
        # scene (membership changed) and live frames after it may rebuild.
        self.assertEqual(([q for q, *_ in found], builds, jobs, diagnosed), ([110], 3, 1, 3))
        static = 'static-compose zoom=1.0000 static_ms=4.08 entry={} repair=0 changed=200 borrowed={}'
        borrowed = [line.format(10, build.format(25, 128) + ' borrowed=1'),
                    line.format(20, build.format(0, 128) + ' borrowed=1'),
                    line.format(30, build.format(25, 128) + ' borrowed=0'),
                    line.format(40, static.format(49, 1)), line.format(50, static.format(41, 1)),
                    line.format(60, static.format(49, 0))]
        self.assertEqual([(q, kind) for q, kind, _ in checker.violations(borrowed)[0]],
                         [(10, 'shadow atlas redraw'), (40, 'static raster repair')])


if __name__ == '__main__':
    unittest.main()
