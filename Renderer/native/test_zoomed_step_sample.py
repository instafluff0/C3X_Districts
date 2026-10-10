"""A step's completed zoomed draw stays displayable while the next camera job runs.

Scrolling zoomed in, each adopted step is also prepared at the settled zoom, and
its projected sampler returns that completed draw even while the next camera
job is active. Holding instead showed each step's canonical (1x) image
magnified: 60-75% of scrolled frames at 2x/3x (performance review, section 49).
"""
import unittest
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


class ZoomedStepSampleTests(unittest.TestCase):
    def test_completed_zoomed_draw_is_sampled_during_camera_jobs(self):
        source = (ROOT / 'Renderer/native/c3x_renderer.cpp').read_text()
        start = source.index('            auto draw=[this,weak,capture,selected,x,y,w,h,sharpness]')
        body = source[start:source.index('            };', start) + len('            };')]
        run_cpp(r'''
#include <cassert>
#include <memory>
struct Rect {int left,top,right,bottom;};
struct Texture {void* Get()const{return (void*)1;}};
struct Sampled {enum class Kind {unchanged,immutable,bgra,frozen,held} kind=Kind::unchanged;
 static Sampled frozen(){Sampled s;s.kind=Kind::frozen;return s;}static Sampled held(){Sampled s;s.kind=Kind::held;return s;}
 static Sampled bgra(void*,Rect,float,unsigned long long){Sampled s;s.kind=Kind::bgra;return s;}};
struct Job {bool ready=false;float zoom=1.f;long long pending_since=0;unsigned device_generation=1;long long serial=5;
 Texture front;unsigned long long source_generation=0;};
struct Valid {bool valid()const{return true;}};
struct Owner {
 struct {unsigned device_generation=1;long long gpu_serial=5;} renderer_state;
 bool camera_active=false,camera_scene_complete=true,usable=false;
 bool completed_scene_usable()const{return usable;}
 Sampled sample(std::shared_ptr<Job> const& job_ptr,float requested){
  std::weak_ptr<Job> weak=job_ptr;auto capture=std::make_shared<Valid>(),selected=std::make_shared<Valid>();
  int x=0,y=0,w=8,h=8;float sharpness=.35f;
''' + body + r'''
  return draw(0,1,requested);
 }
};
int main(){
 Owner o;auto job=std::make_shared<Job>();using K=Sampled::Kind;
 // The next camera job is active and the old scene view retired.
 o.camera_active=true;o.camera_scene_complete=false;
 // Not yet drawn at this zoom: hold (keep completed pixels).
 assert(o.sample(job,2.f).kind==K::held);
 // Drawn at the settled zoom right after adoption: display it.
 job->ready=true;job->zoom=2.f;assert(o.sample(job,2.f).kind==K::bgra);
 // A different zoom still holds, and the canonical 1x sample is unaffected.
 assert(o.sample(job,3.f).kind==K::held);assert(o.sample(job,1.f).kind==K::held);
 // A newer publication retires this sampler.
 o.renderer_state.gpu_serial=6;assert(o.sample(job,2.f).kind==K::frozen);
}
''')


    def test_blocked_preparation_keeps_the_completed_draw(self):
        # The compositor calls prepare every display frame. While a camera job
        # blocks it, clearing readiness first discarded the step's zoomed draw.
        source = (ROOT / 'Renderer/native/c3x_renderer.cpp').read_text()
        start = source.index('            auto prepare=[this,weak,capture,selected,origin,settings,geometry,x,y]')
        body = source[start:source.index('            auto draw=[this,weak,capture,selected', start)]
        gate = body.index('renderer_state.gpu_serial!=job->serial)return;')
        self.assertGreater(body.index('job->ready=false;'), gate)


    def test_adoption_prepares_every_step_at_the_settled_zoom(self):
        # After the canonical publish (so its import is untouched), for deferred
        # steps too: they are drawn canonically at adoption, and zoomed in they
        # were displayed magnified until the camera stopped.
        source = (ROOT / 'Renderer/native/c3x_renderer.cpp').read_text()
        start = source.index('}else if(command==Command::gpu_render){')
        block = source[start:source.index('stage=gpu-map-publication', start) if 'stage=gpu-map-publication' in source[start:start+20000] else start+20000]
        self.assertIn('auto zoomed_prepare=map_sample.prepare;', block)
        self.assertGreater(block.index('zoomed_prepare(visual_ticks,visual_frequency,presented);'),
                           block.index('if(session.publish(initial,'))
        # Only a moved camera: same-view republications keep their completed
        # scene, and a draw per republication slowed the light save.
        self.assertIn('if(zoomed_prepare&&moved&&', block)


if __name__ == '__main__':
    unittest.main()
