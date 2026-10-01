"""Pure retained-image zoom policy; no D3D, presenter or native integration claim."""
import unittest
from Renderer.native.native_cpp_test import ROOT, run_cpp


class ZoomPreviewStateTests(unittest.TestCase):
    def test_overwritten_slot_is_unreadable_until_completed_without_changing_presented_state(self):
        run_cpp(r'''
#include "Renderer/sandbox/zoom_preview_state.h"
#include <cassert>
using namespace c3x_renderer::sandbox;
int main(){
 ZoomPreviewIdentity i{};i.target_width=2240;i.target_height=1260;
 ZoomPreviewSource wide{};wide.id=1;wide.identity=i;
 auto close=wide;close.id=2;close.zoom=close.coverage_min_zoom=1.25;
 close.source_time_ms=close.completed_time_ms=20;
 ZoomPreviewState p;assert(p.seed(wide)&&p.seed(close)&&p.request(1.25,i,30));
 assert(p.present_success(p.preview(30))&&p.displayed_zoom()==1.25);
 // New input selects the coherent wide source. Pending work overwrites only
 // the other slot, and discarding its descriptor is not a presentation.
 assert(p.request(1.1,i,40));assert(p.preview(100).source.id==1);
 ZoomPreviewRefinement job;assert(p.begin(100,job)&&p.refining());
 auto token=p.in_flight().id;
 assert(p.discard_source(2)&&p.source_count()==1&&p.refining()&&p.in_flight().id==token);
 assert(p.displayed_zoom()==1.25&&p.presented_identity()==i);
 assert(!p.discard_source(2)&&!p.discard_source(0)&&!p.discard_source(999));
 auto preview=p.preview(101);assert(preview.valid&&preview.source.id==1);
 p.present_failed();assert(p.displayed_zoom()==1.25);
 assert(p.present_success(preview)&&p.displayed_zoom()==1.1);
 assert(p.request(1.2,i,110));ZoomPreviewRefinement extra;
 assert(!p.begin(170,extra)&&p.preview(170).source.id==1);
 // Until explicit completion, preview never exposes the pending output.
 auto ready=wide;ready.id=3;ready.zoom=ready.coverage_min_zoom=1.1;
 ready.source_time_ms=100;ready.completed_time_ms=180;
 assert(p.complete(job,ready)&&p.source_count()==2&&!p.refining());
 assert(p.preview(180).source.id==3&&!p.full_quality_current());
 assert(p.begin(180,job)&&job.zoom==1.2);
 assert(p.discard_source(3)&&p.source_count()==1&&p.preview(180).source.id==1);
 assert(p.fail(job)&&!p.refining());assert(p.begin(181,job));
 ready.id=4;ready.zoom=ready.coverage_min_zoom=1.2;
 ready.source_time_ms=181;ready.completed_time_ms=260;
 assert(p.complete(job,ready)&&p.source_count()==2&&p.full_quality_current());
 // Removing wide coverage must fail safely on zoom-out; an independently
 // completed wide source restores it without resetting displayed/picking.
 assert(p.discard_source(1)&&p.source_count()==1);
 assert(p.request(1.,i,270)&&!p.preview(270).valid&&p.displayed_zoom()==1.1);
 assert(p.begin(330,job)&&job.zoom==1.);
 wide.id=5;wide.source_time_ms=330;wide.completed_time_ms=420;
 assert(p.complete(job,wide)&&p.source_count()==2&&p.preview(420).source.id==5);
 assert(p.full_quality_current()&&p.present_success(p.preview(420))&&p.displayed_zoom()==1.);
}
''')

    def test_quality_matches_actual_latest_input_at_present_completion(self):
        witness = (ROOT / 'Renderer/sandbox/zoom_preview_witness.h').read_text()
        start = witness.index('namespace c3x_renderer { namespace sandbox {')
        end = witness.index('// Private standalone scheduling witness', start)
        helper = witness[start:end]
        run_cpp(r'''
#include "Renderer/sandbox/zoom_preview_state.h"
#include <cassert>
#include <limits>
''' + helper + r'''
using c3x_renderer::sandbox::zoom_preview_present_is_current;
int main(){
 assert(zoom_preview_present_is_current(0,0,1.,1.,1.,0.,0.));
 assert(zoom_preview_present_is_current(100,100,1.25,1.25,1.25,220.,160.));
 // Input arriving during Present invalidates the old full render, including
 // a control render and an input whose endpoint has the same numeric zoom.
 assert(!zoom_preview_present_is_current(100,101,1.25,1.25,1.25,220.,240.));
 assert(!zoom_preview_present_is_current(100,101,1.25,1.25,1.,220.,240.));
 // Relabeling the source's input cannot make pre-input pixels current.
 assert(!zoom_preview_present_is_current(101,101,1.25,1.25,1.25,220.,240.));
 assert(!zoom_preview_present_is_current(101,101,1.25,1.25,1.25,240.,240.2));
 assert(zoom_preview_present_is_current(101,101,1.25,1.25,1.25,260.,240.2));
 // An affine preview or an outdated displayed scale is not full quality.
 assert(!zoom_preview_present_is_current(100,100,1.,1.25,1.25,220.,160.));
 assert(!zoom_preview_present_is_current(100,100,1.25,1.24,1.25,220.,160.));
 double target=double(float(1.2));
 assert(zoom_preview_present_is_current(100,100,target,target,1.2,220.,160.));
 double nan=std::numeric_limits<double>::quiet_NaN();
 assert(!zoom_preview_present_is_current(100,100,1.25,1.25,nan,220.,160.));
 assert(!zoom_preview_present_is_current(100,100,1.25,1.25,1.25,nan,160.));
 assert(!zoom_preview_present_is_current(100,100,1.25,1.25,1.25,220.,nan));
}
''')

    def test_relative_coverage_identity_and_successful_present(self):
        run_cpp(r'''
#include "Renderer/sandbox/zoom_preview_state.h"
#include <cassert>
#include <initializer_list>
#include <limits>
using namespace c3x_renderer::sandbox;
ZoomPreviewIdentity identity(){ZoomPreviewIdentity i{};i.map_epoch=1;i.scene_epoch=2;i.viewer_epoch=3;i.visibility_epoch=4;i.target_width=2240;i.target_height=1260;return i;}
ZoomPreviewSource source(unsigned id,double zoom,double coverage,unsigned time=0){ZoomPreviewSource s{};s.id=id;s.identity=identity();s.zoom=zoom;s.coverage_min_zoom=coverage;s.source_time_ms=s.completed_time_ms=time;return s;}
int main(){
 ZoomPreviewState p;auto i=identity();assert(p.request(1.,i,0));
 assert(p.seed(source(1,1.,1.)));assert(p.seed(source(2,1.25,1.,10)));
 auto f=p.preview(20);assert(f.valid&&f.source.id==2&&f.relative_zoom==.8&&f.absolute_zoom==1.&&f.source_age_ms==10);
 assert(p.present_success(f)&&p.displayed_zoom()==1.);
 assert(p.request(1.25,i,21));auto next=p.preview(22);assert(next.relative_zoom==1.);
 p.present_failed();assert(p.displayed_zoom()==1.);assert(p.present_success(next)&&p.displayed_zoom()==1.25);
 // The same immutable completed source is selected through every reversal.
 for(int n=0;n<40;++n){assert(p.request(n%2?1.:1.25,i,23+n));auto frame=p.preview(24+n);assert(frame.source.id==2&&p.source_count()==2);}
 ZoomPreviewState close;assert(close.request(1.,i,0));assert(close.seed(source(3,1.25,1.25)));
 assert(!close.preview(1).valid);assert(close.seed(source(4,1.,1.)));
 assert(close.preview(1).valid&&close.preview(1).source.id==4);
 auto bad=i;bad.visibility_epoch++;assert(close.request(1.,bad,2)&&!close.preview(2).valid);
 assert(!close.present_success(f));assert(close.displayed_zoom()==1.&&!close.has_presented());
 for(int field=0;field<10;++field){auto changed=i;
  switch(field){case 0:++changed.map_epoch;break;case 1:++changed.scene_epoch;break;case 2:++changed.viewer_epoch;break;case 3:++changed.visibility_epoch;break;case 4:++changed.camera_x;break;case 5:++changed.camera_y;break;case 6:++changed.native_width;break;case 7:++changed.target_width;break;case 8:++changed.target_height;break;default:changed.camera_x=-1000;}
  assert(p.request(1.,changed,100)&&!p.preview(100).valid);
 }
 auto serial=p.request_serial();for(double invalid:{.99,3.01,std::numeric_limits<double>::quiet_NaN()})assert(!p.request(invalid,i,200));assert(p.request_serial()==serial);
 assert(!p.seed(source(8,0.,1.)));assert(!p.seed(source(8,1.,1.25)));
}
''')

    def test_debounce_coalescing_refinement_and_handoff(self):
        run_cpp(r'''
#include "Renderer/sandbox/zoom_preview_state.h"
#include <cassert>
#include <cmath>
using namespace c3x_renderer::sandbox;
int main(){
 ZoomPreviewIdentity i{};i.target_width=2240;i.target_height=1260;
 ZoomPreviewSource base{};base.id=1;base.identity=i;
 ZoomPreviewState p;assert(p.seed(base)&&p.request(1.25,i,10));
 assert(!p.desired(69).valid);auto intent=p.desired(70);assert(intent.valid&&intent.zoom==1.25&&!intent.refresh_wide);
 ZoomPreviewRefinement first;assert(p.begin(70,first)&&p.refining());
 assert(p.request(3.,i,71));assert(p.request(1.5,i,75));assert(p.request_serial()==3);
 assert(!p.desired(134).valid);assert(p.desired(135).zoom==1.5);
 ZoomPreviewRefinement second;assert(!p.begin(135,second));
 auto done=base;done.id=2;done.zoom=first.zoom;done.source_time_ms=first.source_time_ms;done.completed_time_ms=180;
 assert(p.complete(first,done)&&!p.refining()&&!p.full_quality_current());
 assert(!p.complete(first,done));assert(p.begin(180,second)&&second.zoom==1.5&&second.request_serial==3);
 // New input during expensive work changes display immediately, without a
 // second submitted job. The old render can remain a coherent preview source.
 assert(p.request(1.,i,181));auto f=p.preview(182);assert(f.valid&&f.absolute_zoom==1.&&f.relative_zoom==.8);
 auto old=p.displayed_zoom();p.present_failed();assert(p.displayed_zoom()==old);
 assert(p.present_success(f)&&p.displayed_zoom()==1.);
 assert(p.fail(second)&&!p.refining());assert(!p.fail(second));
 assert(!p.desired(240).valid);assert(p.desired(241).valid&&p.desired(241).zoom==1.);
 assert(!p.full_quality_current()); // old wide pixels precede the final input
 assert(p.begin(241,second));auto returned=base;returned.id=5;returned.source_time_ms=241;returned.completed_time_ms=360;
 assert(p.complete(second,returned)&&p.full_quality_current());
 assert(!p.desired(360).valid&&!p.desired(490).valid);
 assert(p.desired(491).valid&&p.desired(491).zoom==1.&&!p.desired(491).refresh_wide);
 // The affine coordinates are continuous when full quality replaces preview.
 ZoomPreviewState handoff;assert(handoff.seed(base)&&handoff.request(1.25,i,250));
 auto before=handoff.preview(310);assert(before.relative_zoom==1.25&&handoff.begin(310,second));
 auto final=base;final.id=3;final.zoom=1.25;final.coverage_min_zoom=1.25;final.source_time_ms=310;final.completed_time_ms=430;
 assert(handoff.complete(second,final)&&handoff.full_quality_current());
 assert(!handoff.desired(430).valid&&!handoff.desired(559).valid);
 assert(handoff.desired(560).valid&&handoff.desired(560).zoom==1.25&&!handoff.desired(560).refresh_wide);
 assert(handoff.desired(10000).valid); // settled animation continues receiving bounded refreshes
 auto after=handoff.preview(430);assert(after.source.id==3&&after.relative_zoom==1.);
 double center=1120.,canonical=423.;
 double old_pixel=center+(canonical-center)*before.source.zoom;
 double new_pixel=center+(canonical-center)*after.source.zoom;
 assert(std::abs((center+(old_pixel-center)*before.relative_zoom)-(center+(new_pixel-center)*after.relative_zoom))<1.e-10);
 assert(handoff.present_success(after)&&handoff.displayed_zoom()==1.25);
 // A scene change while refining rejects the completed output entirely.
 assert(p.request(2.,i,500)&&p.begin(560,second));auto changed=i;++changed.viewer_epoch;
 assert(p.request(2.,changed,561));auto obsolete=final;obsolete.id=4;obsolete.zoom=2.;obsolete.source_time_ms=560;obsolete.completed_time_ms=700;
 assert(!p.complete(second,obsolete)&&!p.refining()&&!p.preview(700).valid);
}
''')

    def test_continuous_input_refresh_submission_is_bounded(self):
        run_cpp(r'''
#include "Renderer/sandbox/zoom_preview_state.h"
#include <cassert>
using namespace c3x_renderer::sandbox;
int main(){
 ZoomPreviewIdentity i{};i.target_width=2240;i.target_height=1260;
 ZoomPreviewSource base{};base.id=1;base.identity=i;
 ZoomPreviewState p;assert(p.seed(base));
 for(unsigned now=0;now<=240;now+=20){assert(p.request(1.+now*.001,i,now));assert(!p.desired(now).valid);}
 assert(!p.desired(249).valid);auto wanted=p.desired(250);assert(wanted.valid&&wanted.refresh_wide&&wanted.zoom==1.);
 ZoomPreviewRefinement refresh;assert(p.begin(250,refresh));
 assert(p.request(1.1,i,260));ZoomPreviewRefinement duplicate;assert(!p.begin(260,duplicate));
 // Actual age can exceed the scheduling threshold while indivisible GPU work
 // runs; the descriptor reports that age, rather than claiming a hard bound.
 assert(p.preview(390).source_age_ms==390);
 auto fresh=base;fresh.id=2;fresh.source_time_ms=250;fresh.completed_time_ms=390;
 assert(p.complete(refresh,fresh));auto frame=p.preview(390);
 assert(frame.source.id==2&&frame.source_age_ms==140&&frame.absolute_zoom==1.1);
 for(unsigned now=400;now<=480;now+=20)assert(p.request(1.+now*.001,i,now));
 assert(!p.desired(499).valid);assert(p.desired(500).valid&&p.desired(500).refresh_wide);
 assert(p.begin(500,refresh)&&p.source_count()==1);
 // A failed render releases the single admission slot for the latest intent.
 assert(p.fail(refresh));assert(p.begin(500,duplicate)&&duplicate.id!=refresh.id);
}
''')


if __name__ == '__main__':
    unittest.main()
