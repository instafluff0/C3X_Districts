"""Canvases Civ III's form hit test never reads keep no input coverage.

Civ III's form hit test (`get_form_under_mouse`, `FUN_00608d50`) reads only a
form's own canvas, and only when the form's Status1 lacks bit 2. Two canvases
receive most native drawing, yet neither can be read:
- the screen canvas (`p_jgl_screen_canvas`, a standalone global) belongs to no
  form;
- `Units_Control` is created with flags 0x1000022 (`FUN_004e2b00`). Status1 is
  written only at creation, and bit 2 skips the canvas read.
On the busy 1498 AD save they took 88 of the coverage worker's 93 seconds
(performance review, October 7, section 5). The model now drops their
coverage. If an exempt canvas ever feeds another image, that image is exempted
too, and a later query there fails closed instead of answering from partial
history.
"""
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


class HitExemptCanvasTests(unittest.TestCase):
    def test_model_drops_exempt_canvases_and_fails_closed_on_their_use(self):
        run_cpp(r'''
#include "Renderer/native/native_hit_scene.h"
#include <cassert>
#include <stdexcept>
using namespace c3x_native_hit;
int main(){
 Scene scene;const int W=256,H=128;Rect all{0,0,W,H},dot{10,10,11,11};unsigned value=0;
 scene.create(1,W,H,Format::rgb555,0x7c1f); // a hit-tested form canvas
 scene.create(2,W,H,Format::rgb555,0x7c1f); // the unit overlay canvas
 scene.create(3,W,H,Format::rgb555,0);      // the screen canvas
 scene.submit({Kind::fill,2,0,dot,all,0,0,0x1111});
 assert(scene.pixel(2,10,10,value)&&value==0x1111);
 scene.exempt(2);scene.exempt(3);scene.exempt(2);
 // Exempt canvases answer nothing and ignore draws, uploads and transfers.
 assert(!scene.pixel(2,10,10,value)&&!scene.pixel(3,0,0,value));
 scene.submit({Kind::fill,2,0,dot,all,0,0,0x2222});
 std::vector<unsigned> words(W*H,5);scene.upload(3,words.data(),words.size());
 scene.submit({Kind::native_image,3,2,all,all,0,0,0x7c1f,0,0,0,W,H});
 scene.submit({Kind::copy,3,1,all,all,0,0});
 assert(!scene.pixel(2,10,10,value)&&!scene.pixel(3,0,0,value));
 // The hit-tested canvas is unaffected.
 scene.submit({Kind::fill,1,0,dot,all,0,0,0x3333});
 assert(scene.pixel(1,10,10,value)&&value==0x3333&&scene.pixel(1,0,0,value)&&value==0x7c1f);
 // An exempt source must never leave its reader answering from partial
 // history: the reader loses coverage and reports the refusal.
 bool refused=false;
 try{scene.submit({Kind::copy,1,2,dot,all,10,10});}catch(std::runtime_error const&){refused=true;}
 assert(refused&&!scene.pixel(1,0,0,value));
 // A new image with the same identity starts with full coverage again.
 scene.create(2,W,H,Format::rgb555,0x7c1f);assert(scene.pixel(2,0,0,value)&&value==0x7c1f);
}
''')

    def test_client_skips_exempt_draws_and_keeps_others_ordered(self):
        run_cpp(r'''
#include "Renderer/native/gpu_image_worker_client.h"
#include <cassert>
using namespace c3x_gpu_images;
long long next_id=10;
int execute(c3x_renderer_gpu_images_v1 const*,c3x_renderer_gpu_result_v1* out,unsigned*,unsigned count){
 out->image=next_id++;out->pixel_count=count;return C3X_RENDERER_RESULT_OK;}
int main(){
 c3x_renderer_gpu_frame_v1 frame={};frame.struct_size=sizeof(frame);frame.ticket=frame.session=1;
 WorkerClient client(execute,frame,true);
 auto form=client.create(128,64,Format::rgb555),units=client.create(128,64,Format::rgb555);assert(form&&units);
 Rect all{0,0,128,64};unsigned value=0;
 Command fill={Kind::fill,units,0,all,all,0,0,0x1234};assert(client.submit(&fill,1));
 assert(client.hit_pixel(units,5,5,value)&&value==0x1234);
 client.hit_exempt(units);client.hit_exempt(units);client.hit_exempt(0);
 for(unsigned k=0;k<5000;++k){Command dot=fill;dot.area={int(k%128),int(k/128%64),int(k%128)+1,int(k/128%64)+1};dot.color=0x2000+k;
  assert(client.submit(&dot,1));
  if(k%2){dot.destination=form;assert(client.submit(&dot,1));}}
 assert(!client.hit_pixel(units,5,5,value));
 assert(client.hit_pixel(form,4999%128,4999/128%64,value)&&value==0x2000+4999);
 assert(client.hit_pixel(form,0,0,value)&&value==0); // even k never reached the form
 // A destroyed exempt identity is forgotten with its image.
 assert(client.destroy(units));
}
''')

    def test_injected_code_declares_only_unreadable_canvases(self):
        source = (ROOT / 'injected_code.c').read_text()
        body = source.split('translate_custom_renderer_native (int operation', 1)[1].split('\n}\n', 1)[0]
        declaration = body.split('C3X_NATIVE_HIT_EXEMPT', 1)[0].rsplit('if (', 1)[1]
        # The overlay is exempt only while its form flags keep the canvas
        # unread, and only at its transfer onto the screen canvas.
        for condition in ('result > 0', 'operation == C3X_NATIVE_IMAGE_DRAW', 'image == p_jgl_screen_canvas->JGL.Image',
                          'source == p_main_screen_form->Units_Control.Data.Canvas.JGL.Image',
                          'p_main_screen_form->Units_Control.Data.Status1 & 2'):
            self.assertIn(condition, declaration)
        self.assertNotIn('enable_custom_rendering_zoom', declaration)
        owner = (ROOT / 'Renderer/native/native_composition_owner.h').read_text()
        handler = owner.split('if(op==C3X_NATIVE_HIT_EXEMPT){', 1)[1].split('\n        }', 1)[0]
        self.assertIn('adapter->owns(', handler)
        self.assertIn('client->hit_exempt(adapter->image(', handler)


if __name__ == '__main__':
    unittest.main()
