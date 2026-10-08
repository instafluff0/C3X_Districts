"""Execute the real text compiler's refusal/admission paths with bounded GDI doubles.

The Windows native-text fixture supplies actual GDI/GPU parity and can replay
an opt-in owner-thread refusal capture. These host checks do not claim GDI parity.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


NATIVE = Path(__file__).resolve().parent

WINDOWS = r"""
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
using HDC=void*;using HBITMAP=void*;using HGDIOBJ=void*;
using LONG=int;using COLORREF=unsigned;using DWORD=unsigned;
struct POINT {LONG x=0,y=0;};struct RECT {LONG left=0,top=0,right=0,bottom=0;};
struct XFORM {float eM11=1,eM12=0,eM21=0,eM22=1,eDx=0,eDy=0;};
struct LOGFONTA {int lfHeight=0,lfEscapement=0,lfOrientation=0;};
struct SIZE {LONG cx=0,cy=0;};
struct TEXTMETRICA {LONG tmHeight=0,tmOverhang=0,tmAscent=11;};
struct BITMAPINFOHEADER {unsigned biSize=0;LONG biWidth=0,biHeight=0;unsigned short biPlanes=0,biBitCount=0;unsigned biCompression=0;};
struct BITMAPINFO {BITMAPINFOHEADER bmiHeader;};
constexpr int MM_TEXT=1,GM_ADVANCED=2,OBJ_FONT=6,OPAQUE=2,TRANSPARENT=1;
constexpr unsigned CLR_INVALID=~0u,GDI_ERROR=~0u,BI_RGB=0,DIB_RGB_COLORS=0;
constexpr unsigned TA_LEFT=0,TA_TOP=0,TA_NOUPDATECP=0,TA_RIGHT=2,TA_CENTER=6,TA_BOTTOM=8,TA_BASELINE=24;
constexpr int FALSE=0,ERROR=0,NULLREGION=1,SIMPLEREGION=2,COMPLEXREGION=3;
constexpr unsigned TA_UPDATECP=1,TA_RTLREADING=256;
inline int clip_kind=SIMPLEREGION;inline RECT clip_rect={0,0,2240,1260};
inline int GetClipBox(HDC,RECT* p){*p=clip_rect;return clip_kind;}
inline int mapping=MM_TEXT,graphics=1,extra=0,mode=TRANSPARENT,width=24,height=17,overhang=0;
inline unsigned layout=0,align=0,foreground=123,background=456;
inline POINT viewport={},window={};inline XFORM transform={};inline LOGFONTA font={};
inline bool extent_ok=true,metrics_ok=true,dib_ok=true,text_ok=true,varied_curves=false,shaped=false;
inline unsigned dib_creates=0,draws=0,maximum_dib_pixels=0,dib_width=0,dib_height=0;inline std::vector<unsigned> pixels;
inline std::vector<int> draw_counts,draw_y;
inline int GetMapMode(HDC){return mapping;}inline unsigned GetLayout(HDC){return layout;}
inline int GetTextCharacterExtra(HDC){return extra;}
inline int GetViewportOrgEx(HDC,POINT* p){*p=viewport;return 1;}
inline int GetWindowOrgEx(HDC,POINT* p){*p=window;return 1;}
inline int GetGraphicsMode(HDC){return graphics;}
inline int GetWorldTransform(HDC,XFORM* p){*p=transform;return 1;}
inline HGDIOBJ GetCurrentObject(HDC,int){return &font;}
inline int GetObjectA(HGDIOBJ,unsigned n,void* p){std::memcpy(p,&font,n);return int(n);}
inline COLORREF GetTextColor(HDC){return foreground;}inline COLORREF GetBkColor(HDC){return background;}
inline unsigned GetTextAlign(HDC){return align;}inline int GetBkMode(HDC){return mode;}
inline HDC CreateCompatibleDC(HDC){return reinterpret_cast<HDC>(1);}
inline HBITMAP CreateDIBSection(HDC,BITMAPINFO const* info,unsigned,void** p,void*,unsigned){
 ++dib_creates;if(!dib_ok){*p=nullptr;return nullptr;}
 dib_width=unsigned(info->bmiHeader.biWidth);dib_height=unsigned(-info->bmiHeader.biHeight);
 maximum_dib_pixels=std::max(maximum_dib_pixels,dib_width*dib_height);
 pixels.assign(dib_width*dib_height,0);*p=pixels.data();return reinterpret_cast<HBITMAP>(2);
}
inline HGDIOBJ SelectObject(HDC,HGDIOBJ){return nullptr;}
inline int DeleteObject(HGDIOBJ){return 1;}inline int DeleteDC(HDC){return 1;}
inline unsigned SetTextColor(HDC,unsigned){return 0;}inline unsigned SetBkColor(HDC,unsigned){return 0;}
inline int SetBkMode(HDC,int){return 0;}inline unsigned SetTextAlign(HDC,unsigned){return 0;}
inline int GetTextExtentPoint32A(HDC,char const*,int,SIZE* p){p->cx=width;p->cy=height;return extent_ok;}
inline int GetTextMetricsA(HDC,TEXTMETRICA* p){p->tmHeight=height;p->tmOverhang=overhang;return metrics_ok;}
inline int TextOutA(HDC,int x,int y,char const*,int count){
 draw_counts.push_back(count);draw_y.push_back(y);auto sample=draws++;
 if(shaped)for(unsigned row=0;row<dib_height;++row)for(unsigned col=0;col<dib_width;++col)
  if(int(row)>=y&&int(row)<y+height&&int(col)>=x&&int(col)<x+width){
   auto& pixel=pixels[row*dib_width+col];unsigned level=pixel&255;
   unsigned value=(level*128+240*127+127)/255;pixel=value*0x010101;
  }
 if(varied_curves)for(std::size_t n=0;n<pixels.size();++n)pixels[n]=unsigned((n>>(sample%2?8:0))&255);
 return text_ok;
}
inline int GdiFlush(){return 1;}
"""

PROGRAM = r"""
#include "native_text_raster.h"
#include <iostream>
#include <stdexcept>
using namespace c3x_native_text;
void check(bool ok,char const* message){if(!ok)throw std::runtime_error(message);}
int main(){
 auto dc=reinterpret_cast<HDC>(1);State state;Diagnostic d;
 auto reset=[](){mapping=MM_TEXT;graphics=1;extra=0;mode=TRANSPARENT;layout=align=0;foreground=123;background=456;viewport=window={};transform={};font={};};
 check(capture(dc,state,&d),"ordinary DC admission");
 check(!capture(nullptr,state,&d)&&d.reason==Refusal::dc_mapping,"absent owner DC");
 reset();extra=1;d={};check(!capture(dc,state,&d)&&d.reason==Refusal::dc_mapping,"character spacing refused");
 reset();viewport.x=1;d={};check(!capture(dc,state,&d)&&d.reason==Refusal::dc_mapping,"coordinate mapping refused");
 reset();graphics=GM_ADVANCED;transform.eM12=1;d={};check(!capture(dc,state,&d)&&d.reason==Refusal::dc_transform,"nonidentity transformation refused");
 reset();font.lfEscapement=900;d={};check(!capture(dc,state,&d)&&d.reason==Refusal::font,"rotated shaping refused");
 reset();background=CLR_INVALID;d={};check(!capture(dc,state,&d)&&d.reason==Refusal::colors,"invalid background refused");
 reset();check(capture(dc,state,&d),"restored DC admission");
 Raster out;std::string long_text(106,'W');width=1000;height=30;d={};auto before=dib_creates;
 check(!compile(dc,state,long_text.data(),106,out,&d)&&d.reason==Refusal::raster_bounds,"106-byte extent refusal");
 check(d.width==1060&&d.height==90&&dib_creates==before,"refusal preserves exact measured extent and allocates no DIB");
 d={};check(!compile(dc,state,nullptr,106,out,&d)&&d.reason==Refusal::arguments,"missing text has distinct reason");
 width=24;height=17;d={};extent_ok=false;check(!compile(dc,state,"x",1,out,&d)&&d.reason==Refusal::extent,"extent query failure");extent_ok=true;
 d={};dib_ok=false;check(!compile(dc,state,"x",1,out,&d)&&d.reason==Refusal::dib,"DIB allocation failure");dib_ok=true;
 d={};text_ok=false;check(!compile(dc,state,"x",1,out,&d)&&d.reason==Refusal::text_out,"shaping failure");text_ok=true;
 d={};draws=0;check(compile(dc,state,"x",1,out,&d)&&d.reason==Refusal::none,"bounded ordinary response admitted");
 check(out.pixels.size()==58u*51u&&out.curves.size()==17&&draws==17,"all ordered background response samples retained");
 d={};width=515;height=14;draws=0;maximum_dib_pixels=0;draw_counts.clear();draw_y.clear();shaped=true;
 check(compile(dc,state,long_text.data(),106,out,&d)&&d.reason==Refusal::none,"actual 543x42 copied response admitted");
 check(out.width==543&&out.height==42&&out.pixels.size()==22806&&draws==34,"large response uses two full-string sample strips");
 check(maximum_dib_pixels<=response_tile_pixels&&draw_counts.size()==34,"synthetic scratch stays under original bound");
 for(unsigned i=0;i<34;++i)check(draw_counts[i]==106&&draw_y[i]==(i<17?14:-16),"full text and integer origin retained in every strip");
 auto unchanged=apply(out,0,0xff334455,false,true);auto glyph=apply(out,14*out.width+14,0xff334455,false,true);
 check(unchanged==0xff334455&&glyph!=unchanged,"copied shaping preserves native coverage");
 // This second shape crosses a strip boundary; clipped shaping must not lose
 // the second portion or shift either half relative to the native origin.
 width=200;height=60;draws=0;maximum_dib_pixels=0;
 d={};check(!compile(dc,state,long_text.data(),106,out,&d)&&d.reason==Refusal::raster_bounds,"large padded glyph footprint remains refused");
 width=280;height=30;d={};draws=0;
 check(compile(dc,state,long_text.data(),106,out,&d)&&out.width==340&&out.height==90,"supported glyph spans response strips");
 check(apply(out,47*out.width+30,0xff111111,false,true)==apply(out,48*out.width+30,0xff111111,false,true),"no seam across clipped native shaping");
 RECT placed={};check(place(TA_BASELINE,515,11,14,861,592,-14,-14,543,42,placed)&&placed.left==847&&placed.top==567&&placed.right==1390&&placed.bottom==609,"left baseline placement preserves captured basis");
 check(place(TA_CENTER|TA_BASELINE,515,11,14,861,592,-14,-14,543,42,placed)&&placed.left==589&&placed.top==567&&placed.right==1132&&placed.bottom==609,"odd-advance center baseline uses actual GDI ceil rounding");
 check(place(TA_CENTER|TA_BASELINE,516,11,14,861,592,-14,-14,544,42,placed)&&placed.left==589&&placed.right==1133,"even-advance center retains native basis");
 check(!place(TA_CENTER,515,11,14,INT32_MIN,592,-14,-14,543,42,placed),"center arithmetic widens before odd half-width subtraction");
 check(!place(TA_RIGHT,515,11,14,INT32_MIN,592,-14,-14,543,42,placed),"anchor underflow refused before native coordinate arithmetic");
 shaped=false;width=24;height=17;
 d={};draws=0;varied_curves=true;check(!compile(dc,state,"x",1,out,&d)&&d.reason==Refusal::curve_count&&d.curves==1025,"distinct response curve bound preserved");
 varied_curves=false;overhang=INT32_MIN;d={};before=dib_creates;
 check(!compile(dc,state,"x",1,out,&d)&&d.reason==Refusal::raster_bounds&&dib_creates==before,"extreme native overhang refuses without overflow/allocation");overhang=0;
 d={};check(compile(dc,state,nullptr,0,out,&d)&&out.pixels.empty(),"empty native text remains a no-op");
 std::cout<<"PASS real text refusal and bounded response contracts\n";
}
"""


ADAPTER_PROGRAM = r"""
#include "native_text_raster.h"
#include "gpu_image_commands.h"
#include <stdexcept>
#include <iostream>
#include <climits>
using namespace c3x_gpu_images;
namespace c3x_native_access {HDC dc(void* p){return p;}}
struct Backend {
 struct Upload {Id id;std::vector<unsigned> words;};
 Id next=1;unsigned creates=0,destroys=0,uploads=0,submit_count=0,fail_create=0,fail_upload=0;bool fail_submit=false;
 std::vector<Upload> content;std::vector<Command> queued;
 Id create(unsigned,unsigned,Format){++creates;return creates==fail_create?0:next++;}
 void destroy(Id id){if(id)++destroys;}
 bool upload(Id id,unsigned,unsigned const* words,std::size_t count){++uploads;if(uploads==fail_upload)return false;content.push_back({id,{words,words+count}});return true;}
 bool submit(Command const* commands,unsigned count){++submit_count;if(fail_submit)return false;queued.insert(queued.end(),commands,commands+count);return true;}
};
struct Harness {
 struct Image {void* native=reinterpret_cast<void*>(1);Id gpu=999,detail=1000;bool dirty=false;};
 struct Counts {unsigned text_hits=0,text_builds=0,translated=0;} counters;
 Backend gpu;
 // REAL_CACHE
 c3x_native_text::Refusal reason=c3x_native_text::Refusal::none;
 bool text_refused(Image const&,void const*,void const*,unsigned,c3x_native_text::Diagnostic const& d){reason=d.reason;return false;}
 static Rect rect(void const* p){auto r=static_cast<RECT const*>(p);return {r->left,r->top,r->right,r->bottom};}
 // REAL_DRAW
};
void check(bool value,char const* message){if(!value)throw std::runtime_error(message);}
int main(){
 Harness a;Harness::Image destination;RECT anchor={861,592,0,0};std::string message(106,'M');
 width=515;height=14;align=TA_BASELINE;shaped=true;
 check(a.draw_text(destination,message.data(),&anchor,106),"actual large text submitted");
 check(destination.dirty&&a.counters.text_builds==1&&a.gpu.creates==2&&a.gpu.uploads==2&&a.gpu.queued.size()==2,"one immutable raster/curve pair shared by native and detail commands");
 auto first=a.gpu.queued[0];check(first.area.left==847&&first.area.top==567&&first.area.right==1390&&first.area.bottom==609&&first.kind==Kind::native_text&&a.gpu.queued[1].destination==destination.detail&&first.source==a.gpu.queued[1].source,"actual native baseline and ordering preserved");
 unsigned draw_count=draws;clip_rect={900,575,1200,601};
 check(a.draw_text(destination,message.data(),&anchor,106)&&a.counters.text_hits==1&&draws==draw_count&&a.gpu.creates==2,"unchanged text borrows cached immutable resource");
 check(a.gpu.queued[2].clip.left==900&&a.gpu.queued[2].clip.right==1200,"current clipping is applied on cache hit");
 message[0]='Q';check(a.draw_text(destination,message.data(),&anchor,106)&&a.counters.text_builds==2&&a.texts[0].text[0]=='M',"caller-owned text copied before changes");
 foreground=777;check(a.draw_text(destination,message.data(),&anchor,106)&&a.counters.text_builds==3,"foreground participates in native response identity");
 mode=OPAQUE;check(a.draw_text(destination,message.data(),&anchor,106)&&a.counters.text_builds==4,"opaque background participates in native response identity");
 clip_kind=NULLREGION;draw_count=draws;auto queued=a.gpu.queued.size();check(a.draw_text(destination,message.data(),&anchor,106)&&draws==draw_count&&a.gpu.queued.size()==queued,"null native clip remains no-op");
 clip_kind=COMPLEXREGION;check(!a.draw_text(destination,message.data(),&anchor,106)&&a.reason==c3x_native_text::Refusal::clip,"unsupported native clip still refuses");
 clip_kind=SIMPLEREGION;align=TA_UPDATECP;check(!a.draw_text(destination,message.data(),&anchor,106)&&a.reason==c3x_native_text::Refusal::alignment,"mutable native CP still refuses");
 align=TA_BASELINE;mode=TRANSPARENT;foreground=123;clip_rect={0,0,2240,1260};
 Harness b;b.gpu.fail_create=2;destination.dirty=false;
 check(!b.draw_text(destination,message.data(),&anchor,106)&&b.reason==c3x_native_text::Refusal::gpu_admission&&b.gpu.destroys==1&&b.text_bytes==0&&!destination.dirty,"curve allocation failure releases staged glyph and preserves destination");
 Harness c;c.gpu.fail_upload=2;
 check(!c.draw_text(destination,message.data(),&anchor,106)&&c.reason==c3x_native_text::Refusal::gpu_upload&&c.gpu.destroys==2&&c.text_bytes==0&&!destination.dirty,"failed upload releases both staged resources");
 Harness d;d.gpu.fail_submit=true;
 check(!d.draw_text(destination,message.data(),&anchor,106)&&d.reason==c3x_native_text::Refusal::submission&&!destination.dirty,"failed ordered submission still refuses ownership");
 width=1000;height=30;Harness oversized;
 check(!oversized.draw_text(destination,message.data(),&anchor,106)&&oversized.reason==c3x_native_text::Refusal::raster_bounds&&oversized.gpu.creates==0,"oversized shape refuses before GPU allocation");
 std::cout<<"PASS actual text adapter immutable cache, clipping, ordered pair and failure retirement\n";
}
"""


class NativeTextRefusal(unittest.TestCase):
    def test_real_compiler_refusal_and_response_contracts(self):
        compiler = shutil.which("clang++") or shutil.which("c++")
        if not compiler:
            self.skipTest("host C++ compiler unavailable; native fixture remains required")
        with tempfile.TemporaryDirectory(prefix="c3x-text-refusal-") as directory:
            temp = Path(directory)
            (temp / "windows.h").write_text(WINDOWS)
            (temp / "test.cpp").write_text(PROGRAM)
            built = subprocess.run(
                [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-isystem",
                 str(temp), "-I", str(NATIVE), str(temp / "test.cpp"), "-o", str(temp / "test")],
                capture_output=True, text=True,
            )
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            run = subprocess.run([str(temp / "test")], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("PASS real text refusal", run.stdout)

    def test_busy_map_label_set_stays_cached(self):
        # A busy map draws a few hundred distinct strings per pass. A 32-entry
        # cache rebuilt every label raster (17 GDI renders and four GPU
        # operations) on each redraw of the 1498 AD save.
        compiler = shutil.which("clang++") or shutil.which("c++")
        if not compiler:
            self.skipTest("host C++ compiler unavailable; native fixture remains required")
        source = (NATIVE / "native_image_adapter.h").read_text()
        cache = source[source.index("    struct Text {"):source.index("    void text_record(")]
        draw = source[source.index("    bool draw_text("):source.index("    using Get=")]
        harness = ADAPTER_PROGRAM.split("int main(){", 1)[0]
        program = harness.replace("// REAL_CACHE", cache).replace("// REAL_DRAW", draw) + r"""
int main(){
 Harness a;Harness::Image destination;RECT anchor={400,300,0,0};
 width=60;height=12;align=TA_BASELINE;shaped=false;
 for(int pass=0;pass<2;++pass)for(int n=0;n<200;++n){std::string label="City "+std::to_string(n);
  check(a.draw_text(destination,label.data(),&anchor,unsigned(label.size())),"label submitted");}
 check(a.counters.text_builds==200&&a.counters.text_hits==200,"a redrawn busy label set is served from the cache");
 std::cout<<"PASS busy label set cached\n";
}
"""
        with tempfile.TemporaryDirectory(prefix="c3x-text-labels-") as directory:
            temp = Path(directory)
            (temp / "windows.h").write_text(WINDOWS)
            (temp / "test.cpp").write_text(program)
            built = subprocess.run(
                [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-isystem",
                 str(temp), "-I", str(NATIVE), str(temp / "test.cpp"), "-o", str(temp / "test")],
                capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            run = subprocess.run([str(temp / "test")], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("PASS busy label set cached", run.stdout)

    def test_actual_adapter_cache_submission_and_failure_retirement(self):
        compiler = shutil.which("clang++") or shutil.which("c++")
        if not compiler:
            self.skipTest("host C++ compiler unavailable; native fixture remains required")
        source = (NATIVE / "native_image_adapter.h").read_text()
        cache = source[source.index("    struct Text {"):source.index("    void text_record(")]
        draw = source[source.index("    bool draw_text("):source.index("    using Get=")]
        program = ADAPTER_PROGRAM.replace("// REAL_CACHE", cache).replace("// REAL_DRAW", draw)
        with tempfile.TemporaryDirectory(prefix="c3x-text-adapter-") as directory:
            temp = Path(directory)
            (temp / "windows.h").write_text(WINDOWS)
            (temp / "test.cpp").write_text(program)
            built = subprocess.run(
                [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-isystem",
                 str(temp), "-I", str(NATIVE), str(temp / "test.cpp"), "-o", str(temp / "test")],
                capture_output=True, text=True,
            )
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            run = subprocess.run([str(temp / "test")], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("PASS actual text adapter", run.stdout)

    def test_native_thread_and_cpu_ui_delegation_remain(self):
        source = (NATIVE / "native_image_adapter.h").read_text()
        operation = source[source.index("    int operation("):]
        self.assertIn('GetCurrentThreadId()!=thread', operation)
        self.assertIn('destination&&destination->owned&&draw_text', operation)
        draw = source[source.index("    bool draw_text("):source.index("    using Get=")]
        for forbidden in ("readback(", "Map(", "Flush(", "Sleep(", "GetCurrentObject("):
            self.assertNotIn(forbidden, draw)
        self.assertIn("c3x_native_text::place", draw)

    def test_capture_is_bounded_metadata_only(self):
        source = (NATIVE / "native_image_adapter.h").read_text()
        start = source.index("    void text_record(")
        end = source.index("    bool draw_text(", start)
        capture = source[start:end]
        self.assertIn("text_refusal_reports>=4", capture)
        self.assertIn("text_candidate_reports>=4", capture)
        self.assertIn('"C3X_RENDERER_TEXT_REFUSAL_CAPTURE"', capture)
        self.assertIn('"C3X_RENDERER_TEXT_CANDIDATE_CAPTURE"', capture)
        self.assertIn("unsigned wanted=106", capture)
        self.assertIn("std::min(count,1024u)", capture)
        for forbidden in ("gpu.", "readback(", "Map(", "Flush(", "Sleep(", "words("):
            self.assertNotIn(forbidden, capture)


if __name__ == "__main__":
    unittest.main()
