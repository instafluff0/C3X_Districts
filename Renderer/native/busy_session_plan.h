#pragma once

// Standalone scripted input only. No renderer-owned camera or game simulation.
namespace c3x_renderer {
struct BusySessionView {
    int x=0,y=0,width=128,phase=0;
    char const* name="idle";
    bool same_camera(BusySessionView const& other) const {
        return x==other.x && y==other.y && width==other.width;
    }
};
struct BusySessionPlan {
    static constexpr long long duration_us=60000000,slot_us=33333;
    int home_x,home_y,map_width,map_height;
    BusySessionView at(long long us) const {
        long long ms=us/1000;
        BusySessionView view{home_x,home_y,128,0,"idle"};
        if(ms<10000)return view;
        if(ms<20000){
            int t=int(ms-10000);view.phase=1;view.name="scroll";
            view.x+=2*((t<5000?t:10000-t)/500);return view;
        }
        if(ms<28000){
            int widths[]={160,192,160,128};view.width=widths[(ms-20000)/2000];
            view.phase=2;view.name="zoom";return view;
        }
        if(ms<40000){
            view.x=(map_width/4)|1;view.y=(map_height*3/4)|1;
            view.phase=ms<34000?3:4;view.name=ms<34000?"jump1_idle":"jump1_scroll";
            if(ms>=34000)view.x+=2*int((ms-34000)/1000);return view;
        }
        if(ms<50000){
            view.x=(map_width*3/4)|1;view.y=(map_height*3/4)|1;
            view.phase=ms<44000?5:6;view.name=ms<44000?"jump2_idle":"jump2_scroll";
            if(ms>=44000)view.x-=2*int((ms-44000)/1000);return view;
        }
        view.phase=7;view.name="return_idle";return view;
    }
};
}
