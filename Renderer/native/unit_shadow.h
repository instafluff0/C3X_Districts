#ifndef C3X_UNIT_SHADOW_H
#define C3X_UNIT_SHADOW_H
#include <array>
#include <vector>
#include <algorithm>
#include <cmath>

namespace c3x_renderer {
// Small pose-local directional height map. Translation is deliberately absent.
// One map supports self-shadow tests and a ground-plane cast footprint.
struct UnitShadow {
    static constexpr int extent=128;
    std::array<float,extent*extent> heights{};
    float left=0,top=0,width=1,height=1,dx=0,dy=0;
    using Point=std::array<float,3>;
    Point project(Point p) const {return {p[0]-dx*p[2],p[1]-dy*p[2],p[2]};}
    bool fit(std::vector<Point> const& points,float light_x,float light_y) {
        for(auto const& p:points)for(float value:p)
            if(!std::isfinite(value) || std::abs(value)>1024.f)return false;
        if(!std::isfinite(light_x) || !std::isfinite(light_y))return false;
        float length=std::hypot(light_x,light_y);
        dx=length>1e-5f?light_x/length*(150.f/96.f):0;
        dy=length>1e-5f?light_y/length*(150.f/96.f):0;
        float right=-1e6f,bottom=-1e6f;left=top=1e6f;
        for(auto p:points)if(p[2]>=0) {
            p=project(p);left=std::min(left,p[0]);top=std::min(top,p[1]);
            right=std::max(right,p[0]);bottom=std::max(bottom,p[1]);
        }
        if(right<left){left=top=0;right=bottom=1;}
        left-=.025f;top-=.025f;right+=.025f;bottom+=.025f;
        width=std::max(.05f,right-left);height=std::max(.05f,bottom-top);
        heights.fill(-1.f);return true;
    }
    void triangle(Point a,Point b,Point c) {
        a=project(a);b=project(b);c=project(c);
        for(auto p:{&a,&b,&c}) {(*p)[0]=((*p)[0]-left)/width*extent;(*p)[1]=((*p)[1]-top)/height*extent;}
        float area=(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
        if(std::abs(area)<1e-7f)return;
        int x0=std::max(0,int(std::floor(std::min({a[0],b[0],c[0]}))));
        int y0=std::max(0,int(std::floor(std::min({a[1],b[1],c[1]}))));
        int x1=std::min(extent-1,int(std::ceil(std::max({a[0],b[0],c[0]}))));
        int y1=std::min(extent-1,int(std::ceil(std::max({a[1],b[1],c[1]}))));
        for(int y=y0;y<=y1;++y)for(int x=x0;x<=x1;++x) {
            float px=x+.5f,py=y+.5f;
            float u=((b[0]-px)*(c[1]-py)-(b[1]-py)*(c[0]-px))/area;
            float v=((c[0]-px)*(a[1]-py)-(c[1]-py)*(a[0]-px))/area;
            float w=1-u-v,z=u*a[2]+v*b[2]+w*c[2];
            if(u>=0 && v>=0 && w>=0 && z>=.002f)heights[y*extent+x]=std::max(heights[y*extent+x],z);
        }
    }
    float coverage(float x,float y,float z=0) const {
        int px=int(std::floor((x-dx*z-left)/width*extent));
        int py=int(std::floor((y-dy*z-top)/height*extent));
        float count=0;
        for(int oy=-1;oy<=1;++oy)for(int ox=-1;ox<=1;++ox) {
            int sx=px+ox,sy=py+oy;
            if(sx>=0 && sy>=0 && sx<extent && sy<extent && heights[sy*extent+sx]>z+.006f)count+=1.f/9;
        }
        return count;
    }
};
}
#endif
