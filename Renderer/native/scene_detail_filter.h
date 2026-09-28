#pragma once
// CAS no-scaling kernel, adapted from AMD FidelityFX CAS and 0 A.D.'s cas.fs.
// Copyright (c) 2020 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
namespace c3x_renderer {
inline char const* scene_detail_filter(){return R"(
float4 scene_pixel(int2 p){uint w,h;input_image.GetDimensions(w,h);
 return input_image.Load(int3(clamp(p,int2(0,0),int2(w-1,h-1)),0));}
float4 scene_detail(int2 p,float amount){
 float4 center=scene_pixel(p);if(amount<=0)return center;
 float4 a=scene_pixel(p+int2(-1,-1)),b=scene_pixel(p+int2(0,-1)),c=scene_pixel(p+int2(1,-1));
 float4 d=scene_pixel(p+int2(-1,0)),f=scene_pixel(p+int2(1,0));
 float4 g=scene_pixel(p+int2(-1,1)),h=scene_pixel(p+int2(0,1)),i=scene_pixel(p+int2(1,1));
 // Do not sharpen through a coverage boundary or change alpha. This import
 // precedes native overlays and HUD; their text is never a sharpening input.
 if(min(center.a,min(min(min(a.a,b.a),min(c.a,d.a)),min(min(f.a,g.a),min(h.a,i.a))))<.999)return center;
 float3 lo=min(min(min(d.rgb,center.rgb),min(f.rgb,b.rgb)),h.rgb);
 float3 hi=max(max(max(d.rgb,center.rgb),max(f.rgb,b.rgb)),h.rgb);
 lo+=min(lo,min(min(a.rgb,c.rgb),min(g.rgb,i.rgb)));
 hi+=max(hi,max(max(a.rgb,c.rgb),max(g.rgb,i.rgb)));
 float3 amplitude=sqrt(saturate(min(lo,2-hi)/max(hi,1.e-6)));
 float3 weight=-amplitude/8;
 float3 sharpened=saturate(((b.rgb+d.rgb+f.rgb+h.rgb)*weight+center.rgb)/(1+4*weight));
 return float4(lerp(center.rgb,sharpened,amount),center.a);
}
)";}
}
