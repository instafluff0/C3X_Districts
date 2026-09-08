#include "response_v1.hlsl"
#ifndef Q6_SHADOWS
#define Q6_SHADOWS 1
#endif
#ifndef Q6_CONTACT
#define Q6_CONTACT 1
#endif
#ifndef Q6_LEGACY
#define Q6_LEGACY 0
#endif
#ifndef Q6_EMISSIVE_ONLY
#define Q6_EMISSIVE_ONLY 0
#endif
cbuffer Frame : register(b0) {
    float4 Sun, SunColorExposure, Moon, MoonColorNight, AmbientEmission, Viewport;
};
struct V {float3 position:POSITION; float2 uv:TEXCOORD0;};
struct P {float4 position:SV_POSITION; float2 uv:TEXCOORD0;};
P VSMain(V v) {P p;p.position=float4(v.position,1);p.uv=v.uv;return p;}
P VSFeature(V v) {return VSMain(v);}

// Constructed boxes, not a substitute for normalized city/vegetation source art.
void bounds(int id, out float3 lo, out float3 hi) {
    int x=id%4, y=id/4;
    float h=0.50+0.16*((id*7)%5);
    lo=float3((x-1.5)*1.08-.32,(y-1.5)*1.02-.30,0);
    hi=lo+float3(.64,.60,h);
}
bool hitbox(float3 o,float3 d,float3 lo,float3 hi,out float t,out float3 n) {
    float3 inv=1.0/(d+float3(0.0000001,0.0000001,0.0000001));
    float3 a=(lo-o)*inv,b=(hi-o)*inv;
    float3 nearv=min(a,b),farv=max(a,b);
    t=max(nearv.x,max(nearv.y,nearv.z));
    float end=min(farv.x,min(farv.y,farv.z));
    n=t==nearv.x?float3(-sign(d.x),0,0):(t==nearv.y?float3(0,-sign(d.y),0):float3(0,0,-sign(d.z)));
    return t>0.0001 && end>=t;
}
bool cutout(float3 o,float3 d,out float t) {
    t=(-2.42-o.y)/(d.y+0.0000001);
    float3 p=o+d*t;
    return t>0.0001 && abs(p.x)<1.9 && p.z>.05 && p.z<.92 &&
        frac((p.x+2)*5)>.25 && frac(p.z*6)>.27;
}
float visibility(float3 p,float3 dir) {
    float t;float3 n,lo,hi;
    for(int i=0;i<16;i++) {
        bounds(i,lo,hi);
        if(hitbox(p,dir,lo,hi,t,n)) return 0;
    }
    if(cutout(p,dir,t)) return 0;
    return 1;
}
float3 radiance(float3 p,float3 n,float3 albedo,float3 emission,bool water) {
    float3 dir=normalize(Sun.xyz*Sun.w+Moon.xyz*Moon.w);
    // Fixed stylized projection slope shared by ALL casters/receivers.
    float3 ray=normalize(float3(normalize(dir.xy),1.35));
    float shadow=Q6_SHADOWS?visibility(p+n*.003,ray):1;
    // Bounded local geometric occlusion of ambient only, never a screen blob.
    float contact=1;
    if(Q6_CONTACT && !water) {
        float hits=0;
        for(int i=0;i<4;i++) {
            float angle=(i+.5)*1.5707963;
            float3 r=normalize(n+float3(cos(angle),sin(angle),.3));
            for(int j=0;j<16;j++) {
                float3 lo,hi,n2;float t;bounds(j,lo,hi);
                if(hitbox(p+n*.004,r,lo,hi,t,n2)&&t<.13) hits+=(1-t/.13)*.25;
            }
        }
        contact=1-min(.18,hits*.18);
    }
    float3 ambient=AmbientEmission.rgb*.46*contact;
    float3 direct=SunColorExposure.rgb*Sun.w*(.16+.84*saturate(dot(n,Sun.xyz)))+
        MoonColorNight.rgb*Moon.w*(.20+.80*saturate(dot(n,Moon.xyz)))*.82;
    // Shadow gates direct only; bounded contact never multiplies direct/emissive.
    float3 lit=albedo*(ambient+direct*shadow);
    return (Q6_EMISSIVE_ONLY?0:lit)+emission*MoonColorNight.w*AmbientEmission.w;
}
float4 PSMain(P input):SV_TARGET {
    float2 uv=input.uv*2-1;
    float3 right=normalize(float3(1,-1,0));
    float3 d=normalize(float3(-1,-1,-0.81649658));
    // Correct orthogonal basis with the exact 2:1 screen diamond.
    float3 up=normalize(cross(d,right));
    float3 origin=-d*10 + right*uv.x*4.1 + up*(uv.y*2.73+.15);
    float t=-origin.z/d.z;float3 n=float3(0,0,1);int object=-1;
    for(int i=0;i<16;i++) {
        float3 lo,hi,hn;float ht;bounds(i,lo,hi);
        if(hitbox(origin,d,lo,hi,ht,hn)&&ht<t) {t=ht;n=hn;object=i;}
    }
    float ct;
    if(cutout(origin,d,ct)&&ct<t) {t=ct;n=float3(0,1,0);object=16;}
    float3 pos=origin+d*t;
    float3 albedo=float3(.22,.29,.16),emission=0;
    if(object>=0) {
        albedo=object==16?float3(.12,.24,.10):float3(.42,.36,.27);
        if(n.z>.5) albedo=float3(.24,.15,.10);
        // Authored synthetic window mask; no source binding claim or local lights.
        float axis=abs(n.x)>.5?pos.y:pos.x;
        float mask=step(.62,frac((axis+3)*9))*step(.54,frac(pos.z*6));
        emission=(object<16 && abs(n.z)<.5)?float3(5,2.2,.48)*mask:0;
    }
    bool water=pos.x>2.05 && object<0;
    float3 lit=radiance(pos,n,albedo,emission,water);
    if(water) {
        // Flat transparent water diagnostic: bed visible, no ambient contact leak.
        float3 surface=radiance(pos,float3(0,0,1),float3(.035,.14,.23),0,true);
        lit=q6_over(float4(surface*.38,.38),float4(lit,1)).rgb;
    }
    float3 color=Q6_LEGACY?pow(saturate(lit*SunColorExposure.w/(1+lit*SunColorExposure.w*.30)),1.0/2.2)
        :q6_srgb_encode(q6_display_linear(lit,SunColorExposure.w));
    return float4(color,1);
}
float4 PSFeature(P p):SV_TARGET {return PSMain(p);}
