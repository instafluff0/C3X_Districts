// Standalone study ONLY. The runner binds packed source crest/delay to t21
// and auxiliary foam to t23, which the current natural-water branch does not use.
// Source distances are calibrated at 64 authored units per world unit here;
// these are not claimed to be pixels or the recovered source engine equations.
float4 study_surf(PixelInput input, float4 water) {
    float d=max(0,-input.hydrology_data.x);
    if(d>.95)return water;
    float2 world=q3_source_world(input);
    // This fixture's coast runs along raw map Y. Production needs connected
    // contour arc length; this intentionally bounded experiment does not own it.
    float along=(world.x-world.y)*.70710678;
    float cell=floor(along/2.3);
    float coverage=0;
    [unroll] for(int neighbor=-1;neighbor<=1;neighbor++) {
        float id=cell+neighbor;
        float seed=frac(sin(id*127.1+17.3)*43758.5453);
        float seed2=frac(sin(id*311.7+5.9)*22578.1459);
        float scale=lerp(.30,1.,seed);
        float duration=lerp(15.,30.,seed2),delay=4.*seed;
        float age=fmod(STUDY_TIME+seed*37.,duration+delay)/duration;
        float life=smoothstep(0,.1,age)*(1-smoothstep(.85,1.,age));
        float v=(along-(id+.5)*2.3)/(2.*scale)+.5;
        float center=lerp(40.,2.,saturate(age))/64.;
        float u=(d-center)/(20./64.*scale)+.2;
        float inside=step(0,u)*step(u,1)*step(0,v)*step(v,1);
        float page=floor(seed2*16.);
        float2 origin=float2(fmod(page,8.),floor(page/8.));
        float2 uv=(origin+clamp(float2(u,v),float2(.5/128.,.5/512.),float2(1.-.5/128.,1.-.5/512.)))/float2(8,2);
        float2 crest=water_large_lean1_texture.SampleLevel(material_sampler,uv,0).rg;
        float local_delay=crest.g;
        float valid=1-step(.99,local_delay);
        float aux=water_small_lean1_texture.Sample(material_sampler,
            float2(u,2*v+STUDY_TIME*.5)).r;
        float trail=smoothstep(0,.06,u-local_delay)*(1-smoothstep(.08,.48,u-local_delay));
        float breaking=1-smoothstep(8./64.,20./64.,center);
        float foam=(crest.r*1.75+aux*trail*breaking*.24)*inside*valid*life;
        coverage=1-(1-coverage)*(1-saturate(foam));
    }
    coverage*=smoothstep(.002,.022,d);
    float3 lit=float3(.74,.80,.83)*q6_receiver_illumination(input,float3(0,0,1),1,1);
    float a=water.a+(1-water.a)*coverage;
    return float4((water.rgb*water.a*(1-coverage)+lit*coverage)/max(a,.0001),a);
}
