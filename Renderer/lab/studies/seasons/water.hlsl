cbuffer Frame:register(b0){float4 Sun;float4 SunColorExposure;float4 Ambient;float4 View;float4 Detail;};
Texture2D Sand:register(t0);
Texture2D LargeSlope:register(t70);Texture2D SmallSlope:register(t71);
Texture2D CrossSlope:register(t72);Texture2D RiverSlope:register(t73);
SamplerState Wrap:register(s0);SamplerState Clamp:register(s1);
#include "seasonal_policy.hlsl"
#include "lab_scene.hlsl"
struct V {float3 position:POSITION;float4 world:TEXCOORD0;float3 normal:NORMAL;float2 uv:TEXCOORD1;};
struct P {float4 position:SV_POSITION;float3 world:TEXCOORD0;};
struct Output{float4 color:SV_Target0;float validity:SV_Target1;};
P VSMain(V i){P o;o.position=float4(i.position,1);o.world=i.world.xyz;return o;}
P VSFeature(V i){return VSMain(i);}
float3 target_water(float3 world,float shore,float river,bool channel){
    float2 p=world.xy,uv=season_surface_uv(p,.80),fine=season_surface_uv(p,3.7);
    float2 warp=float2(season_noise(p,.74,3571u),season_noise(p,.74,3917u))-.5;
    float2 a=LargeSlope.Sample(Wrap,uv+warp*.22).rg*2-1;
    float2 b=SmallSlope.Sample(Wrap,fine+warp*.19+float2(.27,.61)).rg*2-1;
    float2 c=CrossSlope.Sample(Wrap,float2(fine.y,-fine.x)+warp*.13).rg*2-1;
    c=float2(-c.y,c.x);
    float2 slope=a*.10+b*.19+c*.16;
    if(channel)slope=(RiverSlope.Sample(Wrap,fine).rg*2-1)*.16;
    float3 normal=normalize(float3(-slope,1));
    float2 raw=season_raw(p)-LabCamera.xy;
    if(SeasonWrap.x>.5)raw.x-=round(raw.x/Season.z)*Season.z;
    if(SeasonWrap.y>.5)raw.y-=round(raw.y/Season.w)*Season.w;
    float2 delta=float2(raw.x+raw.y,raw.x-raw.y)*.5;
    float3 view=normalize(float3(float2(.43,-.43)*2.5-delta,2.5));
    if(channel)view=normalize(View.xyz);
    float3 key=season_key(SunColorExposure.rgb)*Sun.w,sky=season_ambient(Ambient.rgb)*Ambient.a;
    float shadow=lab_shadow(world,float3(0,0,1));
    float depth=channel?1-saturate(river/5):smoothstep(.035,.70,-shore);
    float3 body=lerp(float3(.022,.145,.19),float3(.006,.052,.115),depth);
    body*=(sky*.70+key*.62)*(.82+.18*shadow);
    float3 optical=normalize(View.xyz);
    float fresnel=.035+.965*pow(1-saturate(dot(normal,optical)),5);
    float3 reflected=(sky*.62+key*.35)*float3(.24,.37,.54);
    float2 along=normalize(LabL.xy+float2(.00001,0));
    float3 glint_normal=normalize(float3(-(a*.12+b*.62+c*.46),1));
    float2 error=glint_normal.xy-normalize(view+LabL.xyz).xy;
    float across=dot(error,float2(-along.y,along.x)),longitudinal=dot(error,along);
    float glint=exp2(-220*across*across-18*longitudinal*longitudinal)*saturate(LabL.z);
    float facets=smoothstep(.15,.32,abs(b.y-c.x*.65));
    body*=.97+.06*(b.x*c.y);
    float3 color=lerp(body,reflected,saturate(fresnel)*.72)+key*glint*facets*(channel?.10:1.35)*shadow;
    // A variable narrow foam fringe follows actual signed coast distance.
    // Cached crossing facets break the line instead of outlining every shore.
    float coast_grain=season_noise(p,9.3,6511u)*.45+(c.x*.5+.5)*.55;
    float width=lerp(.023,.061,coast_grain);
    float openness=0;
    [unroll]for(int k=0;k<8;k++){
        float angle=k*.78539816;
        openness+=step(lab_surface(p+float2(cos(angle),sin(angle))*1.25).x,-.25);
    }
    float ocean=smoothstep(.15,.50,openness/8);
    if(!channel)color=lerp(color*float3(.94,.93,.83),color,ocean);
    float fringe=channel?0:(1-smoothstep(.006,width,abs(shore+.012)+.005*b.y))*ocean;
    float breakup=lerp(.12,.85,smoothstep(.32,.78,coast_grain));
    return lerp(color,float3(.67,.75,.79)*(sky+key*.48),fringe*breakup);
}
float3 beauty_water(float3 world,float shore,float river,bool channel){
    // Static Lab adaptation of the shared q3 natural-water response. Cached
    // slope maps supply facets; a finite reflection eye localizes the light path.
    float2 p=world.xy,uv=season_surface_uv(p,.80),fine=season_surface_uv(p,2.8);
    float2 warp=float2(season_noise(p,.74,3571u),season_noise(p,.74,3917u))-.5;
    float2 a=LargeSlope.Sample(Wrap,uv+warp*.16).rg*2-1;
    float2 b=SmallSlope.Sample(Wrap,fine+warp*.12+float2(.27,.61)).rg*2-1;
    float2 c=CrossSlope.Sample(Wrap,float2(fine.y,-fine.x)+warp*.1).rg*2-1;c=float2(-c.y,c.x);
    float2 slope=(a*.45+b*.35+c*.20)*lerp(.35,1,smoothstep(.20,.78,warp.x+.5));
    if(channel)slope=(RiverSlope.Sample(Wrap,fine).rg*2-1)*.25;
    float3 normal=normalize(float3(-slope,1));
    float2 raw=season_raw(p)-LabCamera.xy;
    if(SeasonWrap.x>.5)raw.x-=round(raw.x/Season.z)*Season.z;
    if(SeasonWrap.y>.5)raw.y-=round(raw.y/Season.w)*Season.w;
    float2 delta=float2(raw.x+raw.y,raw.x-raw.y)*.5;
    float3 view=normalize(float3(float2(.43,-.43)*2.5-delta,2.5));
    if(channel)view=normalize(View.xyz);
    float3 key=season_key(SunColorExposure.rgb)*Sun.w,sky=season_ambient(Ambient.rgb)*Ambient.a;
    float shadow=lab_shadow(world,float3(0,0,1));
    float depth=channel?1-saturate(river/5):smoothstep(.025,.55,-shore);
    float3 body=lerp(float3(.016,.12,.17),float3(.003,.040,.090),depth)*(sky*.70+key*.60)*(.78+.22*shadow);
    // The map's orthographic eye supplies Fresnel. The finite eye above is
    // only the bounded glint-path approximation, not a second scene camera.
    float3 optical_view=normalize(View.xyz);
    float fresnel=.035+.965*pow(1-saturate(dot(normal,optical_view)),5);
    float3 ray=reflect(-optical_view,normal);
    float3 reflected=(sky*.60+key*.60)*lerp(float3(.16,.25,.36),float3(.42,.55,.68),smoothstep(.30,.90,ray.y));
    float2 micro=SmallSlope.Sample(Wrap,season_surface_uv(p,3.4)+float2(.71,.29)).rg*2-1;
    float3 glint_normal=normalize(normal+float3(-micro*.03,0));
    float2 along=normalize(LabL.xy+float2(.00001,0));
    float2 error=glint_normal.xy-normalize(view+LabL.xyz).xy;
    float across=dot(error,float2(-along.y,along.x)),longitudinal=dot(error,along);
    float glint=exp2(-160*across*across-12*longitudinal*longitudinal)*saturate(LabL.z);
    float3 color=lerp(body,reflected,saturate(fresnel))+key*glint*(channel?.075:.51)*shadow;
    float foam=channel?0:(1-smoothstep(.010,.085,abs(shore)))*(.25+.45*smoothstep(.15,.72,b.x*.5+.5));
    return lerp(color,float3(.60,.68,.71)*(sky+key*.55),foam);
}
float3 winter_water(float3 world,float shore,float river,bool channel){
    float2 p=world.xy;
    float2 uv=season_surface_uv(p,.52);
    float2 fine=season_surface_uv(p,.72);
    float2 warp=float2(season_noise(p,.74,3571u),season_noise(p,.74,3917u))-.5;
    float2 slope=LargeSlope.Sample(Wrap,uv+warp*.17).rg*2-1;
    float2 micro=SmallSlope.Sample(Wrap,fine+warp*.11).rg*2-1;
    float2 crossing=CrossSlope.Sample(Wrap,float2(fine.y,-fine.x)+float2(.27,.61)).rg*2-1;
    crossing=float2(-crossing.y,crossing.x);
    slope=slope*.45+micro*.35+crossing*.20;
    if(channel)slope=lerp(slope,(RiverSlope.Sample(Wrap,fine).rg*2-1)*.70,.72);
    float3 normal=normalize(float3(-slope,1));
    float3 view=normalize(View.xyz);
    float3 key=season_key(SunColorExposure.rgb)*Sun.w;
    float3 sky=season_ambient(Ambient.rgb)*Ambient.a;
    float shadow=lab_shadow(world,float3(0,0,1));
    float depth=channel?1-saturate(river/5):smoothstep(.01,.85,-shore);
    float3 body=lerp(float3(.022,.23,.32),float3(.007,.075,.17),depth);
    body*=(sky*.72+key*.64)*(.76+.24*shadow);
    float3 ray=reflect(-view,normal);
    float fresnel=.055+.945*pow(1-saturate(dot(normal,view)),5);
    float3 reflected=(sky+key*.35)*lerp(float3(.19,.34,.52),float3(.57,.74,.94),smoothstep(.12,.86,ray.z));
    float3 color=lerp(body,reflected,fresnel);
    float2 along=normalize(LabL.xy+float2(.00001,0));
    float2 difference=normal.xy-normalize(view+LabL.xyz).xy;
    float cross_error=dot(difference,float2(-along.y,along.x)),along_error=dot(difference,along);
    float glint=exp2(-180*cross_error*cross_error-18*along_error*along_error);
    // Source-derived facets, no repeating screen-space sparkle grid.
    color+=key*glint*(channel?.070:.18)*shadow;
    if(Season.x==1){
        float foam=channel?0:1-smoothstep(.012,.065,abs(shore));
        return lerp(color*float3(.67,.77,1.02),float3(.38,.55,.60)*(sky+key*.78*shadow),foam*.35);
    }
    float coast_frost=channel ? smoothstep(3.65,5.1,river) : 1-smoothstep(.015,.070,abs(shore));
    float crust=clamp(1+(SeasonalSnowHeight.Sample(Wrap,fine*2).r-.5)*.35,.8,1.15);
    float3 rim=season_linear(float3(.79,.91,.99))*crust*(sky+key*.78*shadow);
    return lerp(color,rim,coast_frost*(channel?.30:.43));
}
Output shade(P i){
    Output o;float4 surface=lab_surface(i.world.xy);float shore=surface.x,river=surface.y;
    float3 sand=Sand.Sample(Wrap,i.world.xy*.43+float2(.31,.17)).rgb;
    if(LabDepth.w>.5){
        // The old diagnostic underlay filled every uncovered land fringe with
        // sand, creating broad beige rings even around grassy inland pools.
        float4 b=lab_biomes(i.world.xy);
        float2 ground_uv=i.world.xy*.43+float2(.31,.17);
        float3 substrate=SeasonalGroundGrass.Sample(Wrap,ground_uv).rgb*b.x+
            SeasonalGroundPlains.Sample(Wrap,float2(i.world.y,-i.world.x)*.3913+float2(.63,.29)).rgb*b.y+
            SeasonalGroundDesert.Sample(Wrap,ground_uv).rgb*b.z+
            SeasonalGroundTundra.Sample(Wrap,i.world.xy*.3612+float2(.19,.71)).rgb*b.w;
        float beach=(1-smoothstep(LabDepth.w>1.5?.004:.012,LabDepth.w>1.5?.032:.085,shore))*(1-b.z)+b.z;
        sand=lerp(substrate,sand,saturate(beach));
    }
    float3 normal=float3(0,0,1);float gloss=.04;
    lab_season_ground(sand,normal,gloss,normal,i.world,0);
    float shadow=lab_shadow(i.world,normal);
    sand*=Ambient.rgb*Ambient.a+season_key(SunColorExposure.rgb)*Sun.w*.79*shadow;
    float edge=saturate(-shore*2.5);
    float3 sea=lerp(float3(.035,.20,.29),float3(.015,.09,.19),edge);
    float wave=sin((i.world.x+i.world.y)*21+sin(i.world.y*7))*.5+
        sin((i.world.x-i.world.y)*31+cos(i.world.x*9))*.5;
    float sparkle=pow(saturate(.5+wave*.5),12);
    sea+=float3(.08,.13,.18)*sparkle*.34;
    float foam=(1-smoothstep(.018,.080,abs(shore)))*.32;
    sea=lerp(sea,float3(.47,.63,.69),foam);
    float3 channel=float3(.025,.15,.25)+float3(.03,.06,.07)*sparkle*.15;
    channel*=.60+.40*shadow;
    if(season_quality()){
        sea=winter_water(i.world,shore,river,false);
        channel=winter_water(i.world,shore,river,true);
    }
    if(LabDepth.w>.5){
        sea=beauty_water(i.world,shore,river,false);
        channel=beauty_water(i.world,shore,river,true);
    }
    if(LabDepth.w>1.5){sea=target_water(i.world,shore,river,false);channel=target_water(i.world,shore,river,true);}
    float river_mask=(1-smoothstep(4.9,7.4,river))*smoothstep(-.08,.11,shore);
    float land=smoothstep(-.005,.028,shore);
    float3 color=lerp(sea,sand,land);
    color=lerp(color,channel,river_mask);
    o.color=float4(color,1);o.validity=1;return o;
}
Output PSMain(P i){return shade(i);} Output PSFeature(P i){return shade(i);}
