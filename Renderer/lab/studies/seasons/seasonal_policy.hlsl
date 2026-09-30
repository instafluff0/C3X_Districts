// Generic material-stage policy. Inputs are sampled linear source materials,
// canonical world coordinates and semantic roles; no source-game identifiers.
// Summer is an exact early return. Nothing changes geometry, opacity or time.
cbuffer SeasonalState : register(b5) {
    float4 Season;       // season (summer/fall/winter/spring), enabled, width, height
    float4 SeasonRole;   // role (none/deciduous/evergreen/tropical), winter art, flower art, snow atlas
    float4 SeasonWrap;   // wrap X/Y, reserved, autumn policy (0 hue blend / 1 tint)
    float4 SeasonAtlas;  // snow grid columns/rows, admitted first cell/count
    float4 SeasonFall;   // grass RGB multiplier and blend strength; offline pack calibration
    float4 SeasonFlowers;// flower columns/rows/count and footprint in cell units
};
Texture2D SeasonalSnowColor : register(t90);
Texture2D SeasonalSnowHeight : register(t91);
Texture2D SeasonalSnowGloss : register(t92);
Texture2D SeasonalFlowerSprites : register(t93);
Texture2D SeasonalSnowPatches : register(t89);
Texture2D SeasonalFoliageColor : register(t105);
Texture2D SeasonalFoliageGloss : register(t106);

float3 season_linear(float3 c) {
    return lerp(c/12.92, pow((c+.055)/1.055,2.4), step(.04045,c));
}
float season_luma(float3 c) {return dot(c,float3(.2126,.7152,.0722));}
float2 season_raw(float2 p) {return float2(p.x+p.y-1,p.x-p.y);}
uint season_hash(uint x) {x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
float season_random(int2 p,uint salt,int2 cells) {
    if(SeasonWrap.x>.5)p.x=(p.x%cells.x+cells.x)%cells.x;
    if(SeasonWrap.y>.5)p.y=(p.y%cells.y+cells.y)%cells.y;
    return season_hash(uint(p.x)*73856093u^uint(p.y)*19349663u^salt)/4294967295.;
}
float season_noise(float2 p,float frequency,uint salt) {
    // Integral cells per actual map axis close both value and derivative at wrap.
    int2 cells=max(int2(1,1),int2(round(Season.zw*frequency)));
    float2 q=season_raw(p)*float2(cells)/max(Season.zw,float2(1,1));
    int2 i=int2(floor(q));float2 f=frac(q);f=f*f*(3-2*f);
    return lerp(lerp(season_random(i,salt,cells),season_random(i+int2(1,0),salt,cells),f.x),
        lerp(season_random(i+int2(0,1),salt,cells),season_random(i+1,salt,cells),f.x),f.y);
}
float3 season_recolor(float3 source,float3 hue,float strength,float brightness) {
    float3 target=season_linear(hue);
    target*=season_luma(source)/max(.0001,season_luma(target));
    return lerp(source,target,strength)*brightness;
}
float3 season_tint(float3 source,float3 tint,float strength,float brightness){
    float3 colored=source*lerp(float3(1,1,1),tint,strength);
    return colored*(season_luma(source)/max(.0001,season_luma(colored)))*brightness;
}
float4 season_biomes(float2 biome,float tundra,float floodplain) {
    // x grass, y plains, z desert, w tundra. Floodplain remains an independent
    // eligibility/appearance weight, supplied alongside the current terrain field.
    float4 b=float4(max(0,1-biome.x-biome.y-tundra),biome.x,biome.y,tundra);
    return b/max(.0001,dot(b,1));
}
float3 season_key(float3 key) {
    // Material-local correction counters the existing warm source response.
    // Hour/moon intensity and shared light direction remain authoritative.
    return Season.x==2 && Season.y>.5 ? key*float3(.94,1.035,1.30) : key;
}
float season_flowers(float2 p,float eligibility,out float3 color) {
    int2 cells=max(int2(1,1),int2(round(Season.zw*12)));
    float2 q=season_raw(p)*float2(cells)/max(Season.zw,float2(1,1));
    float2 dqdx=ddx(q),dqdy=ddy(q),footprint=abs(dqdx)+abs(dqdy);
    int2 cell=int2(floor(q));float2 f=frac(q);
    float patch=season_noise(p,.60,817u)*.70+season_noise(p,1.9,237u)*.30;
    patch=smoothstep(.51,.65,patch)*eligibility;
    float cluster=smoothstep(.40,.69,season_noise(p,3.6,721u));
    float pick=saturate((season_noise(p,1.2,3109u)-.25)*2);
    color=pick<.30 ? season_linear(float3(.97,.53,.73)) :
          pick<.58 ? season_linear(float3(.51,.76,.98)) :
          pick<.88 ? season_linear(float3(1,.84,.33)) :
                     season_linear(float3(.98,.96,.89));
    float fade=1-smoothstep(1.2,2.6,length(footprint));
    if(patch<=.0001 || fade<=.0001)return 0;
    float coverage=0;
    float aa=max(.025,length(footprint)*.36);
    [unroll]for(int y=-1;y<=1;y++)[unroll]for(int x=-1;x<=1;x++) {
        int2 c=cell+int2(x,y);
        float keep=season_random(c,103u,cells);
        if(keep>.06+.56*cluster)continue;
        float2 jitter=float2(season_random(c,211u,cells),season_random(c,313u,cells));
        float2 d=f-float2(x,y)-(.20+.60*jitter);
        float angle=atan2(d.y,d.x);
        float radius=.24+.065*cos(angle*5+keep*6.2831853);
        float blossom=1-smoothstep(radius-aa,radius+aa,length(d));
        if(SeasonRole.z>.5){
            if(max(abs(d.x),abs(d.y))>SeasonFlowers.w*.5)continue;
            float2 local=d/SeasonFlowers.w+.5;
            uint variant=uint(floor(season_random(c,411u,cells)*max(1,SeasonFlowers.z-.001)));
            float2 grid=max(float2(1,1),SeasonFlowers.xy);
            float2 sprite_uv=(float2(variant%uint(grid.x),variant/uint(grid.x))+saturate(local))/grid;
            float4 sprite=SeasonalFlowerSprites.SampleGrad(Clamp,sprite_uv,
                dqdx/grid/SeasonFlowers.w,dqdy/grid/SeasonFlowers.w);
            float shading=lerp(.83,1.08,saturate(season_luma(sprite.rgb)*1.3));
            blossom=sprite.a*step(max(abs(d.x),abs(d.y)),SeasonFlowers.w*.5)*shading;
        }
        coverage=max(coverage,blossom);
    }
    // At reduced zoom retain a gentle aggregate color, never a flickering stamp.
    return coverage*patch*fade;
}
float season_ground(inout float3 albedo,inout float3 normal,inout float gloss,
    float3 geometric,float3 world,float4 biome,float floodplain,float stone,float source_grain) {
    if(Season.y<.5 || Season.x==0)return 0;
    float grass=biome.x*(1-floodplain),plains=biome.y*(1-floodplain);
    if(Season.x==1) {
        bool tint=SeasonWrap.w>.5;
        float3 a=tint?season_tint(albedo,SeasonFall.rgb,SeasonFall.w,1.055):
            season_recolor(albedo,float3(.68,.59,.36),.68,1.055);
        float3 p=tint?season_tint(albedo,float3(1.13,.98,.77),.36,1.16):
            season_recolor(albedo,float3(.80,.67,.41),.35,1.16);
        float3 f=tint?season_tint(albedo,float3(.76,1.29,1.05),.62,1.055):
            season_recolor(albedo,float3(.53,.55,.34),.46,1.025);
        float active=(1-stone);
        albedo=lerp(albedo,a,grass*active);
        albedo=lerp(albedo,p,plains*active);
        albedo=lerp(albedo,f,floodplain*active);
        return 0;
    }
    if(Season.x==2) {
        float broad=season_noise(world.xy,.71,983u)*.72+season_noise(world.xy,3.8,619u)*.28;
        float target=dot(biome,float4(.93,.86,.73,.985));
        target=lerp(target,.965,floodplain);
        float upward=smoothstep(.20,.80,geometric.z);
        float coverage=saturate(target+(broad-.5)*.24)*lerp(.22,1,upward);
        coverage*=lerp(1,.62,stone);
        // Even exposed stone is cool gray; preserve relief and source grain,
        // while suppressing strong sandy/brown streaks beneath the snow.
        float3 cool_hue=biome.x*float3(.52,.65,.68)+biome.y*float3(.66,.68,.72)+
            biome.z*float3(.73,.73,.72)+biome.w*float3(.57,.66,.78);
        cool_hue=lerp(cool_hue,float3(.44,.65,.64),floodplain);
        float3 cool=season_recolor(albedo,cool_hue,.88,1.0);
        float3 snow=SeasonalSnowColor.Sample(Wrap,world.xy*.48).rgb;
        float snow_luma=season_luma(snow);
        snow=lerp(season_linear(float3(.89,.94,.995)),snow_luma.xxx,.20);
        snow*=.90+.16*season_noise(world.xy,4.7,1597u);
        // Cool differences and retained grain identify the buried biome.
        // Grass is celadon-blue, plains pearl-gray, sand pale silver, tundra icy.
        snow*=biome.x*float3(.78,.94,1.03)+biome.y*float3(1.06,1.055,1.075)+
            biome.z*float3(1.12,1.025,.89)+biome.w*float3(.70,.86,1.12);
        snow*=lerp(float3(1,1,1),float3(.90,1.10,.88),floodplain);
        // Carry the authored source microrelief into the snow's light/dark
        // grain too. Do not replace rock patches with a flat white swatch.
        snow*=.84+.32*saturate(source_grain);
        float rock_grain=.68+.55*pow(saturate(season_luma(albedo)*6),.40);
        snow*=lerp(1,rock_grain,stone*.85);
        if(SeasonRole.w>.5){
            // Generic atlas metadata comes from the offline adapter. Only
            // admitted granular cells are used, never the atlas carrier region.
            int2 cells=max(int2(1,1),int2(round(Season.zw*2.4)));
            float2 q=season_raw(world.xy)*float2(cells)/max(Season.zw,float2(1,1));
            int2 cell=int2(floor(q));float2 uv=frac(q);
            float2 grid=max(float2(1,1),SeasonAtlas.xy);
            uint variant=uint(floor(season_random(cell,1223u,cells)*max(1,SeasonAtlas.w-.001))+SeasonAtlas.z);
            float2 atlas_uv=(float2(variant%uint(grid.x),variant/uint(grid.x))+uv)/grid;
            float4 detail=SeasonalSnowPatches.SampleGrad(Wrap,atlas_uv,ddx(q)/grid,ddy(q)/grid);
            float grain=saturate(season_luma(detail.rgb)*2.4);
            snow*=lerp(1,.75+.45*grain,detail.a*.65);
        }
        albedo=lerp(cool,snow,coverage);
        float h=SeasonalSnowHeight.Sample(Wrap,world.xy*.48).r;
        float3 dx=ddx(world),dy=ddy(world),r1=cross(dy,geometric),r2=cross(geometric,dx);
        float determinant=dot(dx,r1);
        float3 grad=(ddx(h)*r1+ddy(h)*r2)*sign(determinant)/max(abs(determinant),1e-6);
        normal=normalize(lerp(normal,normalize(geometric-grad*.045),coverage*lerp(.64,.40,stone)));
        gloss=lerp(gloss,.05+.05*SeasonalSnowGloss.Sample(Wrap,world.xy*.48).r,coverage);
        return coverage;
    }
    if(Season.x==3) {
        float3 fresh=albedo*float3(1.01,1.075,1.025);
        albedo=lerp(albedo,fresh,grass*.80*(1-stone));
        float3 flood_fresh=season_tint(albedo,float3(.85,1.20,1.08),.82,1.04);
        albedo=lerp(albedo,flood_fresh,floodplain*(1-stone));
        float3 flower;
        float eligibility=(grass+plains*.24+floodplain*.75+biome.w*.055)*(1-stone);
        float cover=season_flowers(world.xy,eligibility,flower);
        albedo=lerp(albedo,flower,cover*.91);
        gloss=lerp(gloss,.035,cover);
    }
    return 0;
}
float season_foliage(inout float3 albedo,inout float3 normal,inout float gloss,
    float3 geometric,float3 world,float2 uv) {
    if(Season.y<.5 || Season.x==0 || SeasonRole.x<.5)return 0;
    // Semantic deciduous eligibility is metadata. Green chroma protects wood
    // and equipment when one authored atlas contains several material regions.
    float green=smoothstep(.006,.065,albedo.g-max(albedo.r*.85,albedo.b));
    if(Season.x==1 && SeasonRole.x<1.5) {
        float v=saturate((season_noise(world.xy,5.2,113u)-.25)*2);
        float3 hue=v<.57?float3(.99,.72,.12):v<.89?float3(.98,.55,.11):float3(.77,.29,.12);
        float3 tint=v<.57?float3(2.45,1.55,.25):v<.89?float3(3.50,.90,.30):float3(2.80,.36,.30);
        albedo=SeasonWrap.w>.5?season_tint(albedo,tint,green*.96,1.28):
            season_recolor(albedo,hue,green*.96,1.28);
    } else if(Season.x==2) {
        if(SeasonRole.y>.5){
            // Original opacity/normals remain authoritative. The source's
            // snowy albedo retains needles, dark branches and granular caps.
            albedo=SeasonalFoliageColor.Sample(Wrap,uv).rgb;
            gloss=SeasonalFoliageGloss.Sample(Wrap,uv).r;
            return 1;
        }
        float upward=smoothstep(.04,.56,geometric.z);
        float grain=season_noise(world.xy,19.3,1217u);
        float cover=upward*lerp(.71,.95,grain);
        float3 snow=season_linear(float3(.91,.955,1))*(.93+.09*grain);
        albedo=lerp(albedo*float3(.83,.93,1.07),snow,cover);
        normal=normalize(lerp(normal,geometric,cover*.40));gloss=lerp(gloss,.055,cover);
        return cover;
    } else if(Season.x==3 && SeasonRole.x<1.5) {
        albedo*=lerp(float3(1,1,1),float3(1.04,1.10,1.02),green*.40);
    }
    return 0;
}
