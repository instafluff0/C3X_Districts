// Generic material-stage policy. Inputs are sampled linear source materials,
// canonical world coordinates and semantic roles; no source-game identifiers.
// Summer is an exact early return. Nothing changes geometry, opacity or time.
cbuffer SeasonalState : register(b5) {
    float4 Season;       // season (summer/fall/winter/spring), enabled, width, height
    float4 SeasonRole;   // role (none/deciduous/evergreen/tropical), winter art, flower art, snow atlas
    float4 SeasonWrap;   // wrap X/Y, UV exposure mask present, autumn policy (0 hue blend / 1 tint)
    float4 SeasonAtlas;  // snow grid columns/rows, admitted first cell/count
    float4 SeasonFall;   // grass RGB multiplier and blend strength; offline pack calibration
    float4 SeasonFlowers;// flower columns/rows/count and footprint in cell units
    float4 SeasonSnow;   // layered deposition enabled, drift height, deposition, frost relief
    float4 SeasonAutumn; // refined profile enabled, leaf brightness, litter blend, grass brightness
    float4 SeasonLeaf;   // source linear leaf p10/p50/p90; explicit tissue mask present
    float4 SeasonCrown;  // broad irradiance blend, transmission, turf contrast, floor relief contrast
};
Texture2D SeasonalSnowColor : register(t90);
Texture2D SeasonalSnowHeight : register(t91);
Texture2D SeasonalSnowGloss : register(t92);
Texture2D SeasonalFlowerSprites : register(t93);
Texture2D SeasonalSnowPatches : register(t89);
Texture2D SeasonalSnowPatchRelief : register(t88);
Texture2D SeasonalFoliageColor : register(t105);
Texture2D SeasonalFoliageGloss : register(t106);
Texture2D SeasonalFoliageExposure : register(t104);
Texture2D SeasonalFoliageSource : register(t107);
Texture2D SeasonalLeafTissue : register(t108);
Texture2D SeasonalGroundGrass : register(t110);
Texture2D SeasonalGroundPlains : register(t111);
Texture2D SeasonalGroundDesert : register(t112);
Texture2D SeasonalGroundTundra : register(t113);
Texture2D SeasonalGrassHeight : register(t114);
Texture2D SeasonalPlainsHeight : register(t115);
Texture2D SeasonalFloorColor : register(t116);

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
    float3 correction=SeasonRole.x>.5 ? float3(.95,1.34,1.86) : float3(.94,1.035,1.30);
    if(Season.x==1 && Season.y>.5 && SeasonAutumn.x>2.5)
        return key*(SeasonRole.x>.5 && SeasonRole.x<1.5?float3(.98,1.08,1.16):float3(.98,1.0,1.08));
    if(Season.x==1 && Season.y>.5 && SeasonAutumn.x>.5 && SeasonRole.x>.5 && SeasonRole.x<1.5)
        return key*float3(.98,1.12,1.18);
    return Season.x==2 && Season.y>.5 ? key*correction : key;
}
float3 season_ambient(float3 ambient) {
    if(Season.x==1 && Season.y>.5 && SeasonAutumn.x>2.5)return ambient*float3(.86,.98,1.12);
    return Season.x==2 && Season.y>.5 ? ambient*float3(.24,.55,1.10) : ambient;
}
float2 season_surface_uv(float2 p,float frequency) {
    // Even integral raw-axis periods preserve material value and derivatives
    // when either canonical map axis wraps through the isometric basis.
    float2 cells=max(float2(2,2),round(Season.zw*frequency*.5)*2);
    float2 q=season_raw(p)*cells/max(float2(1,1),Season.zw);
    return float2(q.x+q.y,q.x-q.y)*.5+float2(.31,.17);
}
bool season_quality() {
    return Season.y>.5 && ((Season.x==2 && SeasonSnow.x>.5) || (Season.x==1 && SeasonAutumn.x>.5));
}
bool season_autumn_beauty() {return Season.x==1 && Season.y>.5 && SeasonAutumn.x>1.5;}
bool season_autumn_match() {return Season.x==1 && Season.y>.5 && SeasonAutumn.x>2.5;}
float season_crown_diffuse(float original,float3 normal,float3 crown,float3 light,float leaf) {
    if(leaf<=0 || dot(crown,crown)<.01)return original;
    float3 broad=normalize(crown);
    float wrap=saturate((dot(broad,light)+.12)/1.12);
    // Broad volume is a second irradiance scale. Authored mapped normals remain
    // in the detailed term and in the source gloss response; no normal replacement.
    float detail=saturate((dot(normal,light)+.08)/1.08);
    return lerp(original,lerp(detail,wrap,SeasonCrown.x),leaf);
}
float3 season_crown_transmission(float3 albedo,float3 normal,float3 crown,float3 light,float3 key,float shadow,float leaf) {
    if(!season_autumn_match() || leaf<=0 || dot(crown,crown)<.01)return 0;
    float back=saturate(-dot(normal,light)*.7+.25);
    float exposed=smoothstep(-.3,.8,normalize(crown).z);
    return albedo*key*SeasonCrown.y*back*exposed*shadow*leaf;
}
void season_leaf_litter(inout float3 albedo,float3 world,float eligibility=1,float2 uv=float2(0,0)) {
    if(Season.y<.5 || Season.x!=1 || SeasonAutumn.x<.5)return;
    if(season_autumn_match()){
        float3 source=SeasonalFloorColor.Sample(Clamp,uv).rgb;
        float average=season_luma(SeasonalFloorColor.SampleBias(Clamp,uv,3.5).rgb);
        float relative=season_luma(source)/max(.008,average);
        float leaf=smoothstep(.92,1.28,relative)*eligibility;
        float3 hue=lerp(float3(.91,.66,.19),float3(.78,.43,.10),season_noise(world.xy,2.1,2801u));
        float3 litter=season_recolor(albedo,hue,.90,1.22);
        litter*=clamp(1+(relative-1)*SeasonCrown.w,.75,1.30);
        albedo=lerp(albedo,litter,leaf*SeasonAutumn.z);
        return;
    }
    // Existing forest-floor color, height, alpha and footprint supply all detail.
    // A broad broken tint envelope leaves gaps, rather than stamping new leaves.
    float patch=smoothstep(.31,.68,season_noise(world.xy,3.4,2203u))*eligibility;
    float v=season_noise(world.xy,1.8,2801u);
    float3 tint=lerp(float3(3.0,1.65,.43),float3(3.8,1.02,.35),v);
    albedo=season_tint(albedo,tint,patch*SeasonAutumn.z,1+patch*.24);
}
float season_drift(float2 p) {
    // Jittered, elongated local pillows avoid an uninterrupted wave train.
    // Canonical cells and their neighbors close at both map seams. No height
    // or tree geometry changes; this field supplies bounded material relief.
    int2 cells=max(int2(1,1),int2(round(Season.zw*float2(2.2,.92))));
    float2 q=season_raw(p)*float2(cells)/max(Season.zw,float2(1,1));
    int2 cell=int2(floor(q));float2 f=frac(q);float mound=0;
    [unroll]for(int y=-1;y<=1;y++)[unroll]for(int x=-1;x<=1;x++){
        int2 c=cell+int2(x,y);
        float a=season_random(c,4523u,cells),b=season_random(c,5237u,cells);
        float2 d=f-float2(x,y)-(.18+.64*float2(a,b));
        float angle=(a-.5)*.7;float co=cos(angle),si=sin(angle);
        d=float2(d.x*co-d.y*si,d.x*si+d.y*co);
        float radius=dot(d*float2(1.05,1.35),d*float2(1.05,1.35));
        float pillow=exp2(-radius*5.5)*(1-smoothstep(.72,1.0,radius));
        mound=max(mound,pillow*lerp(.35,1,season_random(c,6311u,cells)));
    }
    return mound*.74+season_noise(p,3.7,6833u)*.26;
}
float3 season_height_gradient(float3 world,float3 geometric,float height) {
    float3 dx=ddx(world),dy=ddy(world),r1=cross(dy,geometric),r2=cross(geometric,dx);
    float determinant=dot(dx,r1);
    return (ddx(height)*r1+ddy(height)*r2)*sign(determinant)/max(abs(determinant),1e-6);
}
float3 season_snow_palette(float4 biome,float floodplain) {
    float3 grass=SeasonSnow.x>.5?float3(.94,.975,1.0):float3(.78,.94,1.03);
    float3 plains=SeasonSnow.x>.5?float3(1.06,1.08,.93):float3(1.06,1.055,1.075);
    float3 desert=SeasonSnow.x>.5?float3(1.12,1.025,.73):float3(1.12,1.025,.89);
    float3 palette=biome.x*grass+biome.y*plains+biome.z*desert+biome.w*float3(.70,.86,1.12);
    return palette*lerp(float3(1,1,1),float3(.90,1.10,.88),floodplain);
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
        if(season_autumn_match()){
            float2 guv=season_surface_uv(world.xy,.43);
            float2 plains_uv=season_surface_uv(world.xy,.3913);
            float2 puv=float2(plains_uv.y,-plains_uv.x)+float2(.63,.29);
            float gy=season_luma(SeasonalGroundGrass.SampleBias(Wrap,guv,4).rgb);
            float gm=season_luma(SeasonalGroundGrass.SampleBias(Wrap,guv,8).rgb);
            float py=season_luma(SeasonalGroundPlains.SampleBias(Wrap,puv,4).rgb);
            float pm=season_luma(SeasonalGroundPlains.SampleBias(Wrap,puv,8).rgb);
            float turf=gy/max(.015,gm),straw=py/max(.015,pm);
            float broad=season_noise(world.xy,.63,3677u);
            float dry=saturate(pow(smoothstep(.92,1.23,turf),1.55)*.80+broad*.20);
            float3 grass_hue=lerp(float3(.57,.59,.32),float3(.85,.70,.38),dry);
            float3 a=season_recolor(albedo,grass_hue,.84,SeasonAutumn.w);
            a*=clamp(1+(turf-1)*SeasonCrown.z,.73,1.28)*lerp(.94,1.08,broad);
            float3 p=season_recolor(albedo,float3(.86,.73,.46),.78,1.13);
            p*=clamp(1+(straw-1)*SeasonCrown.z*.55,.80,1.20);
            float3 f=season_recolor(albedo,float3(.48,.62,.35),.78,.98);
            float active=1-stone;
            albedo=lerp(albedo,a,grass*active);
            albedo=lerp(albedo,p,plains*active);
            albedo=lerp(albedo,f,floodplain*active);
            // Exposed stone retains its original patterns and normals. This
            // restrained neutral grade makes gray facets readable beside turf.
            float3 rock=season_recolor(albedo,float3(.80,.81,.80),.70,1.09);
            albedo=lerp(albedo,rock,stone*(1-biome.z));
            // A second, larger source-height band complements the retained
            // fine normal. Bounded perturbation keeps the original relief and
            // excludes stone, desert and tundra; no mesh/displacement changes.
            float gh=SeasonalGrassHeight.SampleBias(Wrap,guv,4).r;
            float ph=SeasonalPlainsHeight.SampleBias(Wrap,puv,4).r;
            float relief=(gh*grass+ph*plains)*active;
            float3 gradient=season_height_gradient(world,geometric,relief)*.012*SeasonCrown.z;
            gradient*=min(1,.16/max(.001,length(gradient)));
            normal=normalize(normal-gradient);
            return 0;
        }
        bool tint=SeasonWrap.w>.5;
        float3 a=tint?season_tint(albedo,SeasonFall.rgb,SeasonFall.w,SeasonAutumn.x>.5?SeasonAutumn.w:1.055):
            season_recolor(albedo,float3(.68,.59,.36),.68,1.055);
        float3 p=tint?season_tint(albedo,SeasonAutumn.x>1.5?float3(1.18,1.00,1.08):SeasonAutumn.x>.5?float3(1.18,1.00,.82):float3(1.13,.98,.77),.36,1.16):
            season_recolor(albedo,float3(.80,.67,.41),.35,1.16);
        float3 f=tint?season_tint(albedo,float3(.76,1.29,1.05),.62,1.055):
            season_recolor(albedo,float3(.53,.55,.34),.46,1.025);
        float active=(1-stone);
        if(SeasonAutumn.x>1.5){
            // Quiet bronze/olive regions above the original composed grain.
            // Slow world variation breaks a yellow wash without new ground art.
            float warm=smoothstep(.28,.72,season_noise(world.xy,.57,3677u));
            a=season_tint(a,lerp(float3(.95,1.035,1.06),float3(1.12,.965,.92),warm),.52,1);
            a*=lerp(.92,1.07,warm);
        }
        albedo=lerp(albedo,a,grass*active);
        albedo=lerp(albedo,p,plains*active);
        albedo=lerp(albedo,f,floodplain*active);
        return 0;
    }
    if(Season.x==2) {
        float source_mean=dot(biome,float4(
            season_luma(SeasonalGroundGrass.SampleLevel(Wrap,float2(.5,.5),12).rgb),
            season_luma(SeasonalGroundPlains.SampleLevel(Wrap,float2(.5,.5),12).rgb),
            season_luma(SeasonalGroundDesert.SampleLevel(Wrap,float2(.5,.5),12).rgb),
            season_luma(SeasonalGroundTundra.SampleLevel(Wrap,float2(.5,.5),12).rgb)));
        // Use the already-composed substrate at this exact surface point.
        // This preserves grass grit, straw, dune streaks and rock patterns in
        // the snow without carrying their summer green/brown chroma across.
        float substrate_contrast=clamp(season_luma(albedo)/max(.025,source_mean),.64,1.46);
        float broad=season_noise(world.xy,.71,983u)*.72+season_noise(world.xy,3.8,619u)*.28;
        float target=dot(biome,float4(.93,.86,.73,.985));
        target=lerp(target,.965,floodplain);
        float upward=smoothstep(.20,.80,geometric.z);
        float coverage=saturate(target+(broad-.5)*.24)*lerp(.16,1,upward);
        coverage*=lerp(1,.56,stone);
        float drift=SeasonSnow.x>.5?season_drift(world.xy):.5;
        if(SeasonSnow.x>.5){
            // Retained material highs break through thin snow; troughs and
            // sheltered lee-facing slopes collect it. Desert remains wind
            // scoured, while wet floodplains carry a more continuous crust.
            float grain=smoothstep(.24,.77,source_grain);
            float shelter=saturate(.5+dot(geometric.xy,float2(.81,.59))*.75);
            float depth=(drift-.45)*.42+(shelter-.5)*.15-(grain-.5)*.17;
            coverage=saturate(coverage+depth*SeasonSnow.z*lerp(1,.65,stone));
            coverage*=1-smoothstep(.44,.83,grain)*stone*.16;
        }
        // Even exposed stone is cool gray; preserve relief and source grain,
        // while suppressing strong sandy/brown streaks beneath the snow.
        float3 cool_hue=biome.x*float3(.52,.65,.68)+biome.y*float3(.66,.68,.72)+
            biome.z*float3(.73,.73,.72)+biome.w*float3(.57,.66,.78);
        cool_hue=lerp(cool_hue,float3(.44,.65,.64),floodplain);
        float3 cool=season_recolor(albedo,cool_hue,.88,1.0);
        float2 snow_uv=season_surface_uv(world.xy,.48);
        float2 fine_uv=season_surface_uv(world.xy,.94);
        float3 snow_source=SeasonalSnowColor.Sample(Wrap,snow_uv).rgb;
        float snow_luma=season_luma(snow_source);
        float snow_mean=season_luma(SeasonalSnowColor.SampleBias(Wrap,snow_uv,5).rgb);
        // Normalize the source's average gray to pearl white, retaining its
        // full local contrast. The old 80% constant-color blend erased it.
        float snow_grain=clamp(snow_luma/max(.025,snow_mean),.55,1.30);
        float h=SeasonalSnowHeight.Sample(Wrap,snow_uv).r;
        float mean_h=SeasonalSnowHeight.SampleBias(Wrap,snow_uv,5).r;
        float fine_h=SeasonalSnowHeight.Sample(Wrap,fine_uv).r;
        float fine_mean=SeasonalSnowHeight.SampleBias(Wrap,fine_uv,4).r;
        float cavity=clamp(1+(h-mean_h)*2.0+(fine_h-fine_mean)*1.0,.72,1.16);
        float3 snow=season_linear(float3(.97,.985,1))*pow(snow_grain,1.12)*cavity*1.08;
        snow*=.94+.12*season_noise(world.xy,4.7,1597u);
        if(SeasonSnow.x>.5){
            // Quiet luminous shoulders and blue translucent-looking lows
            // create an organized middle scale above the authored fine grain.
            snow*=lerp(float3(.955,.975,1.015),float3(1.025,1.025,1.02),smoothstep(.18,.78,drift));
        }
        // Cool differences and retained grain identify the buried biome.
        // Grass is celadon-blue, plains pearl-gray, sand pale silver, tundra icy.
        snow*=season_snow_palette(biome,floodplain);
        // Carry the authored source microrelief into the snow's light/dark
        // grain too. Do not replace rock patches with a flat white swatch.
        snow*=.83+.34*saturate(source_grain);
        if(SeasonSnow.x>.5)snow*=pow(substrate_contrast,lerp(.52,.30,stone));
        float rock_grain=.68+.55*pow(saturate(season_luma(albedo)*6),.40);
        snow*=lerp(1,rock_grain,stone*.85);
        float3 patch_relief=0;
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
            float2 encoded=SeasonalSnowPatchRelief.SampleGrad(Wrap,atlas_uv,ddx(q)/grid,ddy(q)/grid).rg;
            // The retained slot is a heightmap, not a confirmed tangent
            // normal. R is centered relief; G's source meaning remains
            // unresolved. Derive C3X slopes from R and keep G untouched.
            patch_relief=-season_height_gradient(world,geometric,(encoded.r-.5)*detail.a)*.018;
        }
        albedo=lerp(cool,snow,coverage);
        float relief=(h-mean_h)+(fine_h-fine_mean)*.32;
        float3 grad=season_height_gradient(world,geometric,relief);
        // Snow adds its own relief over the surviving substrate normal. Rock
        // and hill normals no longer get blended 64% toward a smooth plane.
        float3 retained=normalize(lerp(normal,geometric,coverage*.12*(1-stone)));
        normal=normalize(retained-grad*coverage*lerp(.075,.040,stone)+patch_relief*coverage);
        if(SeasonSnow.x>.5){
            float drift_height=drift*SeasonSnow.y*dot(biome,float4(.95,1.20,1.45,.65))*lerp(1,.45,stone)*upward;
            normal=normalize(normal-season_height_gradient(world,geometric,drift_height)*coverage);
        }
        gloss=lerp(gloss,.10+.12*SeasonalSnowGloss.Sample(Wrap,snow_uv).r,coverage);
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
    float3 geometric,float3 world,float2 uv,float appearance=-1,float tissue=-1) {
    if(Season.y<.5 || Season.x==0 || SeasonRole.x<.5)return 0;
    // Semantic deciduous eligibility is metadata. Green chroma protects wood
    // and equipment when one authored atlas contains several material regions.
    float green=smoothstep(.006,.065,albedo.g-max(albedo.r*.85,albedo.b));
    if(Season.x==1 && SeasonRole.x<1.5) {
        if(season_autumn_match()){
            float3 source=SeasonalFoliageSource.Sample(Wrap,uv).rgb;
            float slow=season_luma(SeasonalFoliageSource.SampleBias(Wrap,uv,3).rgb);
            float leaf=SeasonLeaf.w>.5?SeasonalLeafTissue.Sample(Wrap,uv).r:green;
            if(tissue>=0)leaf*=saturate(tissue);
            float wood=(albedo.r-albedo.g)/max(.01,max(albedo.r,max(albedo.g,albedo.b)));
            leaf*=1-smoothstep(.20,.45,wood);
            float low=SeasonLeaf.w>.5?SeasonLeaf.x:.04;
            float high=SeasonLeaf.w>.5?SeasonLeaf.z:.20;
            float value=saturate((slow-low)/max(.008,high-low));
            float v=appearance<0?season_noise(world.xy,.71,113u):appearance;
            float3 shade=season_linear(v<.76?float3(.62,.42,.075):v<.97?float3(.65,.28,.05):float3(.46,.12,.035));
            float3 mid=season_linear(v<.76?float3(.99,.78,.20):v<.97?float3(1,.57,.095):float3(.84,.29,.065));
            float3 top=season_linear(v<.76?float3(1,.93,.48):v<.97?float3(1,.78,.30):float3(.99,.53,.18));
            float3 ramp=lerp(shade,mid,smoothstep(.02,.56,value));
            ramp=lerp(ramp,top,smoothstep(.48,.98,value));
            float fine=clamp(pow(season_luma(source)/max(.006,slow),.78),.32,1.48);
            float3 autumn=ramp*.62*SeasonAutumn.y*fine;
            albedo=lerp(albedo,autumn,leaf);
            return leaf;
        }
        if(SeasonAutumn.x>.5){
            // One stable palette per original tree. UV-scale leaf detail and
            // dark interiors survive; crown boundaries never follow noise cells.
            float v=appearance<0?season_noise(world.xy,.71,113u):appearance;
            float3 tint=v<.67?float3(3.6,1.85,.32):v<.93?float3(4.6,1.12,.26):float3(4.0,.55,.29);
            float leaf=smoothstep(.001,.032,albedo.g-max(albedo.r*.82,albedo.b*.95));
            float variation=lerp(.94,1.06,season_noise(world.xy,13.1,1427u));
            if(SeasonAutumn.x>1.5){
                float3 original=SeasonalFoliageSource.Sample(Wrap,uv).rgb;
                // Test the original atlas before its optional owner-color tint.
                // Normalize chroma so shaded green texels remain leaf eligible.
                float chroma=(original.g-max(original.r,original.b))/max(.01,max(original.r,max(original.g,original.b)));
                leaf=smoothstep(.008,.10,chroma);
                float wood=(albedo.r-albedo.g)/max(.01,max(albedo.r,max(albedo.g,albedo.b)));
                leaf*=1-smoothstep(.20,.45,wood);
                // Stronger red/green ratios remove chlorophyll's olive cast.
                // This still multiplies source texels and retains their variation.
                tint=v<.67?float3(6.0,2.30,.40):v<.93?float3(9.0,1.45,.35):float3(7.0,.55,.35);
            }
            albedo=season_tint(albedo,tint,leaf*.98,lerp(1,SeasonAutumn.y*variation,leaf));
            return leaf;
        }
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
            float cap=smoothstep(.24,.56,season_luma(albedo));
            float3 pearl=season_recolor(albedo,float3(.94,.98,1),.75,1.45);
            albedo=lerp(albedo*float3(.72,.85,1.12),pearl,cap);
            gloss=SeasonalFoliageGloss.Sample(Wrap,uv).r;
            return 1;
        }
        float upward=smoothstep(.32,.88,geometric.z)*.45+smoothstep(.30,.87,normal.z)*.55;
        float grain=season_noise(world.xy,19.3,1217u);
        float source_luma=season_luma(albedo);
        float cover=upward*lerp(.73,.94,grain)*green*smoothstep(.026,.15,source_luma);
        if(SeasonWrap.z>.5){
            float exposure=SeasonalFoliageExposure.Sample(Wrap,uv).r;
            cover*=SeasonSnow.x>.5?lerp(.28,1.12,smoothstep(.12,.75,exposure)):lerp(.62,1.08,exposure);
        }
        float mean_luma=season_luma(SeasonalFoliageSource.SampleBias(Wrap,uv,3).rgb);
        float relative=source_luma/max(.025,mean_luma);
        // Existing leaf/bough detail selects small frost clumps. Preserve dark
        // interstices instead of treating the whole crown as a white surface.
        float clumps=smoothstep(.82,1.12,relative+(grain-.5)*.32);
        cover*=lerp(SeasonSnow.x>.5?.22:.38,1,clumps);
        float structure=clamp(pow(relative,.65),.26,1.22);
        float2 frost_uv=season_surface_uv(world.xy,.94);
        float frost_luma=season_luma(SeasonalSnowColor.Sample(Wrap,frost_uv).rgb);
        float frost_mean=season_luma(SeasonalSnowColor.SampleBias(Wrap,frost_uv,4).rgb);
        float frost=clamp(frost_luma/max(.025,frost_mean),.65,1.25);
        float3 snow=season_linear(float3(.96,.98,1))*(.93+.10*grain)*structure*frost;
        float3 substrate=lerp(albedo*float3(.69,.84,1.08),
            season_recolor(albedo,float3(.33,.46,.53),.78,.73),green);
        albedo=lerp(substrate,snow,cover);
        normal=normalize(lerp(normal,geometric,cover*.12));
        if(SeasonSnow.x>.5){
            // A little crust relief follows the same exposed leaf cards. The
            // original opacity and mapped bough normals still define the tree.
            float frost_h=SeasonalSnowHeight.Sample(Wrap,frost_uv).r;
            float frost_low=SeasonalSnowHeight.SampleBias(Wrap,frost_uv,4).r;
            normal=normalize(normal-season_height_gradient(world,geometric,frost_h-frost_low)*SeasonSnow.w*cover);
        }
        gloss=lerp(gloss,.09,cover);
        return cover;
    } else if(Season.x==3 && SeasonRole.x<1.5) {
        albedo*=lerp(float3(1,1,1),float3(1.04,1.10,1.02),green*.40);
    }
    return 0;
}
