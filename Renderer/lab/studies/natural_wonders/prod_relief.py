"""Production relief edits for natural-wonder kits (prod_tree.py merges these).

A wonder's terrain stamp, region colours, still water and foam collars follow its
topology relief slot (bits 28-31), never a tile record, so relief keeps compiling
from topology alone (docs/natural_wonder_rendering.md, Facts).
"""

PACK = "::c3x_renderer::render_core::NaturalWonderPack"

DATA_FIELDS = f"""    bool volcano_ready=false;
    // Natural-wonder kits (docs/natural_wonder_rendering.md) from NaturalWonderRuntime:
    // terrain stamps, attached effects and foam collars, found by topology relief slot.
    // Optional: without the pack a wonder tile keeps its host terrain.
    {PACK} wonder_pack;
    std::vector<std::array<HeightField,3>> wonder_fields; // per kit: height, vegetated ground, special
"""

DATA_LOADER = f"""    template<class Read> void load_wonder_pack(Read read){{
        std::vector<std::uint8_t> bytes;wonder_pack={{}};wonder_fields.clear();
        if(!read(std::string("Renderer/packs/NaturalWonderRuntime/wonders.bin"),bytes) ||
           !wonder_pack.parse(bytes.data(),bytes.size()))return;
        wonder_fields.resize(wonder_pack.kits.size());
        for(std::size_t k=0;k<wonder_pack.kits.size();++k){{
            auto& kit=wonder_pack.kits[k];
            std::vector<std::uint8_t>* sources[]={{&kit.height,&kit.ground,&kit.special}};
            for(unsigned i=0;i<3;++i){{auto& f=wonder_fields[k][i];
                f.width=f.height=kit.field_size;f.pixels=std::move(*sources[i]);f.minimum=0;f.maximum=1;}}
        }}
    }}
    // Signed distance (tiles, negative inside) to a kit's still water (lake and channel
    // records) when its stamp is centred at (cx, cy). rise, when given, receives the
    // water's level there in relief units above the natural ground.
    static float wonder_water({PACK}::Kit const& kit,float x,float y,float cx,float cy,float* rise=nullptr){{
        float best=1e9f;
        for(std::size_t i=0;i+10<=kit.effects.size();i+=10){{
            float const* e=kit.effects.data()+i;
            if(e[0]!=float({PACK}::effect_lake) && e[0]!=float({PACK}::effect_channel))continue;
            float fx=cx+e[1],fy=cy+e[2],dx=e[3]-e[1],dy=e[4]-e[2];
            float length=std::max(std::sqrt(dx*dx+dy*dy),1e-4f),half=e[6]*.5f;
            float px=x-fx,py=y-fy,along=(px*dx+py*dy)/(length*length),across=(-px*dy+py*dx)/length/half;
            float wobble=.05f*std::sin(x*23.f+y*7.f)+.04f*std::sin(y*31.f-x*11.f);
            float distance;
            if(e[0]==float({PACK}::effect_channel)){{
                float beyond=std::max(-along,along-1)*length,side=(std::abs(across)-1)*half;
                distance=std::hypot(std::max(beyond,0.f),std::max(side,0.f))+std::min(std::max(beyond,side),0.f)+wobble*half;
            }}else distance=(std::sqrt(across*across+(2*along-1)*(2*along-1))-1+wobble)*std::min(half,length*.5f);
            if(distance<best && rise)*rise=e[7]+(e[5]-e[7])*std::clamp(along,0.f,1.f);
            best=std::min(best,distance);
        }}
        return best;
    }}
    void measure_volcano_trend(){{"""

ADD_WONDER = """    // A natural wonder's baked stamp (NaturalWonderRuntime kit index): its own span,
    // height and fixed turn, with the kit's region weights for the mountain shader.
    void add_wonder(NaturalData const& natural,unsigned kit,float x,float y,bool snow){
        auto const& k=natural.wonder_pack.kits[kit];auto const& f=natural.wonder_fields[kit];
        auto& p=pieces[count++];
        p.height_field=p.blend_field=&f[0];p.wonder_ground=&f[1];p.wonder_special=&f[2];p.wonder=&k;
        p.center_x=x;p.center_y=y;p.turn_cos=std::cos(k.turn);p.turn_sin=std::sin(k.turn);
        p.inv_long=p.inv_cross=1/k.span;
        p.height=k.units;p.reach=std::min(1.06f,reach_scale*k.span);p.snow=snow?1.f:0.f;
    }
    MountainShape()=default;"""

PATCHES = {
    "Renderer/native/render_core/terrain_query.h": [(
        "struct Tile { int base = 2, real = 2; bool present = false, hidden = false, snow = false, active = false, erupting = false; };",
        "// wonder: a natural wonder's topology relief slot (bits 28-31; docs/natural_wonder_rendering.md).\n"
        "struct Tile { int base = 2, real = 2; bool present = false, hidden = false, snow = false, active = false, erupting = false;\n"
        "    unsigned wonder = 0; };")],
    "Renderer/native/render_core/world_topology.h": [(
        "(value&(1u<<26))!=0,(value&(1u<<24))!=0,(value&(1u<<27))!=0};",
        "(value&(1u<<26))!=0,(value&(1u<<24))!=0,(value&(1u<<27))!=0,(value>>28)&15u};")],
    "Renderer/native/source_fidelity/kernels.h": [(
        "struct Tile { int source_x,source_y,column,row,real; };",
        "// wonder: a natural wonder's topology relief slot (render_core::Tile::wonder).\n"
        "struct Tile { int source_x,source_y,column,row,real; unsigned wonder=0; };")],
    "Renderer/lab/shared/natural/queries.h": [(
        "        return Tile{x,y,c,r,value.real};", "        return Tile{x,y,c,r,value.real,value.wonder};")],
    "Renderer/lab/shared/natural/data.h": [
        ('#include "../../../native/source_fidelity/kernels.h"',
         '#include "../../../native/source_fidelity/kernels.h"\n#include "../../../native/render_core/natural_wonder_pack.h"'),
        ("    bool volcano_ready=false;\n", DATA_FIELDS),
        ("        volcano_ready=load_volcano_element(read);\n",
         "        volcano_ready=load_volcano_element(read);\n        load_wonder_pack(read);\n"),
        ("    void measure_volcano_trend(){", DATA_LOADER)],
    "Renderer/lab/shared/natural/mountain_shape.h": [
        ("struct MountainCell { bool mountain=false,volcano=false,snow=false; unsigned activity=0; std::uint32_t seed=0; };",
         "// wonder: a stamped natural-wonder kit's index + 1 (its topology relief slot resolved).\n"
         "struct MountainCell { bool mountain=false,volcano=false,snow=false; unsigned activity=0; std::uint32_t seed=0; unsigned wonder=0; };"),
        ("    unsigned activity=0;\n    bool mirror=false;\n};",
         "    unsigned activity=0;\n    bool mirror=false;\n"
         "    // A natural wonder's stamp: its kit and region weight fields.\n"
         f"    HeightField const* wonder_ground=nullptr;HeightField const* wonder_special=nullptr;{PACK}::Kit const* wonder=nullptr;\n}};"),
        ("    float volcano=0,volcano_u=.5f,volcano_v=.5f,channel=0;unsigned activity=0;",
         "    float volcano=0,volcano_u=.5f,volcano_v=.5f,channel=0;unsigned activity=0;\n"
         "    // A natural-wonder stamp's region weights (vegetated ground, special material),\n"
         "    // easing out where another piece takes over, and its region kind wherever its\n"
         "    // reach covers, so the kind cannot blend across a triangle.\n"
         "    float wonder_ground=0,wonder_special=0;unsigned wonder_kind=0;"),
        ("    MountainShape()=default;", ADD_WONDER),
        ("                auto t=lookup(nc+dc,nr+dr);bool volcano=t.real==10,mountain=t.real==6 || volcano;\n"
         "                unsigned bits=mountain?unsigned(flags(nc+dc,nr+dr)):0u;\n"
         "                cells[i]=MountainCell{mountain,volcano,!volcano && (bits&mountain_snow_cap),\n"
         "                    volcano?((bits&volcano_erupting)?2u:(bits&volcano_active)?1u:0u):0u,mountain_seed(t)};",
         "                auto t=lookup(nc+dc,nr+dr);\n"
         "                // A natural wonder's stamp raises its tile on any host terrain.\n"
         "                auto const* kit=natural.wonder_pack.relief(t.wonder);\n"
         "                unsigned wonder=kit && kit->stamp()?unsigned(kit-natural.wonder_pack.kits.data())+1:0u;\n"
         "                bool volcano=t.real==10 && !wonder,mountain=t.real==6 || volcano || wonder;\n"
         "                unsigned bits=mountain?unsigned(flags(nc+dc,nr+dr)):0u;\n"
         f"                bool snow=wonder?(kit->flags&{PACK}::flag_snow)!=0:!volcano && (bits&mountain_snow_cap);\n"
         "                cells[i]=MountainCell{mountain,volcano,snow,\n"
         "                    volcano?((bits&volcano_erupting)?2u:(bits&volcano_active)?1u:0u):0u,mountain_seed(t),wonder};"),
        ("            std::uint32_t seed=mix(owner.seed);\n",
         "            std::uint32_t seed=mix(owner.seed);\n"
         "            if(owner.wonder){\n"
         "                add_wonder(natural,owner.wonder-1,float(nc+dc)+.5f,float(nr+dr)+.5f,owner.snow);\n"
         "                continue;\n"
         "            }\n"),
        ("            if(distance>=p.reach)continue;\n",
         "            if(distance>=p.reach)continue;\n"
         "            if(p.wonder)result.wonder_kind=p.wonder->region_kind;\n"),
        ("            float shape=std::max(0.f,h)*smooth01((blend-.28f)/.34f);\n",
         "            float shape=std::max(0.f,h)*smooth01((blend-.28f)/.34f);\n"
         "            if(p.wonder)shape=std::max(0.f,h);\n"),
        ("        MountainSample result;float second=0,volcano=0,other=0;",
         "        MountainSample result;float second=0,volcano=0,other=0,norm=height,wonder_g=0,wonder_s=0;"),
        ("                result.u=u;result.v=v;result.snow=p.snow;",
         "                result.u=u;result.v=v;result.snow=p.snow;norm=p.wonder?p.height:height;\n"
         "                wonder_g=p.wonder?p.wonder_ground->sample(u,v):0.f;\n"
         "                wonder_s=p.wonder?p.wonder_special->sample(u,v):0.f;"),
        ("        result.height=std::min(1.f,result.displacement/height);",
         "        // Snow and top-rock bands follow a wonder's own peak, not the standard mountain.\n"
         "        result.height=std::min(1.f,result.displacement/norm);\n"
         "        float owned=smooth01((result.dominant-second)/blend_k+.5f);\n"
         "        result.wonder_ground=wonder_g*owned;result.wonder_special=wonder_s*owned;")],
    "Renderer/lab/shared/natural/relief_mesh_body.h": [
        ("                    out.base_terrain=42+coverage*source_weight;",
         "                    // A natural wonder's relief opens where it falls below the kit's still\n"
         "                    // water (drawn by the production water layers), as land does at a shore.\n"
         "                    // Height above the water is linear across each triangle, so the shader's\n"
         "                    // coverage clip follows the exact waterline rather than the vertex grid.\n"
         "                    float wonder_shore=1e9f;\n"
         "                    for(unsigned piece=0;piece<mountain_shape.count;++piece)if(mountain_shape.pieces[piece].wonder){\n"
         "                        auto const& wonder_piece=mountain_shape.pieces[piece];float rise=0;\n"
         "                        if(natural.wonder_water(*wonder_piece.wonder,world_x,world_y,wonder_piece.center_x,\n"
         "                                wonder_piece.center_y,&rise)<0)\n"
         "                            wonder_shore=std::min(wonder_shore,(elevation-height_natural(wonder_piece.center_x,wonder_piece.center_y,nullptr)-rise)/6.f);\n"
         "                    }\n"
         "                    out.base_terrain=42+std::max(-.5f,std::min(coverage*source_weight,wonder_shore));"),
        ("out.relief_owner_u=sample.volcano_u-.5f;out.relief_owner_v=sample.volcano_v-.5f;",
         "// Without a volcano the owner channel carries a wonder stamp's region weights\n"
         "                    // (and its state the region kind; beauty_mountain.hlsl).\n"
         "                    out.relief_owner_u=sample.volcano>0?sample.volcano_u-.5f:sample.wonder_ground*flat_blend;\n"
         "                    out.relief_owner_v=sample.volcano>0?sample.volcano_v-.5f:sample.wonder_special*flat_blend;"),
        ("out.relief_owner_state=2.f*float(sample.activity)+sample.channel;",
         "out.relief_owner_state=sample.volcano>0?2.f*float(sample.activity)+sample.channel:float(sample.wonder_kind);")],
}
