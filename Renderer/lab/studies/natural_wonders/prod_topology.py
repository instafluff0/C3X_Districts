"""Natural-wonder relief slots, folded by Renderer64 (prod_tree.py merges these).

C3X only reports a wonder's name in its tile record. The renderer resolves the kit
and writes its relief slot into topology bits 28-31 of its own viewer topology,
whose revision already invalidates every relief proof that sampled the word.
Explored records name wonders, so a wonder's stamp appears with exploration.
"""

VIEWER_DOC = """// Every unexplored tile also carries hidden_bit (and loses its effect bit), so
// a revealed neighbour can tell where no terrain will be drawn.
//
// Natural wonders: an explored record that names a wonder kit with a stamp or
// foam collars sets that kit's relief slot in bits 28-31 (render_core::Tile::wonder;
// docs/natural_wonder_rendering.md). C3X reports only the name, so the kit appears with
// exploration and its neighbours' relief recompiles through this revision."""

VIEWER_MARK = """            visit([&](int x,int y,unsigned flags,unsigned wonder=0){
                bool unexplored=(flags&(C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED))==C3X_RENDERER_TILE_VISIBILITY_KNOWN;
                if(!unexplored && !wonder)return;
                if(x<0 || y<0 || x>=width || y>=height || ((x+y)&1))return;
                auto index=(std::size_t(y)*std::size_t(width)+std::size_t(x))/2;
                if(index>=count)return;
                auto value=!unexplored?(next[index]&~wonder_bits)|((wonder&15u)<<28):
                    real_relief?next[index]&~live_bits:hide(next[index]);
                if(value!=next[index]){next[index]=value;++(unexplored?hidden:wonders);}
            });"""

PATCHES = {
    "Renderer/native/render_core/viewer_topology.h": [
        ("// Every unexplored tile also carries hidden_bit (and loses its effect bit), so\n"
         "// a revealed neighbour can tell where no terrain will be drawn.", VIEWER_DOC),
        ("    static constexpr std::uint32_t live_bits=(1u<<24)|(1u<<27);\n",
         "    static constexpr std::uint32_t live_bits=(1u<<24)|(1u<<27),wonder_bits=0xf0000000u;\n"),
        ("    std::size_t hidden_tiles=0;\n", "    std::size_t hidden_tiles=0,wonder_tiles=0;\n"),
        ("    // visit(mark) calls mark(tile_x,tile_y,tile_flags) for every world input.\n",
         "    // visit(mark) calls mark(tile_x,tile_y,tile_flags[,wonder_relief_slot]) for every world input.\n"),
        ("        std::size_t hidden=0;\n", "        std::size_t hidden=0,wonders=0;\n"),
        ("            visit([&](int x,int y,unsigned flags){\n"
         "                if((flags&(C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED))!=C3X_RENDERER_TILE_VISIBILITY_KNOWN)return;\n"
         "                if(x<0 || y<0 || x>=width || y>=height || ((x+y)&1))return;\n"
         "                auto index=(std::size_t(y)*std::size_t(width)+std::size_t(x))/2;\n"
         "                if(index>=count)return;\n"
         "                auto value=real_relief?next[index]&~live_bits:hide(next[index]);\n"
         "                if(value!=next[index]){next[index]=value;++hidden;}\n"
         "            });", VIEWER_MARK),
        ("        bool next_masked=hidden!=0;\n", "        bool next_masked=hidden+wonders!=0;\n"),
        ("masked=next_masked;hidden_tiles=hidden;", "masked=next_masked;hidden_tiles=hidden;wonder_tiles=wonders;")],
    "Renderer/native/c3x_renderer.cpp": [
        ("    // Seasons: whether a chunk draws in the current season.\n",
         "    // A tile record's natural-wonder relief slot in the pack (topology bits 28-31); 0 none.\n"
         "    unsigned wonder_slot(char const* name) const {\n"
         "        auto const* kit=name[0]?natural.wonder_pack.find(name):nullptr;return kit?kit->relief_slot:0u;\n    }\n"
         "    // Seasons: whether a chunk draws in the current season.\n"),
        ("                    mark(input.occurrence.tile_x,input.occurrence.tile_y,flags);});},&masked_change);",
         "                    mark(input.occurrence.tile_x,input.occurrence.tile_y,flags,\n"
         "                        wonder_slot(input.occurrence.natural_wonder));});},&masked_change);")],
    "Renderer/native/test_loading_world_preparation.py": [
        (" std::size_t tile_geometry_cache_bytes=0;unsigned calls=0;int fail_call=0;\n",
         " std::size_t tile_geometry_cache_bytes=0;unsigned calls=0;int fail_call=0;\n"
         " unsigned wonder_slot(char const*)const{return 0;}\n")],
}
