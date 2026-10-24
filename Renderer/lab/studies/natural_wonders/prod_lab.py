"""Lab preview support for the natural-wonder port (prod_tree.py merges these): a Lab scene
marks its wonder tile with BIQ bonus 0x40000000 and names it in C3X_LAB_NATURAL_WONDER; the
preview fills the tile record as C3X does."""

WORLD = """    QueryPerformanceCounter(&definitions_done);
    // Lab natural wonder (studies/natural_wonders): BIQ bonus 0x40000000 marks its tile and
    // C3X_LAB_NATURAL_WONDER names it in the tile record, as C3X reports it; Renderer64
    // resolves the kit and its relief slot itself.
    char lab_wonder[32]={};GetEnvironmentVariableA("C3X_LAB_NATURAL_WONDER",lab_wonder,sizeof(lab_wonder));
"""

PATCHES = {
    "Renderer/native/biq_preview.cpp": [
        ("    QueryPerformanceCounter(&definitions_done);\n", WORLD),
        ("        tile.square_parts = source.bonus;\n",
         "        tile.square_parts = source.bonus & ~0x40000000u;\n"
         "        if((source.bonus&0x40000000u) && lab_wonder[0]){\n"
         "            std::memcpy(tile.natural_wonder,lab_wonder,sizeof(tile.natural_wonder));\n"
         "            tile.natural_wonder_flags=C3X_RENDERER_NATURAL_WONDER_NAME;\n"
         "        }\n")],
}
