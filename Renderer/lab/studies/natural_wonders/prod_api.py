"""Frame ABI 23 for natural wonders (prod_tree.py merges these): the tile record's wonder
name and show-name flag, their transport, keys and tests (docs/natural_wonder_rendering.md, Facts).
Follows the Day-night session's API 22 (minute)."""

TILE_FIELDS = """    /* 0 means no city-site overlay; 1..11 are C3X's existing grade + 1. */
    c3x_renderer_u32 city_site_grade;
    /* API 23: the C3X natural wonder on this tile by its effective config name
       (empty when none; withheld with every object on unexplored tiles), and its
       flags (C3X_RENDERER_NATURAL_WONDER_NAME: C3X shows the name on the map). */
    char natural_wonder[32];
    c3x_renderer_u32 natural_wonder_flags;
};"""

PATCHES = {
    "Renderer/native/c3x_renderer_api.h": [
        ("#define C3X_RENDERER_API_VERSION 22u", "#define C3X_RENDERER_API_VERSION 23u"),
        ("enum c3x_renderer_invalidation_flags {\n",
         "enum c3x_renderer_natural_wonder_flags {\n    C3X_RENDERER_NATURAL_WONDER_NAME = 1u\n};\n\n"
         "enum c3x_renderer_invalidation_flags {\n"),
        ("    /* 0 means no city-site overlay; 1..11 are C3X's existing grade + 1. */\n"
         "    c3x_renderer_u32 city_site_grade;\n};", TILE_FIELDS)],
    "Renderer/native/input_recording/codec.h": [
        ("constexpr std::uint32_t protocol_version=13;", "constexpr std::uint32_t protocol_version=14;"),
        ("    io(v.barbarian_tribe_id);\n    io(v.city_site_grade);\n}",
         "    io(v.barbarian_tribe_id);\n    io(v.city_site_grade);\n    io(v.natural_wonder);\n    io(v.natural_wonder_flags);\n}")],
    "Renderer/native/helper_trial/scene_wire.h": [
        ("wire_version=17,", "wire_version=18,")],
    "Renderer/tools/scripted_game_test.ps1": [
        ("    public const int WireVersion = 17, FrameOffset", "    public const int WireVersion = 18, FrameOffset")],
    "Renderer/native/world_preparation.h": [
        ("    constexpr auto text_end=offsetof(c3x_renderer_tile_v1,barbarian_tribe_id);\n"
         "    static_assert(text_begin%4==0 && sizeof(facts)==text_end+8,\"object input layout\");",
         "    constexpr auto text_end=offsetof(c3x_renderer_tile_v1,barbarian_tribe_id);\n"
         "    constexpr auto wonder_begin=offsetof(c3x_renderer_tile_v1,natural_wonder);\n"
         "    static_assert(text_begin%4==0 && wonder_begin==text_end+8 &&\n"
         "        sizeof(facts)==wonder_begin+sizeof(facts.natural_wonder)+4,\"object input layout\");"),
        ("    scalar(std::uint32_t(facts.barbarian_tribe_id));scalar(facts.city_site_grade);\n",
         "    scalar(std::uint32_t(facts.barbarian_tribe_id));scalar(facts.city_site_grade);\n"
         "    for(std::size_t at=wonder_begin;at<wonder_begin+sizeof(facts.natural_wonder);++at)byte(input[at]);\n"
         "    scalar(facts.natural_wonder_flags);\n")],
    "Renderer/native/terrain_scene_runtime.cpp": [
        ("        hash_bytes(result.geometry, tile.resource_name, sizeof(tile.resource_name));\n",
         "        hash_bytes(result.geometry, tile.resource_name, sizeof(tile.resource_name));\n"
         "        // A natural wonder's bodies and name label.\n"
         "        hash_bytes(result.scene, tile.natural_wonder, sizeof(tile.natural_wonder));\n"
         "        hash_bytes(result.geometry, tile.natural_wonder, sizeof(tile.natural_wonder));\n"
         "        hash_value(result.scene, tile.natural_wonder_flags);\n"
         "        hash_value(result.geometry, tile.natural_wonder_flags);\n")],
    "Renderer/native/test_native_bridge_contract.py": [
        ('            "city_era", "city_flags",\n        ):',
         '            "city_era", "city_flags", "natural_wonder", "natural_wonder_flags",\n        ):')],
    "Renderer/native/render_core/world_key_diagnostic.h": [
        ('            {"barbarian",offsetof(c3x_renderer_tile_v1,barbarian_tribe_id),4}};',
         '            {"barbarian",offsetof(c3x_renderer_tile_v1,barbarian_tribe_id),4},\n'
         '            {"natural_wonder_digest",offsetof(c3x_renderer_tile_v1,natural_wonder),sizeof(before.natural_wonder)},\n'
         '            {"natural_wonder_flags",offsetof(c3x_renderer_tile_v1,natural_wonder_flags),4}};')],
    "Renderer/native/render_core/captured_scene.h": [
        ("            std::memset(next.occurrence.resource_name,0,sizeof(next.occurrence.resource_name));\n",
         "            std::memset(next.occurrence.resource_name,0,sizeof(next.occurrence.resource_name));\n"
         "            std::memset(next.occurrence.natural_wonder,0,sizeof(next.occurrence.natural_wonder));\n"
         "            next.occurrence.natural_wonder_flags=0;\n")],
    "Renderer/native/test_input_recording.cpp": [
        ("tile.barbarian_tribe_id=18;tile.city_site_grade=11;\n",
         "tile.barbarian_tribe_id=18;tile.city_site_grade=11;\n"
         "    std::memcpy(tile.natural_wonder,\"Old Faithful\",13);tile.natural_wonder_flags=C3X_RENDERER_NATURAL_WONDER_NAME;\n"),
        ('    require(restored.tiles[0].city_site_grade==11,"city-site presentation grade lost");\n',
         '    require(restored.tiles[0].city_site_grade==11,"city-site presentation grade lost");\n'
         '    require(!std::strcmp(restored.tiles[0].natural_wonder,"Old Faithful") &&\n'
         '            restored.tiles[0].natural_wonder_flags==C3X_RENDERER_NATURAL_WONDER_NAME,"natural wonder facts lost");\n')],
}
