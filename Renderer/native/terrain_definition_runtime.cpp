#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "terrain_definition_runtime.h"

namespace c3x_renderer {
namespace {

struct Definition {
    std::string type;
    std::string id;
    std::map<std::string, std::string> values;
    std::string source_path;
    int layer = 0;
    int order = 0;
};

std::string trim(std::string value) {
    auto not_space = [](unsigned char ch) { return std::isspace(ch) == 0; };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

bool read_text(char const * path, std::string & text) {
    if (path == nullptr || path[0] == '\0')
        return false;
    HANDLE file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE)
        return false;
    LARGE_INTEGER size = {};
    bool ok = GetFileSizeEx(file, &size) != 0 && size.QuadPart > 0 && size.QuadPart <= 4ll * 1024ll * 1024ll;
    if (ok) {
        text.resize(static_cast<std::size_t>(size.QuadPart));
        DWORD read = 0;
        ok = ReadFile(file, text.data(), static_cast<DWORD>(text.size()), &read, nullptr) != 0 &&
             read == static_cast<DWORD>(text.size());
    }
    CloseHandle(file);
    if (!ok)
        text.clear();
    return ok;
}

void finish_definition(Definition & current, std::vector<Definition> & output) {
    auto found = current.values.find("id");
    if (!current.type.empty() && found != current.values.end() && !found->second.empty()) {
        current.id = found->second;
        output.push_back(current);
    }
    current = {};
}

bool parse_layer(char const * path, int layer, bool required,
                 std::vector<Definition> & output, std::string & diagnostic) {
    std::string text;
    if (!read_text(path, text)) {
        if (!required)
            return true;
        diagnostic = "required renderer definition file is unavailable";
        return false;
    }
    Definition current;
    current.source_path = path;
    current.layer = layer;
    int line_number = 0;
    int order = 0;
    std::size_t cursor = 0;
    while (cursor <= text.size()) {
        std::size_t end = text.find('\n', cursor);
        std::string line = trim(text.substr(cursor, end == std::string::npos ? std::string::npos : end - cursor));
        ++line_number;
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (!line.empty() && line.front() == '#') {
            finish_definition(current, output);
            current.type = trim(line.substr(1));
            current.source_path = path;
            current.layer = layer;
            current.order = order++;
        } else if (!line.empty() && line.front() != '[') {
            std::size_t equals = line.find('=');
            if (equals == std::string::npos || current.type.empty()) {
                diagnostic = "malformed renderer definition at line " + std::to_string(line_number);
                return false;
            }
            std::string key = trim(line.substr(0, equals));
            std::string value = trim(line.substr(equals + 1));
            if (key.empty() || value.empty() || current.values.count(key) != 0) {
                diagnostic = "invalid or duplicate renderer definition key at line " + std::to_string(line_number);
                return false;
            }
            current.values[key] = value;
        }
        if (end == std::string::npos)
            break;
        cursor = end + 1;
    }
    finish_definition(current, output);
    return true;
}

bool disabled(Definition const & value) {
    auto found = value.values.find("disabled");
    return found != value.values.end() && lower(found->second) == "true";
}

std::string parent_path(std::string const & path) {
    std::size_t slash = path.find_last_of("\\/");
    return slash == std::string::npos ? "." : path.substr(0, slash);
}

bool safe_relative(std::string const & path) {
    if (path.empty() || path.front() == '\\' || path.front() == '/' || path.find(':') != std::string::npos)
        return false;
    std::string normalized = path;
    std::replace(normalized.begin(), normalized.end(), '\\', '/');
    std::size_t cursor = 0;
    while (cursor <= normalized.size()) {
        std::size_t end = normalized.find('/', cursor);
        std::string part = normalized.substr(cursor, end == std::string::npos ? std::string::npos : end - cursor);
        if (part == "..")
            return false;
        if (end == std::string::npos)
            break;
        cursor = end + 1;
    }
    return true;
}

std::string join_path(std::string root, std::string relative) {
    if (!root.empty() && root.back() != '\\' && root.back() != '/')
        root.push_back('\\');
    std::replace(relative.begin(), relative.end(), '/', '\\');
    return root + relative;
}

bool resolve_pack_root(Definition const & definition, char const * mod_root, std::string & result) {
    auto found = definition.values.find("path");
    if (found == definition.values.end())
        return false;
    std::string const & raw = found->second;
    std::string prefix;
    std::string relative;
    std::size_t colon = raw.find(':');
    if (colon == std::string::npos)
        return false;
    prefix = lower(raw.substr(0, colon));
    relative = raw.substr(colon + 1);
    if (!safe_relative(relative))
        return false;
    if (prefix == "mod")
        result = join_path(mod_root == nullptr ? "" : mod_root, relative);
    else if (prefix == "scenario")
        result = join_path(parent_path(definition.source_path), relative);
    else
        return false;
    return true;
}

int terrain_index(std::string name) {
    name = lower(trim(name));
    std::replace(name.begin(), name.end(), '_', '-');
    static char const * names[terrain_type_count] = {
        "desert", "plains", "grassland", "tundra", "flood-plain", "hills", "mountains",
        "forest", "jungle", "marsh", "volcano", "coast", "sea", "ocean"
    };
    for (int index = 0; index < terrain_type_count; ++index)
        if (name == names[index])
            return index;
    return -1;
}

int integer_value(Definition const & value, char const * key, int fallback) {
    auto found = value.values.find(key);
    if (found == value.values.end())
        return fallback;
    char * end = nullptr;
    long parsed = std::strtol(found->second.c_str(), &end, 10);
    return end != found->second.c_str() && *end == '\0' ? static_cast<int>(parsed) : fallback;
}

} // namespace

bool load_terrain_definition_layers(
    char const * mod_root,
    char const * default_path,
    char const * scenario_path,
    char const * custom_path,
    std::array<TerrainAssetBinding, terrain_type_count> & bindings,
    RendererPackRoots & companion_packs,
    std::string & diagnostic) {
    bindings = {};
    companion_packs = {};
    diagnostic.clear();
    std::vector<Definition> parsed;
    if (!parse_layer(default_path, 0, true, parsed, diagnostic) ||
        !parse_layer(scenario_path, 1, false, parsed, diagnostic) ||
        !parse_layer(custom_path, 2, false, parsed, diagnostic))
        return false;

    std::map<std::string, Definition> merged;
    for (Definition const & value : parsed) {
        std::string key = value.type + "\n" + value.id;
        if (disabled(value))
            merged.erase(key);
        else
            merged[key] = value;
    }

    std::map<std::string, std::string> packs;
    for (auto const & pair : merged) {
        Definition const & value = pair.second;
        if (value.type != "Pack")
            continue;
        std::string root;
        if (resolve_pack_root(value, mod_root, root))
            packs[value.id] = root;
    }
    auto companion = [&packs](char const * id) {
        auto found = packs.find(id);
        return found == packs.end() ? std::string() : found->second;
    };
    companion_packs.vegetation = companion("vegetation_normalized");
    companion_packs.decals = companion("decals_normalized");
    companion_packs.terrain_elements = companion("terrain_elements_normalized");
    companion_packs.shore = companion("shore_normalized");

    struct Asset { std::string root; std::string logical; };
    std::map<std::string, Asset> assets;
    for (auto const & pair : merged) {
        Definition const & value = pair.second;
        if (value.type != "Asset")
            continue;
        auto pack = value.values.find("pack");
        auto logical = value.values.find("asset");
        if (pack != value.values.end() && logical != value.values.end() &&
            packs.count(pack->second) != 0 && safe_relative(logical->second))
            assets[value.id] = {packs[pack->second], logical->second};
    }

    struct Winner { int priority = -2147483647; int layer = -1; int order = -1; std::string asset; };
    std::array<Winner, terrain_type_count> winners = {};
    for (auto const & pair : merged) {
        Definition const & value = pair.second;
        auto category = value.values.find("category");
        auto terrain = value.values.find("terrain_type");
        auto asset = value.values.find("asset");
        auto replacement = value.values.find("replacement");
        if (value.type != "Rule" || category == value.values.end() || lower(category->second) != "terrain" ||
            terrain == value.values.end() || asset == value.values.end() ||
            (replacement != value.values.end() && lower(replacement->second) != "replace"))
            continue;
        int index = terrain_index(terrain->second);
        int priority = integer_value(value, "priority", 0);
        if (index >= 0 && assets.count(asset->second) != 0) {
            Winner & winner = winners[index];
            if (priority > winner.priority ||
                (priority == winner.priority && (value.layer > winner.layer ||
                 (value.layer == winner.layer && value.order > winner.order))))
                winner = {priority, value.layer, value.order, asset->second};
        }
    }

    for (int index = 0; index < terrain_type_count; ++index) {
        if (winners[index].layer < 0)
            continue;
        Asset const & asset = assets[winners[index].asset];
        bindings[index] = {true, asset.root, asset.logical};
    }
    return true;
}

} // namespace c3x_renderer
