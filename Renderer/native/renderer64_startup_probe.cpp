// Reproduce the game's Renderer64 bootstrap without launching Civ III.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdio>
#include <cstring>

#include "c3x_renderer_api.h"

int main(int argc, char** argv) {
    bool late_cache = argc == 4 && std::strcmp(argv[3], "--late-cache") == 0;
    if (argc != 3 && !late_cache) {
        std::fprintf(stderr, "usage: renderer64_startup_probe <mod-relative-dir> <bridge-dll> [--late-cache]\n");
        return 2;
    }
    if (late_cache) {
        SetEnvironmentVariableA("C3X_RENDERER_WORLD_RASTER_GRID", "0");
        SetEnvironmentVariableA("C3X_RENDERER_WORLD_REGIONS", "0");
    }
    HMODULE bridge = LoadLibraryA(argv[2]);
    if (!bridge) {
        std::fprintf(stderr, "FAIL bridge-load win32=%lu\n", GetLastError());
        return 1;
    }
    auto lifetime = reinterpret_cast<c3x_renderer_native_lifetime_fn>(
        GetProcAddress(bridge, "c3x_renderer_native_lifetime"));
    auto mode = reinterpret_cast<int (*)(int)>(GetProcAddress(bridge, "c3x_renderer_set_backend_mode"));
    auto healthy = reinterpret_cast<int (*)()>(GetProcAddress(bridge, "c3x_renderer_backend_healthy"));
    auto version = reinterpret_cast<c3x_renderer_get_api_version_fn>(
        GetProcAddress(bridge, "c3x_renderer_get_api_version"));
    auto image = reinterpret_cast<c3x_renderer_native_image_fn>(
        GetProcAddress(bridge, "c3x_renderer_native_image"));
    auto definitions = reinterpret_cast<c3x_renderer_set_definition_paths_fn>(
        GetProcAddress(bridge, "c3x_renderer_set_definition_paths"));
    auto reset = reinterpret_cast<c3x_renderer_reset_fn>(GetProcAddress(bridge, "c3x_renderer_reset"));
    if (!lifetime || !mode || !healthy || !version || !image || !definitions || !reset ||
        version() != C3X_RENDERER_API_VERSION) {
        std::fprintf(stderr, "FAIL bridge-api\n");
        FreeLibrary(bridge);
        return 1;
    }
    lifetime(C3X_NATIVE_VERIFY, nullptr, 0);
    int selected = mode(1);
    if (selected != C3X_RENDERER_RESULT_OK) {
        std::fprintf(stderr, "FAIL renderer64-select result=%d\n", selected);
        FreeLibrary(bridge);
        return 1;
    }
    image(C3X_NATIVE_VISUAL_POLICY, nullptr, nullptr, nullptr, nullptr, 1);
    if (!healthy() || mode(1) != C3X_RENDERER_RESULT_OK) {
        std::fprintf(stderr, "FAIL renderer64-early-screen-reselection\n");
        reset();
        FreeLibrary(bridge);
        return 1;
    }
    if (late_cache) {
        SetEnvironmentVariableA("C3X_RENDERER_VISUAL_PROFILE", "city-fidelity");
        SetEnvironmentVariableA("C3X_RENDERER_WORLD_RASTER_GRID", "1");
        SetEnvironmentVariableA("C3X_RENDERER_WORLD_REGIONS", "1");
    }
    char defaults[2 * MAX_PATH], custom[2 * MAX_PATH];
    if (std::snprintf(defaults, sizeof defaults, "%s\\Renderer\\default.custom_rendering.txt", argv[1]) >= int(sizeof defaults) ||
        std::snprintf(custom, sizeof custom, "%s\\Renderer\\custom.custom_rendering.txt", argv[1]) >= int(sizeof custom)) {
        std::fprintf(stderr, "FAIL definition-path-length\n");
        FreeLibrary(bridge);
        return 1;
    }
    if (GetFileAttributesA(custom) == INVALID_FILE_ATTRIBUTES) custom[0] = '\0';
    int configured = definitions(argv[1], defaults, nullptr, custom);
    int alive = healthy();
    std::printf("renderer64-bootstrap select=%d definitions=%d healthy=%d\n", selected, configured, alive);
    reset();
    FreeLibrary(bridge);
    return configured == C3X_RENDERER_RESULT_OK && alive ? 0 : 1;
}
