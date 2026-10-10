/* Harmless stand-in for Civ III: exercise the real launch/elevation path and
   the staged DLL's inherited diagnostics, without starting the game or
   allocating GPU resources. 7038b8e2 removed the launcher's composition
   journal request, so only the runtime trace setting is required here. */
#include <windows.h>
#include <stdio.h>
#include <string.h>

int main(void) {
#ifdef C3X_CAPTURE_SKIP_RECORDING
    return 0;
#else
    char trace[32768], dll_path[32768];
    HMODULE dll;
    FILE *file;
    int (*native_image)(int, void *, void *, void const *, void const *, unsigned);
    if (!GetEnvironmentVariableA("C3X_RENDERER_TRACE_FILE", trace, sizeof(trace))) return 2;
    if (!GetModuleFileNameA(NULL, dll_path, sizeof(dll_path))) return 3;
    *strrchr(dll_path, '\\') = '\0';
    strcat(dll_path, "\\..\\..\\..\\bin\\C3XRenderer.dll");
    dll = LoadLibraryA(dll_path);
    if (!dll) return 4;
    native_image = (void *)GetProcAddress(dll, "c3x_renderer_native_image");
    if (!native_image) return 5;
    native_image(0, NULL, NULL, NULL, NULL, 0);
    FreeLibrary(dll);
    file = fopen(trace, "rb");
    if (!file) return 8;
    fclose(file);
    return 0;
#endif
}
