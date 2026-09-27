// Use the existing installer with console diagnostics and the same exit codes.
// Include Windows declarations before redirecting ep.c's two message boxes.
#define NOVIRTUALKEYCODES
#include <windows.h>
#include <stdio.h>

int WINAPI
renderer_install_message (HWND window, LPCSTR message, LPCSTR title, UINT flags)
{
    fprintf ((flags & MB_ICONERROR) ? stderr : stdout, "%s%s%s\n",
        title ? title : "", title ? ": " : "", message);
    fflush ((flags & MB_ICONERROR) ? stderr : stdout);
    return IDOK;
}

#undef MessageBox
#define MessageBox renderer_install_message
#include "../../ep.c"
