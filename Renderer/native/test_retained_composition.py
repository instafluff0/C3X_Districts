import unittest
from Renderer.native.native_cpp_test import run_cpp

# One test function holds every case; its frame exceeds the default 1 MB
# stack (x86 access violation, x64 stack overflow before the first case), so
# it runs on a thread with a 64 MB stack reservation.
ENTRY = r'''
#include <windows.h>
int test_retained_composition();
DWORD WINAPI retained_composition_body(void*){return DWORD(test_retained_composition());}
int main(){
    HANDLE thread=CreateThread(nullptr,64u<<20,retained_composition_body,nullptr,STACK_SIZE_PARAM_IS_A_RESERVATION,nullptr);
    if(!thread)return 2;
    WaitForSingleObject(thread,INFINITE);DWORD code=1;GetExitCodeThread(thread,&code);CloseHandle(thread);
    return int(code);
}
'''

class RetainedCompositionTests(unittest.TestCase):
    def test_independent_visual_frames_and_native_versions(self):
        run_cpp(ENTRY, sources=('Renderer/native/test_retained_composition.cpp',), timeout=180)
