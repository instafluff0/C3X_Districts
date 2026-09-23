# Direct x64 composition-surface trial

The current opt-in x64 renderer produces a completed shared BGRA image for each
accepted map or ambient frame. The x86 process imports it, waits on the keyed
mutex, copies it into its own composition swap chain and calls `Present`. No
normal frame crosses through CPU pixels, but every accepted visual frame incurs
a control reply, resource handoff, GPU synchronization and x86 display work.

An isolated Windows 11 VM proof tested a more direct route. The x86 window
owner created a DirectComposition surface handle and attached its wrapper to
its own HWND. It duplicated that handle into an x64 process once. The x64
process created a BGRA flip swap chain for the handle and called `Present`
without sending image handles or frame-ready messages to x86. The x86 window
thread did not pump messages during ten timed color changes. The desktop pixel
changed nine times during the observation window, at both 320×240 and
1120×1192. The ten-frame producer interval was about 825–830 ms because the
probe deliberately slept 75 ms between frames; it is not a production FPS
measurement. The x86 child-window blue pixel stayed visible above the map.
The existing production `NativePresenter` Gate 1 desktop witness passed on
the same VM, so the direct trial was compared against a working presenter.

The trial also showed the critical UI issue. A parent-HWND GDI pixel was green
before attachment, then covered by the composition map. That agrees with
[Microsoft's HWND target layer order](https://learn.microsoft.com/en-us/windows/win32/api/dcomp/nf-dcomp-idcompositiondevice-createtargetforhwnd):
DirectComposition content is above direct-HDC drawing on the target window;
child windows can remain above a non-topmost visual. Civ III draws more than
child-window UI. Direct x64 presentation must preserve labels, HUD, native
overlays, partial screen transfers and config-off/GDI restoration through an
exact upper native composition plane or another proven equivalent. It cannot
simply attach the x64 map and leave native UI to draw behind it.

The SDK's [composition-surface-handle swap-chain method](https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_3/nf-dxgi1_3-idxgifactorymedia-createswapchainforcompositionsurfacehandle)
is documented as a YUV method, although this VM accepted `DXGI_FORMAT_B8G8R8A8_UNORM`
and displayed the resulting BGRA frames. That behavior requires explicit driver
and OS qualification and a retained fallback. A separate raw shared-texture
experiment imported exact red GPU pixels and returned success from composition
texture creation and `Commit`, but displayed black, including with a local
shared texture; it was discarded. The test code retains only the successful
surface path.

The next production experiment should create one surface per window generation,
let x64 render its real final image directly into that swap chain, and deliver
ambient frames using its own clock. x86 sends only authoritative state/camera/
native-overlay changes and lifecycle commands; it does not request every visual
frame. Add an exact upper native UI plane and retain the current x86 presenter
as the fallback. Full native-interleaved replay must prove pixels, order,
partial transfers, resize, helper restart, device loss and the displayed
first-correct frame before any cutover. Compare all-effects-on idle and Standard
map navigation to the current x64 path; the color-swap proof alone establishes
neither speedup nor correctness for Civ III.
