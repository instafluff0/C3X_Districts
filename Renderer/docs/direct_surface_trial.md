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
child windows can remain above a non-topmost visual. The existing x64
final-image compositor already receives some native image operations, so those
pixels can be inside the direct surface. Any labels, HUD or overlays drawn
directly to the HWND *after* that composition would be hidden. The production
experiment must inventory those writes and preserve their order through the
final image, an upper native plane, or an exact ownership handoff. Partial
screen transfers and config-off/GDI restoration remain required.

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
frame. Preserve native UI operations in the final image where they already
exist; add an upper plane only for remaining same-HWND draws. Retain the current
x86 presenter as fallback. Full native-interleaved replay must prove pixels, order,
partial transfers, resize, helper restart, device loss and the displayed
first-correct frame before any cutover. Compare all-effects-on idle and Standard
map navigation to the current x64 path; the color-swap proof alone establishes
neither speedup nor correctness for Civ III.

## Real-scene opt-in checkpoint

`C3X_RENDERER_DIRECT_SURFACE_TRIAL=1` now routes the x64 helper's retained
final-image composer to a surface swap chain on the renderer's own D3D device.
The x86 bridge owns the HWND, creates/attaches the surface, and duplicates its
handle once per window generation. The helper can present without exporting a
shared image back to x86. The established shared-texture presenter remains the
default and is selected if surface setup or presentation fails. A trial-only
readback supports exact replay fingerprints; it is not used for normal frames.
Direct-surface activation waits until the first completed image so setup does
not cover the previous native display with a blank surface. Returning to a
native partial transfer seeds the x86 presenter from the last completed x64
image once, preserving pixels outside the transfer rectangle.

The first paired `continuous-presentation-final-native` replay completed
9,433 calls per run, but it did **not** compare the routes: replay settings
cleared the environment opt-in before the renderer loaded. Its apparent
50.4-to-29.1 ms map-presentation p95 difference and 337 matching fingerprints
were shared-path versus shared-path observations and are discarded as evidence
for direct presentation. A targeted blocked-window roundtrip did enter the
direct path and exposed an argument gate that rejected x64 visual frames;
the corresponding shared-path control advanced 12. That gate is fixed in the
trial DLL. Replay now has an explicit `--direct-surface-trial` option, restored
after each settings event, and strict route selection that fails rather than
silently falling back. Repeat its performance and pixel comparison before
claiming any real-scene speedup or parity.

The direct-surface candidate now schedules its own ambient frames in Renderer64.
The x86 bridge sends state, camera, presentation ownership and lifecycle changes;
it does not send each direct-surface visual offer. The focused real-scene fixture
held the x86 window thread for 450 ms without issuing a visual request. The
displayed surface changed during that interval, with four visible animations in
the scene. This proves the independent clock in that controlled state. The
shared-texture control still uses x86 visual offers. The trial removes the
per-frame shared-handle import, keyed wait, x86 GPU copy and x86 `Present` on
its direct route, but does not establish that every same-HWND Civ III UI write
is above the surface. Keep the opt-in route out of the installed game until
native UI ordering, interturn, resize, restart, config-off and device loss pass.
There is no valid full-workload speed comparison yet.
