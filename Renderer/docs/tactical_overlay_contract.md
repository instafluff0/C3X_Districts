# Tactical overlay contract

Civ III owns selection, pathfinding, turns, action side effects, mouse picking and
the global grid setting. The DLL owns only the copied draw semantics. The user's
`selected_unit.mov` and `pathfinder.png` supply the visual reference: a broken
white ellipse with rotating inward markers, red route, destination ellipse and
turn label. Source media remain external/local; runtime art is generic.

There is one current selected unit and one active go-to preview for that unit.
The destination ring/turn label belongs to that preview, not another selection.
Older copied commands may survive only as dependencies of native partial display
history; they do not create additional authoritative selections or routes.

The existing two audited map-unit cursor call sites send the selected anchor and
projection to the admitted native composition owner. Native eligibility remains
`Animator + 0x1914` bit 0. The existing visual clock rotates markers; it does not
advance gameplay or the native action director. Native erase/copy/partial display
operations remove or retain these commands in the same order as unit bodies/UI.

The selected marker and completed route batch use the native unit canvas as
their destination and `Main_Screen_Form::Base_Data.Canvas` as their background,
matching the native cursor functions. The unit canvas can contain keyed transparent
pixels: antialiased coverage resolves against that separate map background, not
against transparency. Both background identity and full-color detail are copied
through the existing native operation; no game pointer reaches the GPU worker.

The GOG go-to update wrapper opens a bounded capture scope and executes the
original function. Only that scope's target image intercepts native line/text
operations. No route is recomputed, guessed from terrain, or retained through a
new mouse listener. The separate destination-cursor hook supplies its native
center. The complete native turn string is copied, including an action suffix;
font/color shadow draws collapse into one modern label. At scope close one
immutable primitive batch joins the existing worker and retained history.

The native grid function returns immediately in custom mode; the map insertion
boundary passes the current `MapGrid_Flag` with the captured projected tile
anchors. Each revealed/explored tile contributes its top two diamond edges once.
Unseen tiles contribute no geometry. A native map copy removes the previous grid
before the new setting is applied. No Ctrl+G listener or independent toggle exists.

One instanced GPU pass uses analytic antialias coverage, premultiplied color and
restrained shadows. A generic Segoe UI glyph atlas is prepared once per device.
There is no terrain/unit mesh input or readback in tactical draws. The temporary
RGBA and packed-native scratch attachments are each bounded by 2240×1192; input is capped at 16,384 primitives and
32 label characters. Copied primitive capacity is charged to the existing retained
128 MiB budget, including old partial-display histories. Static routes/grid have constant revisions; only selected
markers request independent samples. Existing retained-composition clipping,
partial copies, native pixel formats, UI order and lifetime limits apply.

The optional native capability query preserves existing cursor/route drawing with an older DLL or before native
composition admission. Once admitted, failed draws
must not publish partial native pixels. Config-off follows original functions.
This checkpoint uses coherent synchronous map publication. M3 still owns the
broader atomic camera/visibility/overlay publication contract.

The `tactical-overlays` Lab category uses the actual connected JGL fixture for
its route/grid previews. Synthetic semantic inputs are explicitly distinguished
from live-game evidence. Visual acceptance and the strategic live-game check are
pending; automated tests do not grant either.
