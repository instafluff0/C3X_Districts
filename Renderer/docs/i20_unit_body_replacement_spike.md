# I20 Unit Body Replacement Spike

Status: read-only boundary audit and offline design complete for the installed
GOG executable; no runtime patch is enabled and no CSV entry is requested.

## Preferred boundary

The smallest safe design is a scoped `Unit::tick_anim` context plus guarded
interception of the two Sprite body-blit routines. An entry wrapper around
`Unit::tick_anim` supplies the authoritative `Unit`, display-civilization
choice, action, facing, progress, anchor, and Army member ID. While that wrapper
is active, interceptors for the normal and reduced Sprite blits replace only
the unit body. Every invocation outside that context passes directly to the
original Sprite routine.

This is preferable to patching six individual call instructions: the Sprite
ABI has no `Unit *`, while the ordinary and Army callers keep their Unit in
different registers. Per-call assembly shims would therefore be
build-specific. It is also preferable to replacing all of `Unit::tick_anim`,
which would duplicate its visibility, selection-underlay, hidden-nationality,
city, palette, and retained-HUD behavior.

Replacement is fail-closed per semantic unit. Before the first body call, the
context resolves every required model, component, material, socket, pose cache,
owner-color row, and output surface. If preflight fails, all body calls remain
native. If it succeeds, the first intercepted body call inserts one complete
custom body and the original body call is omitted. For an Army, the same
preflight covers both commander and `field_1B0[1]` displayed member; both
native body calls are omitted together and the custom pair is inserted once.
The later native HUD call remains untouched. A compound unit likewise succeeds
or falls back as one body regardless of node count.

The scoped context is cleared on every exit from `Unit::tick_anim`. It is not a
second animator: Civ III's invisible FLC still advances, chooses actions,
directions, anchors, completion, and Army membership.

## Installed GOG evidence

Read-only `dumpbin /disasm` evidence was taken from the installed GOG
`Civ3Conquests.exe`, SHA-256
`c85aa82062a0249ce87cafe31d4ed94d98b4f4ef5727939d7311d289eae52710`.

| Responsibility | GOG address | Instruction / target |
| --- | ---: | --- |
| `Unit::tick_anim` entry | `0x005CBF50` | context candidate |
| ordinary normal body | `0x005CC33B` | `E8 70 C5 02 00`, calls `Sprite::FUN_005f88b0` (`0x005F88B0`) |
| ordinary reduced body | `0x005CC3A1` | `E8 9A C5 02 00`, calls `Sprite::FUN_005f8940` (`0x005F8940`) |
| ordinary retained HUD | `0x005CC41D` | calls `FUN_005ba750` after the body branch |
| Army helper entry | `0x005CC430` | two-body native helper |
| Army normal commander | `0x005CC861` | calls `0x005F88B0` |
| Army normal member | `0x005CC8BC` | calls `0x005F88B0` |
| Army reduced commander | `0x005CC926` | calls `0x005F8940` |
| Army reduced member | `0x005CC98D` | calls `0x005F8940` |
| Army retained HUD | `0x005CC9EB` | calls `FUN_005ba750` after both body calls |

The ordinary selected-unit underlay uses a different routine
(`FUN_005f84b0`) before the body branch. Therefore the proposed two guarded
Sprite hooks neither suppress that underlay nor the later health/activity/
stack work. The Army proof also confirms the normal/reduced 40/20-pixel member
placement described in `army_rendering_strategy.md` occurs inside the body
helper, before its single HUD call.

## Patch dependency state

Likely future dependencies are `Unit_tick_anim`,
`Sprite_draw_unit_body_normal` (`Sprite::FUN_005f88b0`), and
`Sprite_draw_unit_body_reduced` (`Sprite::FUN_005f8940`), each as an `inlead`.
Their C-level signatures still need to be frozen against actual calling
conventions, especially the reduced routine's decompiler-damaged prototype.
The GOG addresses above are confirmed, but Steam and PCGames addresses and
bytes are not. Under the renderer patch-ledger rules these remain audit
candidates, not a user action or CSV request.

I20 should request them only after all three supported builds are exact and an
executable native smoke fixture proves: ordinary custom success, ordinary
fallback, Army two-body success, Army two-body fallback, unrelated Sprite
pass-through, normal/reduced zoom, and retained underlay/HUD ordering.
