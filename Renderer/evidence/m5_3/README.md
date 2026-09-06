# M5.3 Frame-Scheduling Evidence

## Audited Civ III Boundary

`on_timer_0x9F6500` is already an injected patch target in `civ_prog_objects.csv`; no new address entry is required. The decompiled function is armed by `Timer::reset_and_activate` with `0x42` millisecond delay and interval values. On its guarded UI-thread path it samples `QueryPerformanceCounter`, performs Civ III timer work, and calls `Animator::refresh` on the main-screen animator.

The M5.3 hook runs immediately before the original function. It asks the native pure scheduler for a decision and, at most, marks the existing animator dirty. It never calls renderer capture, D3D render, readback, blit, a form draw method, `Sleep`, a message pump, or a catch-up loop. The original Civ III timer remains responsible for processing the normal redraw path.

## Runtime Contract

- One `QueryPerformanceCounter` timestamp is captured at the start of each normal map pass and copied unchanged into the frame ABI.
- The scheduler derives event phase from absolute timestamp and event epoch, never from rendered-frame count.
- A single boolean records whether one redraw is already pending, so delayed timers cannot create a queue.
- Static scenes report zero visible renderer-owned animations and request no continuous redraws.
- Loading, non-map, unfocused, modal, nested-draw, pending-redraw, and large/backward-clock states suppress or rebase scheduling.
- Scalar saturating counters and maximum QPC durations cover requested, presented, skipped, capture, native render/readback, blit, and total map-pass work without retaining frame histories.
- M5.3 deliberately enables no animated ownership. M7.4/M7.5 will supply visible renderer-owned animation counts only after their category-specific fallback and lifecycle gates exist.

## Automated Evidence

`Renderer/native/native_smoke.cpp` proves static idle behavior, one bounded redraw after elapsed cadence, skipped-frame accounting, byte-identical decisions, identical absolute-time phase despite different prior frame times, focus/visibility/modal/drawing/pending suppression, and large-pause rebasing.

`Renderer/native/test_native_bridge_contract.py` proves the timer hook only calls the pure scheduler and marks Civ III's animator dirty; it rejects renderer calls, form draws, `Sleep`, threads, swap chains, presenters, and loops in the scheduling path.

`TEST_INJECTED_CODE_COMPILE.bat` and the registered `m5_3_frame_scheduler` verification gate are the executable integration checks.
