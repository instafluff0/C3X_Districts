# Day/night

One authoritative C3X clock, normalized sun/moon basis, ambient light, exposure and emission response across the scene.

The clock is the hour, plus minutes in the Renderer64-only `continuous` and
`user-time` modes. C3X advances them on Civ III's timer. Each frame (API 22)
carries `minute`, the QPC `clock_qpc` at which that minute began, and
`clock_rate`. C3X anchors at the hour's start, so every frame within an hour is
identical; per-minute anchors made a retried capture never converge at
1440× (Civ III redrew the map every timer tick). Renderer64 advances whole game minutes from those
(`clock_hour`), so C3X requests a map capture only for a new anchor, hour or
season: a per-minute capture plus the change it carried cost 8–10 presented
frames (the capture alone about 2–3, native image-batch prefix). A new time is adopted only after the previous change has fully
refined and the camera has been still for 0.4 s (`adopt_clock_time`). A running
clock also relights at most every 10 real seconds (`clock_step_seconds`),
adopting the latest minute; static modes and a season change apply at once. Its static re-raster (~105 ms of Parallels GPU for a
whole view) refines behind the displayed preview in 16 slices
(`lighting_refine_budget`); the mirror (~30 ms) and shadow pages (~13 ms)
still refresh in the adopting frame. Measured 2026-10-23 on the VM (user
3350 BC save, scroll scenario): each change costs about 5–7 of 60 presented
frames. With the throttle, 20× idles at 58.9–59.4 fps (control 60.0) and
1440× at 58.4 (one 4-hour step per 10 s; Civ III CPU 6% vs 2–3%). Trace lines `lighting-adopt` and `lighting-settled`
record each change's wait and cost; `Renderer/tools/day_night_cost_report.py`
summarises a capture.

`standard.json` identifies the shared implementation, dependencies, fixture recipe
and focused regression tests. The current checkout is authoritative.
Reference captures reproduce production using small synthetic diagnostic scenes;
they are not claimed to be live Civ III captures. References are optional review
aids. See `Renderer/docs/visual_fidelity_playbook.md`.
