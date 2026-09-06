# C3X Renderer Visual Validation Plan

## Purpose

Visual validation answers two separate questions:

1. Is the renderer structurally correct and deterministic?
2. Does the result look coherent and sufficiently close to its intended art direction?

Deterministic tests decide the first question. Human and AI-assisted review help answer the second. Qualitative review never replaces structural gates.

## Scene Fixture Pipeline

The canonical renderer input is a versioned `.scene.json` fixture conforming to the visible-scene contract. A save or BIQ is not rendered directly by the standalone renderer.

The preferred initial pipeline is game-assisted:

```text
Civ III save or BIQ
        |
        v
Civ III loads authoritative scenario/map state
        |
        v
C3X exports selected viewport(s) as .scene.json
        |
        v
standalone resolver + renderer
        |
        v
PNG images, metrics, and contact sheet
```

This avoids maintaining an independent parser for every save/BIQ field before the in-game state capture exists. A direct offline save/BIQ reader may be added later, but it must produce the same scene schema and pass equivalence tests against game-assisted exports.

Scene export must support named camera fixtures, map coordinates, viewport size, zoom, hour, season, player/visibility context, and deterministic world seed. Private or copyrighted save data remains local unless deliberately contributed as a redistributable fixture.

The first representative integration fixture should be a deliberately populated save or BIQ containing, within one or more named viewports:

- Several terrain types, coasts/water, relief, transitions, and vegetation.
- At least two city sizes/culture groups where available.
- Roads, rivers, irrigation, mines, and other enabled improvements.
- Resources, units of different owners/directions/actions, borders, fog, labels, and selection/highlight state.

M5.2 automatically exports a populated Civ III viewport to `.scene.json`; strict synthetic scenes and batch-rendered PNGs prove schema completeness, determinism, anchor alignment, composition, and coverage. An ordinary full-screen capture and the user's game-health report confirm that the map remains beneath Civ III-owned fog, borders, labels, highlights, HUD, and UI. Formal named cameras and matched image pairs are optional during routine development and can be promoted when diagnosing a regression or preparing a release review.

If a BIQ contains only a designed map and rules, a paired save made from that BIQ supplies runtime-only state such as unit actions, ownership, visibility, city state, and the active viewport. The BIQ and save should share named camera fixtures so their exports can be compared.

## Batch Render Command Goal

Conceptually:

```powershell
py Renderer\tools\render_fixture_matrix.py `
  --scene Renderer\samples\scenes\reference_map.scene.json `
  --pack Renderer\packs\Civ6Conquests `
  --viewports 640x480,1024x768 `
  --hours 0,6,12,18 `
  --seasons summer,fall,winter,spring `
  --output Renderer\validation\reference_map
```

Outputs include individual PNGs, a contact sheet, machine-readable metrics, unresolved mapping diagnostics, and a manifest recording exact inputs and versions.

## Deterministic Gates

These checks must pass before qualitative review:

- Scene schema and all asset references validate.
- Output is nonblank and confined to the map viewport.
- Repeated renders from identical inputs are byte-stable or meet a documented deterministic pixel tolerance.
- Authoritative Civ III anchors project to expected pixels.
- Required object categories have acceptable mapping coverage.
- No NaN/invalid depth, out-of-bounds writes, or unexplained missing geometry occurs.
- Terrain seams, wrap boundaries, and viewport edges satisfy defined pixel/geometry invariants.
- Object overlap follows depth and layer ownership rules.
- Resize and supported zoom fixtures preserve alignment.
- Time and season fixtures differ where expected and remain unchanged where ownership is 2D.

## Reference Images

Reference images may include:

- Civ VI screenshots for a known scene and environment preset.
- Alternate-skin screenshots showing the intended palette and terrain treatment.
- Existing Civ III screenshots used for anchor, density, and readability calibration.
- Previously accepted C3X renderer output used as regression baselines.

References must carry metadata describing source, scene, approximate camera, time/hour, season, resolution, crop, and intended comparison criteria. Exact pixel equality is inappropriate across different engines, projections, antialiasing, or camera composition.

For controlled Civ VI captures, keep the map/save, camera center, zoom, resolution, graphics settings, expansion/mod set, color management, and crop fixed. Record whether each reference uses vanilla Civ VI or the Civ V Environment Skin. If Civ VI does not expose an exact clock value for a capture, record the reference as an approximate lighting phase rather than fabricating an hour.

Copyrighted reference screenshots and large local outputs should remain ignored/local. Tracked tests may store derived numeric targets, permissibly sourced references, or deliberately contributed small fixtures.

## Day/Night Reference Matrix

At minimum, each representative scene should render:

- Midnight: hour 0.
- Sunrise: hour 6 or the configured sunrise.
- Noon: hour 12.
- Sunset: hour 18 or the configured sunset.

The matrix expands across summer, fall, winter, and spring when seasonal rendering is enabled. Comparisons evaluate:

- Sun and shadow direction.
- Direct and ambient light color.
- Terrain luminance, contrast, and saturation.
- Map readability and category separation.
- Water response and reflections.
- Snow and seasonal-material coverage.
- City/unit emissive behavior.
- Smoothness and plausibility between adjacent sampled hours.

Reference values are targets, not claims that C3X will reproduce Civ VI's renderer exactly.

Maintain separate reference rows for vanilla Civ VI and the preferred alternate skin whenever both are available. The Civ V Environment Skin changes its terrain/water/vegetation treatment and documents a noon lighting color-key adjustment, so a single shared reference row would hide the very stylistic difference the alternate pack is meant to preserve. At minimum, use matched midnight/sunrise/noon/sunset captures of the same Civ VI scene and camera for each reference style.

"Broadly aligned" is a recorded rubric decision, not a raw pixel threshold. A candidate should preserve the intended phase of day, shadow direction, relative luminance ordering, palette character, map readability, and night emissive balance. Histogram/color-distance/edge metrics flag regressions and outliers; a contact sheet and human decision determine whether cross-engine differences are acceptable.

## Quantitative And Perceptual Metrics

Useful deterministic metrics include histogram/luminance ranges, edge density, clipped-pixel counts, shadow direction, color-distance samples, mapping coverage, depth conflicts, and seam scores. SSIM or perceptual hashes may be useful for regression against C3X's own accepted outputs but should not gate similarity to differently framed Civ VI screenshots by themselves.

An AI vision reviewer may inspect contact sheets using a versioned rubric:

- Broken geometry, missing tiles, seams, and clipping.
- Implausible scale or overlap.
- Muddy or unreadable lighting.
- Inconsistent visual style between neighboring assets.
- Seasonal or time-of-day changes that appear excessive or absent.
- Large departures from the stated reference art direction.

AI review output must include observations and confidence, not only a numeric score. Model/version, rubric version, and reviewed image manifest are recorded. A changed model cannot silently redefine a release gate.

## Human Review

Human approval remains required for art-direction gates. The review contact sheet should make comparison inexpensive by showing the same camera across packs, hours, seasons, and viewport sizes. Reviewers can accept, reject, or annotate individual cells and preserve those decisions in a small review manifest.

AI may triage large matrices and identify suspicious cells. Humans decide whether stylistic differences are desirable, especially when matching a Civ V-inspired skin rather than vanilla Civ VI.

## Manual Interaction Budget

Manual in-game screenshots and user feedback are strategic checkpoint evidence, not a requirement for every implementation step, asset revision, or failed test. Agents must finish all portable tests, deterministic standalone/replay renders, metrics, contact sheets, and feasible local capture automation before requesting user action.

The normal policy is:

- Reuse a still-valid accepted screenshot, scene export, and review decision across all steps that do not change the behavior it proves.
- Make at most one bundled manual-evidence request per strategic milestone checkpoint under ordinary circumstances. The request names the exact save/BIQ, camera, configuration, and minimal screenshots or decisions needed.
- Batch variants into one session and one contact sheet instead of asking separately for each viewport, hour, season, asset, or code iteration.
- Do not request a new screenshot merely because implementation changed internally; request one only when a user-visible ownership, compositing, alignment, animation, performance, or art-direction claim changed materially.
- Prefer agent-operated capture or an in-game capture command when available. The user should not become the renderer's test harness.
- When manual evidence is unavailable, mark the strategic checkpoint `pending_manual_checkpoint`, keep the prior decision unchanged, and continue every task that does not depend on that decision. Do not repeatedly ask on later turns.
- A specific failed or ambiguous checkpoint may justify one focused follow-up after the agent has diagnosed and fixed the likely cause. Open-ended screenshot gathering is not acceptable.

Expected strategic checkpoints are the first live map bridge/populated export, a production terrain and alternate-skin review, a consolidated map-object/animation integration review, and release-candidate art-direction approval. Milestone plans may combine or eliminate these when automated or existing evidence is sufficient.

## Seasonal Asset Authoring Validation

Generated seasonal textures and materials must preserve model UVs, texture dimensions, alpha semantics, atlas cell boundaries, and logical asset IDs. Procedural snow/material effects are preferred when they provide consistent geometry-aware results. Image generation may produce new detail or vegetation art, but generated variants must pass the same structural checks and contact-sheet review as imported art.

## Milestone Placement

- M4.2 establishes fixture-matrix rendering, deterministic image metrics, and reference contact sheets, including metadata for controlled Civ VI day/night captures.
- M5.2 automatically exports a populated viewport, runs deterministic strict-schema and standalone gates, and uses a normal full-screen capture as lightweight in-game health evidence. Formal matched drop-in evidence is reserved for regression diagnosis or release review.
- M6.2 compares vanilla and alternate visual-skin packs using stable logical IDs; the Civ V Environment Skin is the preferred candidate if conversion permission is documented.
- M8 adds seasonal authoring automation and production human/AI review workflows.

Each placement above follows the manual interaction budget. Intermediate subtasks generate automated evidence and roll it into the next strategic checkpoint rather than creating their own screenshot request.
