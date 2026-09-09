# Coastal wave study

Recover the embedded 16-crest atlas, auxiliary foam and crest-delay table, then
evaluate them against the current D3D11 water material in an isolated snapshot.
See [findings](../../../docs/ocean_wave_findings.md) for source evidence,
commands, reconstruction choices and production work still required.

`study.py` requires NumPy/Pillow. `prepare` copies local packs/shaders and freezes
a matching candidate DLL/preview pair after category preparation. `render` and
`sequence` dispatch only standalone native fixtures. Output is ignored under
`Renderer/lab/out/waves/`; no source art, production staging or reference
replacement is part of this experiment.

The shader's coastline coordinate, texture-slot reuse and texture-carried clock
are study scaffolding. They are not production contracts or evidence of a
working live-game animation scheduler.

## Current production quality probe

`quality.py prepare` freezes the staged renderer, matching preview, shader and
packs under `lab/out/waves-quality/`. Run `quality.py baseline` before changing
source, then `quality.py refined` to compare the current canonical wave material.
An optional `--dll PATH` evaluates a separately frozen candidate for geometry
changes. Native runs must be serial. Cases, hour and zoom are selectable, and
each run exercises the production wave lifecycle assertions. Preparation and
rendering do not stage or replace references. NumPy/Pillow are not required.

After the two matched renders, `quality.py review` writes a context comparison
and a 2× detail crop. This optional command needs Pillow and first asserts
pixel-identical wave-off controls. No image enhancement is applied.

The follow-up shoaling/long-front study did not replace the preceding production
appearance: direct comparison favored its finer feathered foam. Findings and
local artifact paths are recorded in the source report. `quality.py --output`
selects a separate study directory; `--shader` and `--label` select a local
material experiment. `--sequence 120` writes 30 seconds of quarter-second
samples using a matching newly built preview. `review --label NAME` compares
that candidate with the frozen baseline and verifies identical wave-off pixels.
