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
