# I13A Shared Lighting Integration Evidence

I13A consumes the frozen approved `handoffs/L13A_lighting.json` record. The game
renderer evaluates the existing shared environment from authoritative hour and
season and uploads sun, moon, ambient, exposure, shadow, water, emissive-policy,
and time values through a production-only semantic constant buffer. The frozen
approved shader applies normal self-shading and one coherent cast direction to
raised terrain, volcanoes, forest/jungle bodies, shore features, and river
clutter. Visible object-owned lights remain absent until L17/I17.

`terrain/i13a_handoff_fidelity.json` freezes every approved phase/reference hash
and the production shader revision. Portable contracts reject unapproved road
bindings and prove the environment/shadow wiring. The Windows production smoke
renders noon, sunset, midnight, and sunrise deterministically, exercises both
Civ III zoom bases and device recreation, and reports zero native fallback. The
integration workflow emits ignored near/noon and far/sunset replays from the
approved L13 192-tile BIQ fixture.

Post-gate performance maintenance retains the approved output while reducing a
400-rendered-tile/800-record Windows fixture from 57.790 seconds cold to 5.241
seconds cold. A subsequent anchor-only scroll renders in 1.054 seconds by reusing
an anchor-independent shadow field whose fingerprint still includes every
terrain, feature, visibility, wrap, zoom, environment, content, and device input.
The native smoke rejects scroll reuse unless the scene invalidates normally and
the scroll costs less than half of the cold render. API v7 additionally carries
the authoritative output clip, and an exhaustive native DIB test proves that a
partial `SRCCOPY` changes every pixel inside that rectangle and no pixel outside
it. This closes the live truncated-art regression caused by copying cleared
off-clip pixels over Civ III's retained map surface.

A second live report showed that Civ III can make the base-terrain `m19`
traversal itself partial during unit-selection and UI redraws. Custom-on `m71`
now captures the complete visible traversal regardless of that caller rectangle;
the caller's original clip is retained only for the final `SRCCOPY`. The native
cache additionally proves that a partial or reordered record traversal containing
only unchanged terrain returns the complete retained bitmap with zero renderer
ticks. This makes unit-selection redraws deterministic cache hits instead of
five-second terrain rebuilds and prevents an incomplete traversal from becoming
the next cache entry.

Feature depth now follows the frozen Lab ground-plane rule: screen depth is
derived from the unlifted ground anchor and feature height independently pulls
vegetation and river rocks toward the camera. The former production expression
used the already-lifted screen Y and could place tall bodies behind adjacent
ground diamonds, producing the reported diagonal tree and forest truncation.

The passing replay hashes are:

- near/noon: `55b04776307f0e24dcaaca55f1ef73fc1f2cbc309e8bcad79b11eb676ce8eca7`
- far/sunset: `9a9ac7a93b74d38eb6c54a67d1f08ced630024542f3878d0d55cd519aeece6c4`

No new Civ III patch symbol or `civ_prog_objects.csv` entry is required.
