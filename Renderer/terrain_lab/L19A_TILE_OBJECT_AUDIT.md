# L19A Goody Hut And Colony Audit

Status: complete; deterministic 192-tile alternate-skin candidate critically inspected and explicitly approved under the user's 2026-09-06 autonomous-review authorization.

## Scene and provenance

- The accepted L19 terrain/object scene remains unchanged when L19A bodies are disabled.
- `fixtures/l19a_tile_objects_192.csv` adds twenty-two deterministic Lab-only records without changing BIQ terrain or gameplay fields: every one of the eight Civ III hut buckets, all three source hut variants, one viewer-hidden hut, twelve visible colonies across all four eras and owners, and one hidden colony.
- Every colony occupies a real visible L16 land-resource witness. Colony owner and territory owner are intentionally different in every record, proving that extraterritorial ownership is not inferred from tile territory.
- `tile_object_runtime.bin` recursively flattens the three normalized goody-hut compounds and the three preindustrial plus three strict industrial camp compounds. It preserves every material used by silhouette-bearing source geometry within the generic eight-texture ABI.

## Critical visual review

- The first candidate was rejected because global material frequency discarded the hut foliage/stone atlas and retained broad near-flat camp ground. Huts collapsed into orange palisade blocks and colonies into nearly invisible scatter.
- The second candidate retained all silhouette materials and removed only near-flat recursive ground scatter, but both families remained too small at the 192-tile strategic view.
- The accepted candidate corrects normalized source Z for the established Civ III isometric pixels-per-tile basis and applies final uniform body scales of `1.25` for huts and `0.90` for colonies. Huts are readable tribal compounds while remaining subordinate to cities and relief; colonies are smaller than huts/forts but survive reduced zoom.
- Colonies occupy stable clear quadrants around independently rendered resource bodies. Strategic resources remain separately readable in complete, isolation, and reduced frames.
- Huts use the frozen eight-to-three bucket mapping, culture/era-neutral presentation, and stable diagonal facing. Hidden records are suppressed before geometry generation.
- A restrained seven-percent owner response differentiates colonies without replacing source materials or implying territory control. Each compound emits one shared-direction shadow; child-part shadow duplication is discarded.
- No smoke, particles, procedural campfire, bloom, or invented lamp is present. The source pack exposes no emissive texture for these nine compounds, so the accepted night frame uses only shared environment lighting and source geometry.

## Deterministic evidence

Two final unchanged `python3 Renderer/tools/renderer_dev.py lab` runs produced byte-identical outputs:

- noon complete: `d620366d8e1b1ae13ffee9ce918747f8c0a61ca825612b67560f6cefd4e14736`
- midnight complete: `71f4249910a525deba33e97c3ff90935851c1c8052fef754825d702a9c2ee50d`
- reduced: `e9a485a5deb1082bbf9d026aed40a2020146c90fc76e1c5b95b4df50a542449c`
- no L19A objects: `99a82cff7891d946481a4d96ee5eb80e42b71baa2457a118dedf1a25ee276e2c`
- terrain+resources+L19A isolation: `6d33b0301deccd57cb73b5b7a3009f5d4d46c68cdea218499277ed6bf109ccb0`
- Lab scenario: `b4c5ec90004960237411106104276b5639e31e8c6151688e01f76128d0310fe1`
- runtime bundle: `16e1acdb3835b25cd929ad51221e08ce8b303c0caff4050f98aded15bcb41ec3`

The no-object control is byte-identical to the approved L19 noon render.
