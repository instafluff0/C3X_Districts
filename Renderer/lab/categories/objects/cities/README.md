# Cities

The selected one-tile culture/era/population recipes from
[the Cities Lab](../../../studies/cities/README.md) now supply the production city
pack. They replace the previous generic city growth and runtime wall geometry.
Five cultures, four eras, three sizes, capital state, town walls and three stable
variants produce 480 compositions.

`python3 Renderer/renderer.py prepare` builds through
`Renderer/tools/prepare_city_recipes.py`. Source imports and layout construction
remain offline. Runtime consumes a generic binary, complete DDS texture closure,
and already compiled placements and lighting data.

See [current integration details](../../../../native/city_fidelity/CITY_FIDELITY.md)
for selection, restored material coordinates, shared shadows and verification.
The older reference images remain comparison aids; they are not the new recipes
or an integration gate. No reference was replaced during promotion.

Standalone examples use the actual Renderer64 DLL:

```sh
python3 Renderer/tools/capture_city_border_examples.py
```

Outputs and hash receipts are under `Renderer/native/build/cities-borders/examples`.
They are synthetic scenes, not live Civ III captures.

## Readability candidate (Lab only)

[The city readability study](../../../studies/city_readability/README.md) holds a
Lab candidate: denser, larger bodies, era accents (smokestacks, towers), crisp era
ground plates and a brighter, higher-contrast body response, plus version-5
site awareness (bodies and plates yield to rivers, bridges, water, mountains and
steep relief). It renders through `C3X_RENDERER_CITY_PACK`; the production pack
and its output are unchanged until the user accepts the candidate.
