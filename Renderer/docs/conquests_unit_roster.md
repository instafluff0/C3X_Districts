# Standard Conquests unit animation coverage

This maintenance extension covers all 93 keys in the standard Conquests roster inventory, plus the existing Builder alias. It does not promote a renderer milestone or apply the separate Lab v2 visual package. Runtime art is generic normalized C3X data; installed source-game assets are needed only for offline rebuilding.

## Runtime and integration contracts

- 78 animation families: 55 ordinary kits, 20 composed kits and three original generic missile bodies. Shared-art aliases use the same runtime family and pose cache.
- The catalog loads once; meshes and textures load only on a pose-cache miss. Shared payload residency is bounded at 96 MiB, and completed sprites retain their independent 8 MiB / 128-entry cache.
- Native screen position and unit identity are absent from the pose-cache key. Scrolling reuses the same posed pixels. ATTACK1/2/3 share their canonical attack cache entry when the remaining pose inputs match.
- Native action, cursor, facing, displayed civilization color and gameplay timing remain authoritative. No new input or gameplay-command hook is added.
- The existing three unit inleads also capture an Army commander and its representative member. Successful member draws expand the Army owner rectangle that native Animator erases. No CSV change is needed.
- Hidden parts may have an authored zero-scale pose. Sampling preserves the collapsed positions without rejecting the complete kit for a singular normal matrix. Nonfinite and malformed input still fails safely.
- Paired chariot horse clips already contain the parent socket motion; their explicit parent-space calibration prevents double application. Terminal composed death clips may leave the native sprite area; GPU clipping still confines all output to that same native-sized target.
- Compatible rifle clips require the Spec Ops rifle in the right hand; its binocular attachment is omitted. Worker and Settler death actions use a compatible humanoid clip with held tools hidden.
- Missile geometry is original generic C3X art. Several roster entries share the nearest available source kit; these bindings do not claim exact Civ III silhouettes or source-game visual parity.
- Some source actions are poses or aliases, including siege destruction poses. Native Civ III still owns effects and removal. Unknown/custom units, invalid data or unsupported actions retain native fallback.

## Build and verification

Run from the repository root. Ordinary INSTALL.bat uses the staged DLL and runtime pack; the agent does not launch the game or run the installer.

```sh
python3 Renderer/tools/asset_compiler/build_unit_animation_runtime.py --standard-roster --output Renderer/packs/UnitRosterRuntimeCandidate
python3 Renderer/native/verify_unit_roster.py --hour 12
python3 Renderer/native/verify_unit_roster.py --hour 0
C3X_UNIT_TEST_PACK=UnitRosterRuntimeCandidate python3 -m unittest Renderer.native.test_unit_animation_runtime -q
python3 Renderer/renderer.py integration units
```

`--reuse-source-pack NAME` is an explicit incremental build option for an unchanged source pack. It checks the previous compiled payload hashes and rebuilds every other source pack. Omit it for a clean source rebuild. Native compilation uses the documented `renderer_dev.windows_command_result` helper with `Renderer/native/BUILD.bat candidate-compile`; injected changes use the approved injection smoke workflow.

Local verification output is written to `Renderer/lab/out/verification/animation/roster/`.
The Windows matrix covers all 94 keys, three native cursor phases, four headings
and both zooms, including attack-slot aliases. It also exercises the existing
input-independent canvas, color-key, clipped RGB555/RGB565, native interruption,
retained-terrain and scroll checks. Raw source-pose proof covers ordinary kits;
an additional source comparison verifies the chariot parent-space correction.
Generated reports are disposable and do not replace the user-run live-game
checkpoint.

Historical L/I and campaign gates are retired. Preserve the current unit behavior
checks and explicit category approval; the baseline is the current production
build, including its later source-normal and material improvements.

## Roster bindings

| Conquests unit | Runtime art family |
| --- | --- |
| Settler | `unit/settler` |
| Worker | `unit/worker` |
| Scout | `unit/scout` |
| Explorer | `unit/ranger` |
| Warrior | `unit/warrior` |
| Jaguar Warrior | `unit/aztec_eagle_warrior` |
| Archer | `unit/archer` |
| Longbowman | `unit/crossbowman` |
| Bowman | `unit/archer` |
| Spearman | `unit/spearman` |
| Pikeman | `unit/pikeman` |
| Hoplite | `unit/greek_hoplite` |
| Impi | `unit/zulu_impi` |
| Enkidu Warrior | `unit/warrior` |
| Swiss Mercenary | `unit/pikeman` |
| Chasquis Scout | `unit/inca_warakaq` |
| Javelin Thrower | `unit/mayan_hulche` |
| Swordsman | `unit/swordsman` |
| Medieval Infantry | `unit/man_at_arms` |
| Legionary | `unit/roman_legion` |
| Immortals | `unit/persian_immortal` |
| Samurai | `unit/japanese_samurai` |
| Crusader | `unit/teutonic_knight` |
| Cavalry | `unit/cavalry` |
| Chariot | `unit/heavy_chariot` |
| Cossack | `unit/russian_cossack` |
| Horseman | `unit/horseman` |
| Knight | `unit/knight` |
| War Elephant | `unit/indian_varu` |
| Mounted Warrior | `unit/horseman` |
| Rider | `unit/mongolian_keshig` |
| War Chariot | `unit/egyptian_chariot_archer` |
| Three-Man Chariot | `unit/heavy_chariot` |
| Hussar | `unit/hungary_huszar` |
| Ancient Cavalry | `unit/macedonian_hetairoi` |
| Musketman | `unit/musketman` |
| Musketeer | `unit/french_garde_imperiale` |
| Rifleman | `unit/line_infantry` |
| Infantry | `unit/infantry` |
| Marine | `unit/spec_ops` |
| Paratrooper | `unit/spec_ops` |
| Guerilla | `unit/ranger` |
| WWII Paratrooper | `unit/spec_ops` |
| Mech Infantry | `unit/mechanized_infantry` |
| Tank | `unit/tank` |
| Panzer | `unit/tank` |
| Modern Armor | `unit/modern_armor` |
| TOW Infantry | `unit/modern_at` |
| Flak | `unit/antiair_gun` |
| Mobile SAM | `unit/mobile_sam` |
| Catapult | `unit/catapult` |
| Trebuchet | `unit/trebuchet` |
| Cannon | `unit/field_cannon` |
| Artillery | `unit/artillery` |
| Radar Artillery | `unit/rocket_artillery` |
| Cruise Missile | `unit/cruise_missile` |
| Tactical Nuke | `unit/tactical_nuke` |
| ICBM | `unit/icbm` |
| Curragh | `unit/phoenicia_bireme` |
| Galley | `unit/galley` |
| Dromon | `unit/byzantine_dromon` |
| Caravel | `unit/caravel` |
| Frigate | `unit/frigate` |
| Man-O-War | `unit/de_zeven_provincien` |
| Privateer | `unit/privateer` |
| Galleon | `unit/galleon` |
| Ironclad | `unit/ironclad` |
| Transport | `unit/modernembark` |
| Carrier | `unit/aircraft_carrier` |
| Submarine | `unit/submarine` |
| Destroyer | `unit/destroyer` |
| Battleship | `unit/battleship` |
| AEGIS Cruiser | `unit/missile_cruiser` |
| Nuclear Submarine | `unit/nuclear_submarine` |
| Carrack | `unit/portuguese_nau` |
| Cruiser | `unit/missile_cruiser` |
| Fighter | `unit/fighter` |
| Bomber | `unit/bomber` |
| Helicopter | `unit/helicopter` |
| Jet Fighter | `unit/jet_fighter` |
| F-15 | `unit/jet_fighter` |
| Stealth Fighter | `unit/jet_fighter` |
| Stealth Bomber | `unit/jet_bomber` |
| Leader | `unit/great_general_classical` |
| Army | `unit/great_general_classical` |
| Keshik | `unit/mongolian_keshig` |
| Conquistador | `unit/spanish_conquistador` |
| Berserk | `unit/norwegian_berserker` |
| Sipahi | `unit/cavalry` |
| Gallic Swordsman | `unit/gaul_gaesatae` |
| Ansar Warrior | `unit/arabian_mamluk` |
| Numidian Mercenary | `unit/medjay` |
| Hwacha | `unit/korean_hwacha` |
