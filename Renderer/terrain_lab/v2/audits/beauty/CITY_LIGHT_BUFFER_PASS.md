# Buffered city lights and restored spill

The dense-city Metal compilation blocker is resolved by moving the existing
light/blocker arrays from shader literals into generic shared-frame data. The
first new scene completed preparation, pipeline creation, two renders and
readback in 5.24 seconds; the other five took 2.58-3.96 seconds. These are whole
replay measurements, not isolated compiler timings. The previous static-array
trials were stopped after several minutes of verified active compilation.

The four culture-density scenes now have eight passing Metal/D3D comparisons.
Their Windows BMPs are byte-for-byte identical to the earlier static-array
Windows renders. This supersedes the pending static trials with verified
equivalent buffered scenes; it does not retroactively complete those old runs.

With compilation independent of the light values/count, both Asian medium scenes
can restore their complete 53-proxy spill set instead of the 32-proxy workaround.
[Gameplay-size comparison](out/city-light-buffer-r1/full-light-native-comparison.png):
the right column has warmer, more legible ground around the buildings and less
abrupt darkness between nearby facades. In the inland case 1,378 night pixels
change above 2/255 (maximum 43); the holdout changes 1,143 (maximum 62). Daylight
is exact, as are pixels outside the recorded city regions. This is a small
local night-readability improvement toward `Renderer/canonical/nightlights.jpg`,
not an overall Civ VI-quality claim. The existing surface windows, HDR glow,
geometry, source proportions and single-era city selection stay intact.

## Generic data contract

`qa/city_light_buffer_probe.py` uses a preserved pre-light render and a frozen
`lights.json`. It appends data with `qa/append_frame_data.cpp`, preserving all
old buffers and textures and redirecting each draw's b1 binding to an extended
buffer. The two existing GPU backends already support this size. No backend,
native, injected, pack or texture-slot change is required.

The existing 80-byte shared shadow prefix is unchanged. The complete frame is
7,296 bytes; its additional payload is 7,216 bytes of little-endian float32.

| Byte offset | Data |
| ---: | --- |
| 0 | Existing five shadow float4 values, unchanged |
| 80 | Light count, blocker count, gain, Z metric |
| 96 | Light-envelope low XYZ, padding |
| 112 | Light-envelope high XYZ, padding |
| 128 | 128 position/range float4 values |
| 2,176 | 128 color/intensity float4 values |
| 4,224 | 128 direction/owner float4 values |
| 6,272 | 32 blocker-low float4 values |
| 6,784 | 32 blocker-high float4 values |

Unused rows are zero. Counts, finite values, ranges, gain and owner references
are checked before serialization. Decimal values match the old generated shader
literals before float32 packing; this introduces no half-precision approximation.
Lights remain generic position/color/range/direction records with no source-game
civilization IDs. Main and reflection shaders consume the same b1 data.

The source shader closure is identical between each 32- and 53-proxy pair.
The same inland shader also serves ancient medium and Asian large geometry.
The holdout retains its different terrain closure. This permits shader reuse
instead of recompilation for each city's local light values.

**Composition order:** append city geometry, rebuild shared shadows, then append
local-light frame data. Existing shadow rebuild/reference tools assume the
80-byte shadow layout; use the preserved original packet from `binding.json`
for those operations, then apply the light-buffer stage again. Do not run the
old shadow rebuild directly over the extended frame and silently lose its tail.
This is a Lab adapter contract for future generic runtime implementation, not
native promotion or a new requirement on Civ III capture.

## Checks and limits

Six scenes / twelve day-night Metal/D3D comparisons pass at gameplay zoom1.
Twelve independent `frame_data_contract.cpp` checks verify the original geometry,
constant buffers, textures, material/draw state and shadow prefix. Two focused
payload tests check float32 literal equivalence, array offsets and capacity.
The full-spill controls retain every earlier selected proxy and the same blockers.
Measured night GPU time is 54.58 -> 58.36 ms inland and 56.54 -> 57.45 ms in the
holdout; these single replay samples are descriptive, not a performance gate.
The maximum 128-light capacity is not a claim of measured 128-light performance.

Recheck saved evidence with Python providing NumPy/Pillow:

```sh
python3 Renderer/terrain_lab/v2/qa/city_light_buffer_evidence.py
python3 -m unittest discover -s Renderer/terrain_lab/v2/qa -p test_city_light_buffer.py
```

The selected new night candidates are `city-light-buffer-r1/asian-medium-full`
and `asian-holdout-full`. The large city's scattered geometry remains unselected;
transport equivalence does not validate its composition. The next visible work
is connected large-city growth and broader culture/era/size/palace coverage.
The prior American capital and lake-reflection witness remain preserved. This
pass provides no new shoreline reflection-quality claim. The local-light model
is still an authored source-informed approximation with box blockers, not
recovered source engine light placement or full light transport. All native,
milestone and human approval gates remain unchanged.
