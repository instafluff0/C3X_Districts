# Q0 platform runner (provisional contract v1)

Run from the repository root:

```sh
python3 Renderer/terrain_lab/v2/app/bootstrap_tools.py
python3 Renderer/terrain_lab/v2/app/runner.py quick
python3 Renderer/terrain_lab/v2/app/runner.py check --candidate matrix-01
python3 Renderer/terrain_lab/v2/app/runner.py compose --candidate compose-01
python3 -B -m unittest discover -s Renderer/terrain_lab/v2/tests/platform -p 'test_*.py'
```

The Mac needs Apple command-line tools and GPU access. A restricted shell may
not expose a Metal device; the runner reports this explicitly. The pinned
public Homebrew shader tools are installed only in `.local/`, with SHA-256
verification. `C3X_LAB_SHADER_TOOLS` may name an equivalent tool root. Ordinary
commands never invoke Parallels. Shader compilation uses the same HLSL via
Khronos glslang/SPIRV-Cross, then Apple's runtime Metal compiler. Apple's
preferred DXIL converter is not installed on this host. No separately authored
Metal look is maintained. D3D11 consumes that same HLSL through SM5.

The public coordinator dispatcher at `Renderer/tools/lab_v2.py` has not been
edited: it is outside this package's ownership. This entry point is usable now.

Each fixture declares its owner, normalized pack mounts, modules, camera,
references, isolation requests and effective settings. Use `--fixture`,
`--candidate`, `--settings`, and `--output` to select owned inputs and outputs.
Outputs must be inside the selected owner's `owns_paths`; each includes the
campaign, owner and candidate. Q0 defaults to `app/out/Q1/Q0-platform/`.
Other owners can put their module/shader/fixture and output inside their own
paths; no Q0 source edit is required for shader or setting variants.

Baseline defaults are anisotropy 8, no mip bias, one sample, render scale 1,
byte-box reduction, and zero camera offset. Additional sample counts are
capability checked. Render scale 1/2/4 preserves the final output size.
Offsets are final-output pixels at both zooms. Implicit-gradient HLSL samples
accept mip bias; explicit `SampleLevel` height/geometry reads are unchanged.
Per-material sampler overrides are not yet available.

`check` renders four phases at two zooms and repeats each frame on one shared
Metal device. GPU textures and buffers are reused by content hash across
variants, while shader functions are cached per module. A return-to-origin
camera sequence is reproducible. Cost files declare GPU timing scope, wall
time, current allocation and sampled allocation high-water. The high-water
measurement does not claim to capture undocumented driver transient peaks.

A Q1-owned postprocess setting may replace `"box"` with:

```json
{"shader":"<owner-relative-repository-path>.hlsl","owner":"Q1-sampling","contract":1}
```

Its `CSPost` compute entry receives source texture `t0`, destination UAV `u1`,
and `uint2 input_size; uint2 output_size;` at `b2`, dispatching 8x8 threads.
Contract1 transports frozen display-encoded BGRA8 with straight alpha.
The explicit Q6 linear branch and Q1 contract2 are implemented; see
`../contracts/source_adapters_v1.md`. The interface witness under common
shaders proves plumbing only and is not a selected quality policy. Box-reduced
runs preserve a full-size `.resolved.bmp` before reduction.

C++ translation units and shader entries have separate cache identities.
The default packet wire remains v2; readers accept v1–v5. New linear,
per-draw shader and shared-buffer extensions are opt-in. Texture mips and
vertex/constant buffers use SHA-256 content references and shared hard links;
readers verify each reference before creating GPU resources. This reduced the
initial 7.67 GB packet corpus to 2.20 GB unique content and 1.45 MB metadata.
Fixtures, normalized pack file closures and geometry are content addressed.
The migration provider co-caches its baked shadow fields with geometry because
v1 interleaves them in the same vertex stream. Cached artifacts and metadata are
verified before reuse; corruption and incomplete commits fail explicitly.
Delete only the diagnosed cache entry to rebuild it. The cache uses a lock per
content key and atomic publication. Outputs are not used to bypass rendering.

The baseline provider is an atomic migration adapter. Independent modules can
use `provider: "cpp_packet"`, an owned C++ `source`, and the existing shader
contract. The builder receives output packet, width, height, hour, zoom, and
fixture JSON as positional arguments. Multiple modules compose in the explicit
`after` dependency order; cycles, missing dependencies, viewport drift, and
incompatible attachment contracts fail clearly. Compatible linear modules
may have independent shaders under wire4/5. The two synthetic owner probes have
separate source and output paths and passed concurrent rendering. Versioned world attributes, source metadata, terrain/hydrology/placement hooks,
checked source/corridor sidecars, and owned postprocessors are documented in
`../contracts/source_adapters_v1.md` and `../contracts/scene_exchange_v1.md`.
Visual systems remain owned by their tracks; diagnostic proxies are not beauty
acceptance.
Do not treat this runner or successful frames as an accepted LQ0 release.

The closing baseline matrix is:

```sh
python3 Renderer/terrain_lab/v2/app/runner.py promote \
  --fixture Renderer/terrain_lab/v2/tests/platform/complete.fixture.json \
  --candidate full-01
```

This renders 192 tiles, four phases and both zooms, with repeats and D3D11
parity via the existing `renderer_dev.native_command_result` dispatcher.
It never promotes into Integration or edits v1 outputs/handoffs. Full project
verification and the remaining independent-module gates still belong to formal
package closure; the command name is an execution tier, not approval.

## Verified actual map

`app/real_map.py import` verifies the actual user test.biq through the existing
parser and caches portable data. `export mixed --owner Q0-platform --output
Renderer/terrain_lab/v2/tests/platform/out/my-region --augment` emits exact
terrain/halo plus a separate deterministic layer. Omit --augment for original
terrain only. Registered coordinates, source/parser/profile hashes and overlays
are checked on every fixture load; ordinary replay never calls the VM. The
historical complete.csv is Ancient Treasures regression data, not test.biq.
Registry paths are read-only for visual owners; Q0 registers requested regions.
Default halo6 covers the inherited longest terrain shadow query. User evaluation
regions may be up to192 tiles, without creating automatic acceptance claims.
