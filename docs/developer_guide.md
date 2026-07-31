# Developer Guide

## Windows

Before building, run the environment checker from PowerShell:

```powershell
.\checkEnvironment.ps1
```

It verifies the JDK, native compiler, Vulkan shader tools, DLSS SDK files,
hardware information, and free disk space without changing the system. Add
`-IncludeCacheSizes` to report the size of disposable Gradle, build, and
Minecraft run data:

```powershell
.\checkEnvironment.ps1 -IncludeCacheSizes
```

1. Install the Vulkan SDK from <https://vulkan.lunarg.com/sdk/home>.
   The installer sets `VULKAN_SDK` automatically.
2. Download the DLSS SDK from <https://github.com/NVIDIA/DLSS/releases>.
   Extract it, then set `DLSS_SDK` to the folder you extracted.

   To set it permanently for your Windows user account, run PowerShell with:

   ```powershell
   [Environment]::SetEnvironmentVariable("DLSS_SDK", "C:\path\to\dlss-sdk", "User")
   ```

   Restart your terminal after setting it. To set it only for the current
   PowerShell session, use:

   ```powershell
   $env:DLSS_SDK = "C:\path\to\dlss-sdk"
   ```

3. Configure and build the native shim:

```powershell
.\buildNative.ps1
```

   The script accepts `VULKAN_SDK` and `DLSS_SDK` when they are set. It also
   recognizes the default personal Codex toolchain layout:
   `Documents\Codex\Toolchains\VulkanSDK\1.4.350.0` and
   `Documents\Codex\Toolchains\DLSS`.

4. Run the client. This also performs an incremental native shim build, so it
   is the only command normally needed after initial setup:

```powershell
.\runClient.ps1
```

For repeatable performance measurements, enable Caustica's frame-stage CSV:

```powershell
.\runClient.ps1 -FrameStats
```

After loading a representative world, record a 45-second Java Flight
Recorder profile from a second PowerShell window:

```powershell
.\profileMinecraft.ps1 -DurationSeconds 45
```

CPU frame and stage timings are written to `run\rt-frame-stats\frame.csv`.
Its `frame.traceMs` column remains the combined command-recording time, with
`frame.tracePrimaryMs` and `frame.traceIndirectMs` providing the pass split.
`frame.temporalValidationMs` measures motion-vector reprojection and conservative
surface-history acceptance; debug view `Temporal Validation` visualizes accepted pixels in green
and rejection classes in red, magenta, yellow, blue, or black.
`frame.reservoirInitMs` measures initialization of the current direct-light reservoir slot.
`frame.historyCaptureMs` measures the deterministic surface-history copy boundary.
The same switch also writes non-blocking Vulkan timestamp results to
`run\rt-frame-stats\gpu.csv`, split into entity BLAS, TLAS, primary trace,
indirect trace, DLSS-RR/fallback upscale, exposure, display mapping, and
output-copy stages. `traceMs` remains the comparable total of both trace
passes; `traceIndirectMs` also includes their handoff barrier.
GPU rows can arrive several frames late: query results are read only after the
existing graphics timeline reports completion, never by stalling the GPU.
JFR recordings are written to `run\jfr`. Without `-DurationSeconds`, the
profiler records until Enter is pressed.

The wavefront primary-to-indirect queue owns two 48-byte records per render
pixel, so its allocation is exactly `renderWidth * renderHeight * 96` bytes.
The renderer logs the actual byte and MiB count whenever the render size is
created. Useful reference points:

| Render size | Queue allocation |
| --- | ---: |
| 569x320 (854x480 DLSS Quality reference window) | 16.67 MiB |
| 854x480 (same window at native render resolution) | 37.53 MiB |
| 1920x1080 native | 189.84 MiB |
| 2560x1440 native | 337.50 MiB |
| 3840x2160 native | 759.38 MiB |

The clarity-first direct-light reservoir ABI is generated from Slang reflection
and occupies 80 bytes per pixel per slot. Two history slots use about 131.4 MiB
at 1280x673 and 295.6 MiB at 1920x1009. This intentionally favors inspectable
ReSTIR/GRIS semantics over packing until profiling identifies real bandwidth
or residency pressure.

The path-reservoir ABI is currently 176 bytes per pixel per slot. Its replay-control
lane stores the two wavefront segment seed pairs, segment count, replay version, and
terminal-state hashes; the additional proposal-components lane captures light,
continuation, and roulette PDF products plus event counters (including an explicit
canonical-endpoint validity bit), while the canonical radiance lane stores one
replayable sky/emissive endpoint. The final three lanes store the selected canonical
path's second-hit reconnection vertex, packed valid/depth metadata, oriented normal, and
continuous source-edge proposal density plus the exact selected local first-event RGB throughput.
Replay ABI 9 also packs the first-edge event kind
(diffuse, glossy, delta, or transmission) so a future shift cannot confuse continuous and
discrete measures. Replay ABI 10 uses the previously reserved final 32 bits as a compact
mapping control: zero is an identity path and one is a one-hop diffuse reconnection. Two path-history slots
therefore use about 289.2 MiB at 1280x673 and 650.3 MiB at 1920x1009. This is
intentional: replay correctness is being established before any packing or compression pass.
Debug view 15 is an opt-in seeded replay evaluator: it re-traces the selected reservoir's stored
segment seeds and color-codes independent mismatches in terminal path/proposal state, topology,
endpoint, proposal components, metadata, and replay ABI. It does not feed the estimator or alter
the normal render.

Debug view 16 is the opt-in path-temporal merge reference. It replays the previous same-pixel
reservoir on the current queue, applies strict topology/depth/transport/footprint admission, caps
the historical effective count at eight, and uses an identity shift with Jacobian one. Green means
history was selected; darker green means the merge was accepted but the current sample remained
selected; cyan is replay rejection; blue and purple are compatibility and footprint rejection;
gray is empty history; black is an empty current/history pair. The merged reservoir is stored only
for this debug history chain and is not consumed by the active image estimator.

The next spatial-reuse reference remains CPU-only while replay plumbing exposes the geometric and
directional-PDF terms associated with the newly stored reconnection vertex.
It admits a neighbor only when material, normal, relative depth, path topology/depth/transport,
and footprint agree. Its reconnection Jacobian is the solid-angle geometry ratio followed by the
receiver/source directional-PDF ratio in primary-sample space; a random-replay segment has unit
Jacobian. Do not add a GPU neighbor merge by copying the direct-light pass while these terms are
absent from `PathReservoir`.

Debug view 19 evaluates these terms for strict compatible continuous pairs only. Green means a
finite positive primary-sample-space Jacobian; blue is strict compatibility rejection, magenta is
a missing current edge, yellow is a delta/unsupported event, red is invalid geometry, cyan is an
invalid directional PDF, orange is invalid technique mass, and gray is an invalid Jacobian. It is
diagnostic-only: shifted visibility/radiance are not traced and the estimator is unchanged.

`gRestirPositionMaterial.w` stores an exactly representable 24-bit integer: the low two bits are the
transport material model and the upper 22 bits are the stable material-registry key. Direct ReSTIR
decodes only the model; strict path-spatial diagnostics compare the full packed identity. The path
topology hash also includes the registry key, so unrelated opaque materials cannot pass merely because
both use model zero.

Debug view 20 is the first RT shifted-radiance boundary. It repeats the strict view-19 admission
and Jacobian checks for diffuse first-edge events, replaces the source's stored exact first-event
throughput with the receiver's stored exact factor, and traces receiver-to-second-hit transmittance through the production
shadow SBT. A successful pixel shows the resulting HDR shifted canonical radiance. Purple is an
occluded shifted edge; dark green is a valid visible zero target; the view-19 reject colors remain
unchanged. Pink marks a source radiance/first-throughput support mismatch, mint marks invalid arithmetic, and
white marks a finite shifted value outside the R16F debug-output range. Glossy, delta, and transmission events are
yellow because the guide ABI does not yet expose enough F0/metalness data for an exact glossy shift.
Water and dielectric receivers are yellow as well: the primary pass has already consumed their
interface before the canonical path starts, so their guide and stored first event are different vertices.
View 20 captures only endpoints at hit depth 1 or later; a hit-0 emissive contribution precedes the
reconnection edge and cannot be divided by its first-event throughput.
The view also writes an exact 15-state histogram plus a hash-subsampled, 4096-entry bounded
`float2(shifted luminance, source luminance)` sample set into host-visible diagnostic buffers. The
same samples carry receiver PDF/PSS Jacobian, source-final/merge weight, and current-weight-sum/
selection-probability pairs in four contiguous segments. For finite accepted pairs, the shader also
performs a deterministic diagnostic Bernoulli draw using that probability; the host logs the number
of eligible pairs and source selections. The CPU reads them once every 60 frames after `waitIdle`
and logs state percentages plus finite sample percentiles. These atomics and readback waits are
active only in view 20.
View 20 also performs the first non-persistent scratch merge for finite accepted pairs. After
temporal admission has consumed the previous slot, the pass writes the merged weight sum, effective
count, selected target, and final weight into that otherwise unused slot. The normal current slot is
still committed as history, so the scratch record is overwritten on the next frame and cannot affect
temporal reuse or normal rendering. A source-selected scratch record retains the original source
seeds/states and carries the ABI-10 one-hop diffuse-reconnection descriptor together with the receiver
PDF/throughput. Generic identity replay rejects that mapping kind, and view 20 refuses an already
mapped record as another spatial source. Receiver-aware replay is mandatory before any mapped
record may be committed. The following same-frame validation proves that replay boundary, but it
does not make the scratch slot persistent; cross-frame reuse remains blocked by the source-root
requirements described below.
The pass does not update committed reservoirs, history weights, or the active estimator.
The shader-independent `DiffuseMappingReplay` reference defines that boundary: replay ABI,
mapping kind, and first-edge event must match; receiver PDF and PSS Jacobian must be finite and
positive; replayed source RGB is divided by its exact source throughput, multiplied by the stored
receiver throughput and freshly traced RGB transmittance, and compared with the stored shifted RGB
using the same relative tolerance as GPU seeded replay.

View 20 now dispatches that receiver-aware validation as a second same-frame ray-generation pass
after the scratch merge. It replays the original source queue chain from the stored seeds, checks
the source topology and proposal state, reconstructs the receiver-side geometry, directional PDF,
PSS Jacobian and first-event throughput, traces receiver-to-second-hit visibility again, and compares
the resulting shifted RGB/target with the scratch record. The pass writes only nine terminal
counters: eligible, accepted, and seven mutually exclusive reject categories. A valid frame has
`eligible == scratchSelected` and the sum of accepted plus all rejects equals eligible; the expected
steady-state result is near-total acceptance, while isolated finite-density/PDF outliers may be
rejected rather than admitted by weakening replay tolerances. ABI, source-state, receiver,
visibility, or radiance rejects require investigation. The diagnostic image is unchanged by this
validation dispatch. Mapped records still do not enter committed history or the normal estimator.

For a later frame, replaying only the receiver is insufficient: the mapped record's original
spatial source remains its canonical queue root. The CPU `PersistentDiffuseRemap` contract therefore
requires receiver reprojection, a stable/reprojected source queue root, and exact source replay.
It recomputes a direct source-to-current-receiver Jacobian and explicitly does not multiply by the
stored previous-receiver Jacobian. The current GPU history has receiver surface validation but no
separate motion mapping for that spatial source root, so persistence remains disabled.

## Linux

Set `DLSS_SDK` and `VULKAN_SDK` before configuring CMake:

```bash
export DLSS_SDK=/path/to/dlss-sdk
export VULKAN_SDK=/path/to/vulkan-sdk
```

`DLSS_SDK` must contain the NGX headers and static library. `VULKAN_SDK` must
contain Vulkan headers.

Then configure and build the native shim:

```bash
cmake -S native/ngx_shim -B build/cmake/ngx_shim/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/cmake/ngx_shim/release
```

On NixOS, enter the development shell from `flake.nix` instead of setting up
the toolchain by hand:

```bash
nix develop
cmake -S native/ngx_shim -B build/cmake/ngx_shim/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/cmake/ngx_shim/release
```

## Native Bundling

Gradle bundles NGX natives for the current host platform by default:

```bash
./gradlew build
```

Release builds that already have both platform shims available can request a
cross-platform native bundle:

```bash
./gradlew build -PngxPlatforms=windows-x64,linux-x64
```

Run the Vulkan RT/DLSS-RR client with:

```bash
JAVA_TOOL_OPTIONS='-Xmx8G -XX:+UseCompactObjectHeaders -XX:+AlwaysPreTouch -XX:+UseStringDeduplication -XX:+UseZGC' nvidia-offload ./gradlew runClient --args='--renderDebugLabels --graphicsBackend VULKAN'
```
