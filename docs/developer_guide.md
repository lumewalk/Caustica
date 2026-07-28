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

The path-reservoir ABI is currently 128 bytes per pixel per slot. Its replay-control
lane stores the two wavefront segment seed pairs, segment count, replay version, and
terminal-state hashes; the additional proposal-components lane captures light,
continuation, and roulette PDF products plus event counters (including an explicit
canonical-endpoint validity bit), while the canonical radiance lane stores one
replayable sky/emissive endpoint. Two path-history slots
therefore use about 210.3 MiB at 1280x673 and 473.0 MiB at 1920x1009. This is
intentional: replay correctness is being established before any packing or compression pass.

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
