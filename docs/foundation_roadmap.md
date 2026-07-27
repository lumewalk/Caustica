# Caustica Foundation Roadmap

This fork is an experimental learning project focused on finding out how far
Minecraft's Vulkan ray-tracing backend can be taken without sacrificing a
reliable renderer foundation.

## Reference System

- Windows 11
- NVIDIA GeForce RTX 5060 Ti, 16 GB VRAM
- 40 GB system RAM
- HDR-capable display

The reference system is the primary development and visual-validation target.
Portable Vulkan behavior remains a design requirement even when a feature is
only available through NVIDIA NGX.

## Development Principles

1. Measure before optimizing.
2. Keep vanilla rendering as a safe fallback.
3. Treat Vulkan objects and native feature handles as device-session state.
4. Make renderer shutdown and restart deterministic before adding complexity.
5. Validate shader ABI at build time and Vulkan behavior at runtime.
6. Land changes in small, testable increments that can be compared with
   upstream.

## Milestones

### 1. Foundation

- Harden renderer, Vulkan, worker, and NGX lifecycle handling.
- Support clean RT stop/start within one JVM session.
- Add CPU-only tests that do not require the proprietary DLSS SDK.
- Add structured capability, crash, frame-time, and memory diagnostics.
- Establish repeatable reference scenes and performance captures.

### 2. Performance

- Profile CPU extraction, terrain streaming, BLAS/TLAS work, and GPU stages.
- Reduce allocation, upload, descriptor, and acceleration-structure churn.
- Add frame-budget-aware quality controls.
- Tune shader payload size, divergence, SER behavior, and direct-light RIS.

### 3. Rendering Quality

- Improve temporal stability, materials, exposure, and HDR presentation.
- Complete missing sky, weather, fog, cloud, Nether, and End cases.
- Add LOD and distant-scene representation.

### 4. New Techniques

- Add a non-NGX denoising/upscaling path.
- Evaluate FSR and cross-vendor support.
- Prototype temporal and spatial ReSTIR reuse.
- Evaluate adaptive sampling, radiance caching, and ReSTIR GI only after
  stable performance baselines exist.

## Upstream Workflow

- `upstream/main` tracks `ComfyFluffy/Caustica`.
- `origin/main` tracks the personal fork.
- Work is developed on focused branches, beginning with `foundation`.
- Changes that are generally useful and sufficiently isolated can be proposed
  upstream separately.
