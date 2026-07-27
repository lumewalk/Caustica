# Caustica Foundation Roadmap

This fork is an experimental learning project focused on building a correct,
extensible real-time path-tracing pipeline for Minecraft's Vulkan backend. The
long-term target is GRIS-based ReSTIR path reuse, including the practical
improvements described by ReSTIR PT Enhanced. Performance work supports that
goal; it is not the goal by itself.

The detailed architecture and staged delivery plan live in
[ReSTIR PT Roadmap](restir_pt_roadmap.md).

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
7. Separate candidate generation, resampling, visibility, and shading so each
   stage can evolve independently.
8. Preserve deterministic primary visibility and DLSS Ray Reconstruction
   guides while experimenting with light-transport sampling.
9. Treat temporal history as explicit renderer state with documented ownership,
   validation, and reset rules.
10. Optimize only measured bottlenecks after the intended pipeline is correct.

## Milestones

### 1. Foundation

- Harden renderer, Vulkan, worker, and NGX lifecycle handling.
- Support clean RT stop/start within one JVM session.
- Add CPU-only tests that do not require the proprietary DLSS SDK.
- Add structured capability, crash, frame-time, and memory diagnostics.
- Establish repeatable reference scenes and performance captures.

### 2. Wavefront Pipeline

- Integrate upstream's primary/guide plus indirect wavefront split.
- Keep the current `foundation` branch as the stable comparison point.
- Add independent GPU timestamps for primary, indirect, and future resampling
  passes.
- Establish explicit surface, continuation, history, and reservoir data
  contracts.
- Preserve deterministic guide generation, lifecycle handling, and restart
  behavior.

### 3. ReSTIR DI Foundation

- Replace primary-surface block-emitter RIS with per-pixel reservoirs.
- Add initial candidate generation, temporal reuse, spatial reuse, and one
  final world-space visibility query.
- Validate history with motion, depth, normal, material, stable surface
  identity, and geometry epochs.
- Keep the existing RIS path as a reference and as the initial secondary-bounce
  fallback.
- Add reservoir and history debug views before quality tuning.

### 4. ReSTIR PT and Enhanced Reuse

- Generalize light reservoirs into GRIS-compatible path reservoirs.
- Add robust shift mappings, random replay/reconnection, and path-compatibility
  validation.
- Unify direct and global illumination reuse only after ReSTIR DI is stable.
- Evaluate ReSTIR PT Enhanced techniques such as reciprocal neighbor
  selection, footprint-based reconnection, duplication maps, and targeted
  disocclusion handling.
- Evaluate world-space persistent reservoirs or a radiance cache for diffuse
  off-screen history without replacing view-dependent screen-space reuse.

### 5. Rendering Quality

- Improve temporal stability, materials, reflections, exposure, and HDR
  presentation on top of the stable sampling pipeline.
- Complete missing sky, weather, fog, cloud, Nether, and End cases.
- Improve water, dielectrics, emissive materials, particles, and moving-entity
  lighting.
- Add LOD and distant-scene representation with stable temporal identities.

### 6. Targeted Performance and Portability

- Profile CPU extraction, terrain streaming, BLAS/TLAS work, and every GPU
  pass.
- Optimize only stages that exceed their measured frame or memory budget.
- Add frame-budget-aware quality controls without changing estimator
  correctness silently.
- Add a non-NGX denoising/upscaling path and evaluate FSR and cross-vendor
  support.

## Upstream Workflow

- `upstream/main` tracks `ComfyFluffy/Caustica`.
- `origin/main` tracks the personal fork.
- Work is developed on focused branches, beginning with `foundation`.
- Large wavefront/ReSTIR integration work starts on
  `integration/wavefront-restir`; it does not replace `foundation` until its
  correctness and visual gates pass.
- Changes that are generally useful and sufficiently isolated can be proposed
  upstream separately.
