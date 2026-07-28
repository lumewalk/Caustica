# ReSTIR PT Roadmap

## Objective

Build a real-time path-tracing pipeline that can progress from ReSTIR DI to a
GRIS-based ReSTIR PT implementation and then adopt the practical improvements
from ReSTIR PT Enhanced.

The destination is path reuse, not a feature that merely carries the ReSTIR
name. A completed ReSTIR PT stage must reuse full light-transport samples with
correct contribution weights, compatibility tests, and documented shift
mappings. ReSTIR DI is the first production milestone because it establishes
the same history, reservoir, validation, and debugging disciplines with a much
smaller correctness surface.

The reference target is Windows 11 with an RTX 5060 Ti 16 GB. The design must
remain valid Vulkan and must not make NVIDIA NGX part of the sampling
algorithm. DLSS Ray Reconstruction is a consumer of noisy radiance and guide
buffers, not the owner of lighting history.

## Architecture Decision

Use a wavefront-oriented render graph with explicit pass boundaries. Upstream
[PR #29, `Split ray tracing into wavefront passes`](https://github.com/ComfyFluffy/Caustica/pull/29)
is the preferred starting point because it separates deterministic
primary/guide work from incoherent indirect transport. It is a substrate, not
the finished ReSTIR pipeline.

The intended pass sequence is:

```text
scene update and acceleration structures
    -> primary visibility / deterministic guides / path continuations
    -> initial direct-light candidates
    -> temporal reservoir resampling
    -> spatial reservoir resampling
    -> final world-space visibility and direct-light evaluation
    -> indirect wavefront transport
    -> radiance and guide handoff to DLSS Ray Reconstruction
    -> exposure / SDR or HDR mapping / UI / present
```

Screen-space describes reservoir ownership: one history entry is associated
with a reprojectable visible surface sample. It does not mean screen-space
visibility. Candidate lights and occluders are world-space scene data, and the
selected sample is validated with hardware ray tracing.

## Non-Negotiable Invariants

1. Candidate generation, resampling, visibility, and shading are independent
   operations with explicit inputs and outputs.
2. Every temporal resource has one owner, current/previous storage, a reset
   policy, and a visible debug state.
3. Primary visibility and DLSS RR guides are deterministic for a given frame.
   Stochastic lighting choices must not leak into depth, motion, normal, or
   material guides.
4. Temporal reuse requires motion, depth, normal, material compatibility,
   stable surface identity where available, and a geometry/content epoch.
5. World-space visibility is evaluated after sample selection. A cached
   visibility result may be reused only under an explicitly validated policy.
6. Reservoir math is implemented in a shader-independent reference model and
   covered by deterministic CPU tests before GPU integration.
7. Estimator-changing clamps, age limits, and bias controls are named,
   configurable, documented, and comparable against a reference mode.
8. A no-reuse reference path remains available for correctness and image
   comparisons.
9. Shader records and buffers have reflected or generated ABI validation;
   Java must not duplicate hand-calculated shader offsets.
10. Every pass has GPU timestamps, memory accounting, and an isolated debug
    label before performance claims are accepted.
11. Renderer reload, resize, world/dimension change, teleport, material reload,
    and device restart deterministically invalidate the affected history.
12. Optimization follows profiling. Architectural shortcuts are not accepted
    solely because they improve one reference scene.

## Data Contracts

Exact packing is deferred until the reference implementation is tested, but
the semantic records are fixed early so later ReSTIR PT work does not require
another renderer rewrite.

### Surface Sample

- world or reconstructable position
- geometric and shading normal
- depth
- diffuse albedo and specular/F0 data
- roughness, metalness, and material model
- motion/reprojection data
- stable surface identifier when available
- geometry and material epochs
- validity and disocclusion flags

Minecraft chunk remeshing can change triangle indices without changing the
logical block surface. Stable identity should therefore prefer a logical
world-space identity (dimension, block or section coordinate, face/material
identity) over a raw transient primitive index where practical.

### Direct-Light Sample and Reservoir

- stable light identifier and sampled point/direction
- emitted radiance and source geometry data needed to reconstruct the sample
- proposal and target density terms
- selected-sample contribution weight
- accumulated weight and effective/confidence count
- reservoir age and source frame
- visibility state only when its reuse policy is valid

Reservoir storage is double-buffered. The first implementation favors clarity
and validation over minimum byte size; packing follows only after captures
identify the real bandwidth and memory pressure.
The current eight-lane path record is 128 B/pixel/slot; its proposal lane keeps
light-selection, canonical continuation, and roulette PDF products separate from
the shift Jacobian, and the final lane stores one replayable sky/emissive endpoint.

### Path Sample and Reservoir

The ReSTIR PT milestone extends the reservoir from a light sample to a
canonical path representation:

- path vertices or a replayable compact path description
- random replay state
- sampling-technique identity
- unshadowed contribution and unified contribution weight
- reconnection vertex and footprint data
- shift-mapping Jacobian/density terms
- confidence/effective sample count
- topology and epoch data needed to reject invalid reuse

The current wavefront replay contract is deliberately stricter than a final GRIS shift:
replay version, segment count, transport RNG state, and light-proposal RNG state must all
match before a path is considered replay-compatible. The shader-independent
`RtPathReplayReference` test is the authority for this admission rule. A future reconnection
mapping may relax the identity checks only together with a measured Jacobian/PDF mapping and
new reference tests; it must not silently treat the two RNG streams as one seed.

PDF capture is the current contract boundary. `RtPathPdfReference` treats technique selection,
continuous directional density, and delta mass as separate events. The GPU now records those
components, but the active path estimator still keeps its bootstrap proposal density because one
candidate currently sums NEE and continuation radiance rather than representing one canonical path.

## Delivery Phases

### Phase 0 — Wavefront Integration Baseline

Create `integration/wavefront-restir` while leaving `foundation` untouched as
the known-good fallback.

- integrate upstream PR #29 or its merged successor
- preserve renderer/NGX lifecycle fixes and Windows tooling
- port the cubic sRGB texture decode to the new math module
- adapt non-blocking GPU timestamps to separate primary and indirect passes
- measure continuation-buffer VRAM explicitly
- validate water, dielectric stacks, reflected/refracted motion, HDR, and
  renderer restart

Exit gate:

- clean build, tests, shader compilation, and SPIR-V validation
- no visual regression in the reference world
- separate stable primary/indirect timings
- documented VRAM use at reference and native resolutions
- `foundation` remains runnable without history rewriting

### Phase 1 — Render Graph and History Substrate

Introduce no new lighting estimator yet.

- explicit pass/resource ownership and barriers
- double-buffered history allocation
- central history invalidation reasons
- surface compatibility helpers
- stable logical surface IDs and geometry/material epochs where possible
- debug views for surface ID, epoch, reprojection, compatibility, and
  disocclusion
- per-pass timestamps and memory counters

Exit gate:

- forced camera cuts, teleport, resize, reload, and dimension changes never
  consume stale history
- static-camera reprojection is stable
- moving entities and chunk remeshing reject or remap history intentionally

### Phase 2 — ReSTIR DI Initial Reservoirs

Move primary-surface block-emitter selection out of the indirect megakernel
logic without temporal or spatial reuse.

- implement CPU reference reservoir update/merge tests
- generate candidates from the existing global/local emitter hierarchy
- retain enough source data to reconstruct the selected sample
- evaluate exactly one final world-space visibility sample
- compare energy and noise against current RIS and a no-NEE reference
- keep current secondary-bounce RIS unchanged

Exit gate:

- reservoir math tests cover zero weight, extreme weight, merge, confidence,
  and deterministic RNG cases
- static images match the reference estimator within expected Monte Carlo
  variance
- reservoir debug view identifies the selected light and weight

### Phase 3 — Temporal Reuse

- reproject previous reservoirs with motion/depth
- validate normal, material, logical surface ID, and epochs
- clamp or cap history only through named quality controls
- track age, confidence, rejection reason, and disocclusion
- reset deterministically on all lifecycle events

Exit gate:

- no stale-light trails after moving lights, entities, or chunk updates
- camera motion reduces noise without persistent ghosting
- every rejected history sample has a diagnosable reason

### Phase 4 — Spatial Reuse

- ping-pong spatial reservoir pass
- compatibility-aware neighbor selection
- reject edges across depth, normal, material, and incompatible specular lobes
- measure covariance as well as raw variance
- evaluate reciprocal or compatibility-guided neighbor selection after the
  baseline is correct

Exit gate:

- stable quality improvement in interiors, emissive-heavy scenes, thin
  geometry, and camera motion
- no systematic light leaking across walls or dissimilar surfaces
- configurable radius/count with measured cost and diminishing-return curves

### Phase 5 — ReSTIR DI Hardening

- moving and animated emitters
- translucent/dielectric receivers
- water and submerged paths
- alpha-cutout foliage and particles
- emissive entities and block entities
- robust history behavior under streaming and material reloads
- automated captures of debug views and reference scenes

Exit gate:

- ReSTIR DI is the default primary direct-light sampler
- existing RIS remains a secondary-bounce/reference fallback
- quality and failure modes are documented before path reuse begins

### Phase 6 — GRIS and ReSTIR PT Prototype

- generalize reservoirs to path samples and unified contribution weights
- add a shader-independent GRIS reference implementation and tests
- define random replay and reconnection shift mappings
- reuse diffuse and specular paths only in compatibility domains already
  covered by tests
- keep direct-light-only and no-reuse modes for comparisons

Exit gate:

- path reuse is mathematically and visually validated on controlled scenes
- diffuse, glossy, emissive, and dielectric cases have explicit policies
- correlation, bias controls, disocclusion, and failure cases are measurable

### Phase 7 — ReSTIR PT Enhanced Direction

Adopt Enhanced techniques incrementally rather than as one opaque rewrite:

- reciprocal neighbor selection to reduce spatial reuse cost
- footprint-based reconnection criteria
- duplication maps or equivalent correlation control
- unified direct and global illumination reservoirs
- color-noise and disocclusion-noise handling
- compatibility-guided neighbor selection
- multi-layer or splatted temporal reuse only where the baseline shows a real
  disocclusion problem

Exit gate:

- each technique independently improves the quality/time/correlation frontier
- the combined mode remains debuggable and has a reference fallback
- the implementation is described as ReSTIR PT Enhanced-inspired until it
  satisfies the relevant estimator and path-reuse contracts

### Phase 8 — Optional World-Space Persistence

Evaluate a world-space reservoir or radiance cache for diffuse off-screen
history. Minecraft's block grid is favorable for stable cell addressing, but
view-dependent glossy/specular reuse remains screen-space/path-space work.

This phase is accepted only if it improves rapid camera motion, disocclusion,
or diffuse convergence enough to justify memory, lookup, and invalidation
complexity.

### Phase 9 — Visual Development

With the sampling pipeline stable:

- materials and LabPBR fidelity
- water, ice, stained dielectrics, absorption, and caustics
- reflection quality and temporal stability
- emissive materials and moving lights
- atmosphere, fog, weather, clouds, Nether, and End
- particles, entities, animation, and first-person content
- HDR presentation and exposure
- LOD/distant geometry with stable temporal identity

## Measurement and Review Gates

Every phase records:

- reference commit and configuration
- render/display resolution and SPP
- per-pass GPU average, median, p95, and p99
- CPU frame and extraction metrics
- allocated and peak VRAM where observable
- static and moving-camera captures
- history rejection statistics
- known estimator bias and quality controls

Performance is a gate against accidental regressions, not the primary feature
target. A slower but structurally correct milestone may be retained on the
integration branch while it is being completed, but it does not replace
`foundation` until the agreed correctness, stability, and practical frame
budget gates pass.

## Primary References

- Laine, Karras, Aila — [*Megakernels Considered Harmful: Wavefront Path
  Tracing on GPUs*](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf)
  (2013)
- Bitterli et al. — [*Spatiotemporal Reservoir Resampling for Real-Time Ray
  Tracing with Dynamic Direct Lighting*](https://research.nvidia.com/publication/2020-07_spatiotemporal-reservoir-resampling-real-time-ray-tracing-dynamic-direct)
  (2020)
- Ouyang et al. — [*ReSTIR GI: Path Resampling for Real-Time Path
  Tracing*](https://diglib.eg.org/items/ae55c04f-4832-48af-b60a-95fecd62d0ce)
  (2021)
- Lin et al. — [*Generalized Resampled Importance Sampling: Foundations of
  ReSTIR*](https://research.nvidia.com/publication/2022-07_generalized-resampled-importance-sampling-foundations-restir)
  (2022)
- Lin, Kettunen, Wyman — [*ReSTIR PT Enhanced: Algorithmic Advances for Faster
  and More Robust ReSTIR Path Tracing*](https://research.nvidia.com/labs/rtr/publication/lin2026restirptenhanced/)
  (2026)

These papers define the target vocabulary and estimator lineage. The codebase
must document deliberate deviations rather than using the names as broad
labels for unrelated temporal filtering.
