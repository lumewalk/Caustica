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
The current eleven-lane path record is 176 B/pixel/slot; its proposal lane keeps
light-selection, canonical continuation, and roulette PDF products separate from
the shift Jacobian. Its packed metadata includes an explicit canonical-endpoint
validity bit, one lane stores a replayable sky/emissive endpoint, and the final
three lanes store the selected canonical path's second-hit reconnection vertex,
oriented normal, continuous source-edge proposal density, and exact local RGB throughput
of the selected first event. The source PDF is zero for
delta/refraction transitions because those are discrete measures and are not yet eligible
for a spatial shift.

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
`RtPathReplayReference` test is the authority for this admission rule. Debug view 15 performs an
opt-in seeded replay of the selected reservoir path using the stored seed pairs and compares the
reconstructed terminal states, topology, endpoint, proposal lane, and packed metadata. Its green
output means an exact replay; red/orange/cyan/blue/magenta/yellow channels identify path-state,
proposal-state, ABI, topology/metadata, endpoint, and proposal-lane mismatches respectively.
Debug view 16 is the first opt-in GRIS temporal merge. It uses identity temporal mapping, replays
the historical sample on the current queue, requires an exact replay plus strict
depth/transport/topology/footprint compatibility, caps the source effective count at eight, and
uses an explicit unit shift Jacobian. Green means the historical sample was selected, darker green
means the merge was accepted while the current sample remained selected, cyan is replay rejection,
blue/purple are compatibility/footprint rejection, gray is empty history, and black means both
reservoirs are empty. This mode updates only the path-history reference reservoir; it does not feed
the active image estimator.
A future reconnection
mapping may relax the identity checks only together with a measured Jacobian/PDF mapping and
new reference tests; it must not silently treat the two RNG streams as one seed.

PDF capture is the current contract boundary. `RtPathPdfReference` treats technique selection,
continuous directional density, and delta mass as separate events.
`RtPathCanonicalProposalReference` composes them into the same continuation and roulette products
stored by the GPU and independently mirrors the proposal lane, event counters, transport mask,
float clamping, and packed-bit reinterpretation. Delta reflection/transmission use discrete
technique mass; diffuse and glossy VNDF events use directional density; roulette uses the mass of
the observed outcome; and a sky/emissive endpoint contributes unit measure while setting only the
canonical-valid bit. The active path estimator still keeps its bootstrap proposal density because
one candidate currently sums NEE and continuation radiance rather than representing one canonical
path.
Debug view 13 opts into a storage-only canonical candidate capture: candidates without a recorded
sky/emissive endpoint are skipped, and the continuation/roulette PDF product becomes the first
path-only proposal-density check. Normal rendering and debug view 14 remain on the bootstrap
capture until this mode has its own image and temporal comparisons. The view now exposes
per-sample admission diagnostics without changing the `PathReservoir` ABI: green is admitted,
magenta is a missing endpoint, blue is a valid zero-target endpoint, red is invalid radiance,
cyan is an invalid continuation PDF, orange is an invalid roulette PDF, and yellow is an invalid
combined proposal/importance weight. Colors are averaged across SPP so the view also shows the
local admission ratio. Canonical candidates are rejected before PDF division, keeping invalid
values out of the reservoir; normal rendering and view 14 do not consume this diagnostic path.

Spatial path reuse has a separate reference boundary. Neighbor admission reuses the direct-light
surface policy (material, normal, relative depth) and additionally requires path depth, topology,
transport class, and bounded footprint. The 176-byte record stores the selected canonical path's
second-hit reconnection vertex, normal, source directional proposal density, and exact first-event
RGB throughput. These lanes were the prerequisites for the receiver-side material/PDF evaluation,
shifted target, and non-persistent replay boundary described below; they do not by themselves make
a record safe for persistent reuse.
The CPU reference therefore records the solid-angle Jacobian
`|cos(theta_receiver) / cos(theta_source)| * d_source^2 / d_receiver^2` and the primary-sample
form multiplies it by `p_receiver / p_source`; random replay contributes unit Jacobian. This
keeps spatial admission testable without silently introducing a biased neighbor merge.

The receiver position guide packs the 2-bit transport material model together with a stable 22-bit
material-registry key into its exactly representable 24-bit integer range. Direct-light passes decode
only the model, while strict path-spatial admission compares the full packed identity and the path
topology hash includes the same registry key. This prevents unrelated opaque blocks from being
treated as one material merely because they all use `MATERIAL_OPAQUE`.

Debug view 19 is the next opt-in diagnostic boundary. It stores the first-edge event kind in the
packed reconnection metadata (introduced before the current replay ABI 10), then checks strict neighboring reservoirs for a
continuous diffuse/glossy event with matching topology/depth/transport/footprint. For a selected
pair it evaluates the receiver/source geometric solid-angle ratio and reconstructs the receiver
directional density from the current edge and its shifted edge; the receiver/source PDF ratio
produces a finite primary-sample-space Jacobian. Green means all terms are finite and positive;
blue is strict compatibility rejection; magenta means the current edge is unavailable; yellow
means a delta/unsupported event; red is invalid geometry; cyan is invalid directional density;
orange is invalid technique mass; gray is an invalid Jacobian. View 19 does not trace shifted
visibility or radiance and never changes the estimator or the existing view 17/18 counters.

Debug view 20 adds the first separate ray-generation shifted evaluation. It intentionally accepts
only strict-compatible diffuse pairs. The path record stores the exact selected local first-event RGB
throughput, so the pass can remove the source factor, apply the receiver factor without rebuilding
selected-path state from a quantized guide buffer, and trace the new
receiver-to-second-hit transmittance through the production shadow SBT. Successful pixels display
the resulting HDR shifted canonical radiance; occlusion, spectral-support mismatch, invalid arithmetic,
and debug-output overflow remain separate diagnostic states. Glossy remains rejected until
receiver/source F0 and metalness are available. Water and dielectric receivers are also rejected:
the primary pass consumes those interfaces before the stored canonical wavefront path starts, so
their guide receiver is not the path's first event. View 20 also skips hit-0 emissive endpoints:
a second-hit reconnection may only carry an endpoint at hit depth 1 or later, after the stored first
event throughput has actually entered the contribution. The Jacobian
is revalidated but is not folded into the target, and no reservoir weight or production radiance is
changed. A view-20-only host-visible readback records all 15 mutually exclusive terminal states and
a bounded 4096-sample record. Each accepted sample keeps the shifted/source target pair in the first
segment of the existing buffer, receiver directional PDF/PSS-Jacobian in the second segment,
source-final-weight/spatial-merge-weight in the third segment, and the current reservoir weight sum
plus hypothetical neighbor-selection probability in the fourth segment. For each finite accepted
pair the shader also performs a deterministic Bernoulli draw using that probability and records
eligible/source-selected counts. This distinguishes
genuinely bright finite shifted radiance from the explicit R16F-overflow state and checks the
target/PDF/weight/selection chain without changing normal rendering or the estimator.

`SpatialGrisWeight` evaluates
`shiftedTargetDensity * sourceFinalWeight * min(sourceM, maxSourceM) * primarySampleJacobian`.
The PSS Jacobian is part of the reservoir merge weight and is never applied a second time to the
shifted radiance. This contract is tested on the CPU, including empty/zero-target, source-count
clamping, invalid-term, and overflow cases. The CPU reference mirrors the receiver-side diffuse
density sequence used by view 20:
current/source directional density, technique mass, shifted receiver PDF, and the resulting PSS
Jacobian. Invalid or non-positive terms are rejected before they can form a GRIS weight.
The provisional tail policy is `finite-unclamped`: every finite positive GRIS weight remains
eligible, while only the existing source-count limit is applied. No Jacobian or merge-weight clamp
is introduced. View 20 measures the resulting hypothetical selection probability as
`mergeWeight / (currentWeightSum + mergeWeight)` so decisions can be based on the tail's actual
effect on reservoir selection rather than on isolated absolute maxima. Runtime validation of this
distribution and the stochastic selection gate is complete.

View 20 writes an opt-in diagnostic merge into a dedicated lazy mapped-snapshot buffer. It updates
the accumulated weight, effective M, selected shifted target, and final W, but the frame still
commits only the untouched current candidate slot. Replay ABI 10 uses the reserved
`reconnectionThroughput.w` bits as a compact mapping
control: identity is zero and a one-hop diffuse reconnection is one. A source-selected snapshot record
retains the original source seeds/states, stores receiver PDF/throughput in the reconnection lanes,
and marks the diffuse mapping. Generic identity replay rejects mapped records, and an already mapped
record cannot be selected as another spatial source; this prevents silent mapping/Jacobian
composition. The CPU `DiffuseMappingReplay` reference formalizes receiver-aware reconstruction: it
validates ABI 10,
mapping kind, diffuse event, receiver PDF, and PSS Jacobian; recomputes shifted RGB from replayed
source radiance, exact source/receiver throughput, and newly traced transmittance; then compares the
result with the stored shifted RGB using the seeded-replay tolerance.
A dedicated view-20-only same-frame GPU pass mirrors that boundary. It replays the stored source
seed chain through the production path tracer, validates source state and receiver compatibility,
recomputes geometry/PDF/Jacobian/receiver throughput, re-traces shifted visibility, and compares the
reconstructed shifted sample with the diagnostic snapshot record. Mapping-replay counters expose
eligible/accepted records plus mutually exclusive ABI/source/receiver/geometry/PDF/visibility/
radiance rejects (and the retained-root reject introduced below). This pass
does not modify the diagnostic image, reservoirs, committed history, or the active estimator. Runtime
validation accepted 201419 of 201425 mapped records (99.997021%): every frame preserved exact
category accounting, all ABI/source/receiver/geometry/visibility/radiance rejects stayed zero, and
six isolated PDF-tail records were safely rejected without relaxing replay tolerance.

The source-lifetime sub-gate adds a lazy view-20-only retained-root sidecar without changing the
176-byte reservoir or replay ABI 10. Each 160-byte record stores source and receiver
position/material, normal/roughness, and both packed 48-byte source queue segments; positions are
camera-relative so terrain rebasing does not invalidate the root. Same-frame mapping replay consumes
this retained root instead of the transient current-frame queue. Extra counters expose root writes,
invalid captures, and retained-root replay rejection. Together the 160-byte root and 176-byte mapped
snapshot cost about 276.04 MiB at 1280x673, are absent from ordinary rendering, and are not committed
as path history.

The shader-independent `PersistentDiffuseRemap` reference defines the next history boundary. The
original source remains the canonical replay root, so a cross-frame remap requires valid receiver
reprojection, a stable source root, and exact source replay. View 20 now diagnoses that boundary with
a single persistent read-before-clear snapshot: receiver motion locates the previous mapped record,
while the retained source is independently projected into the current frame and a bounded 3x3 search
requires its motion vector to return to the exact previous source pixel. Both roots must preserve
position, normal, roughness, and full material identity after camera/terrain rebasing. This first
policy intentionally rejects object motion rather than guessing it.

For admitted static roots the original source is replayed, and the new source-to-current-receiver
PDF, Jacobian, current receiver throughput, visibility, and radiance are recomputed directly. The
previous receiver's Jacobian is integrity metadata and is never composed. Exclusive cross-frame
counters separate receiver reprojection, empty snapshot, ABI, source-root/source-reprojection/source-
replay, receiver, geometry, PDF, visibility, radiance, and accepted outcomes. After the cross-frame
read, the mapped snapshot is cleared and rewritten for the current frame; generation reset or a
one-frame view gap invalidates continuity. Mapped records remain outside committed path history and
the estimator until receiver-policy diagnostics pass and moving-surface motion has an explicit
contract.

Fresh Vulkan/RTX 5060 Ti validation at 1280x673 preserved both exclusive accounting invariants on
all 108 cross-frame readbacks, with zero ABI or retained-root rejects. During 90 settled-camera
readbacks, 40,803 of 342,717 eligible records were accepted (11.905741%). The dominant terminal
category was the combined strict receiver-path check (301,409); source reprojection rejected 16 and
exact source replay rejected 489, while geometry, PDF, visibility, and radiance rejects stayed zero.
The 15 moving-camera readbacks remained fail-closed (2 of 39,673 eligible accepted) and returned to
the settled acceptance regime after motion stopped. Same-frame replay independently accepted
382,448 of 382,485 records (99.990326%), with only three source-state and 34 finite PDF-tail rejects.
This proves the diagnostic lifetime/reprojection boundary, but not persistent estimator admission:
the receiver-policy checkpoint therefore splits that combined terminal counter into seven exclusive
causes: surface/material compatibility, usable current sample, first-edge availability, topology,
depth, transport, and footprint. Their sum is logged as `receiverTotal` so the original accounting
identity remains directly auditable. The order is mirrored by a shader-independent CPU reference;
admission rules and tolerances are unchanged. Runtime category ratios must guide a later policy A/B
decision before any mapped history write.

The first split-counter Vulkan run preserved all three accounting identities on 25/25 readbacks.
For 20 settled-camera readbacks, 8,594/76,551 eligible records were accepted (11.226503%). The
dominant exclusive rejection was `receiverSample`: 56,370 (73.637183% of eligible), followed by
topology 6,357, first edge 2,477, footprint 1,105, and surface 356; depth and transport stayed zero.
During five moving-camera readbacks, all 10,486 eligible records failed closed, with 10,457 surface
rejects and 29 source-reprojection rejects. ABI/root and downstream geometry/PDF/visibility/radiance
rejects remained zero. This localizes the next policy question. A shadow `sampleRescue[...]` A/B now
counts every surface-compatible zero-weight/zero-target current sample, then partitions it into edge,
topology, depth, transport, footprint, or fully rescued outcomes. The strict terminal category
remains `receiverSample`, and a fully rescued shadow record returns before geometry/radiance
evaluation, so it cannot be selected or persisted. The CPU reference mirrors the same ordered strict
and sample-relaxed decisions. Do not remove the positive-current-sample requirement until runtime
ratios and the estimator/history mathematics justify it.

Fresh Vulkan validation answered that A/B negatively. Across 20 settled readbacks,
`receiverSample` and `sampleRescue.eligible` both totaled 41,685; all 41,685 shadow records failed
the receiver-edge check, while rescued/topology/depth/transport/footprint were zero. All strict and
shadow accounting identities were exact per frame. Removing current-sample positivity would
therefore admit no additional record in this scene and is not justified. The next diagnostic must
split receiver-edge availability into valid/depth/event/mapping/PDF/finite-throughput causes and
decide whether persistent remapping needs a deterministic receiver-material substrate independent
of a positive current canonical endpoint. Mapped history remains disabled.

The follow-up `edgeBreakdown[...]` shadow diagnostic partitions every failed current receiver edge
into missing valid bit, wrong depth, unsupported event, mapping descriptor, non-positive PDF, or
non-finite vertex/throughput. Its population must equal strict `receiverEdge` plus
`sampleRescue.edge`; outcomes are exclusive and do not change either parent terminal decision. The
ordered policy is mirrored by the CPU reference.

The first edge-breakdown capture was exact on all 12 readbacks. Of 146,307 failed edges, 139,525
(95.364542%) were missing the valid bit and exactly matched the sample-rescue edge population; the
remaining 6,782 (4.635458%) were valid non-diffuse events and exactly matched strict
`receiverEdge`. Depth, mapping, PDF, and finite-state failures were zero. This is not metadata
corruption: a zero/empty current canonical endpoint does not publish receiver-edge state, while
non-diffuse events are intentionally outside the current mapping support. Before persistent history
can reuse through an empty current endpoint, define and validate a deterministic receiver-side
diffuse material/PDF/throughput substrate independent of canonical endpoint selection. First audit
whether the existing quantized guide is mathematically sufficient or whether an exact guide is
required; do not allocate or admit it by assumption.

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

The first implementation milestone is debug view 16: an identity-map temporal merge with a unit
Jacobian and strict seeded replay. It intentionally precedes motion-vector reprojection and
reconnection so the GRIS weight and effective-count accounting can be validated without hiding
mapping errors behind a more permissive shift.

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
