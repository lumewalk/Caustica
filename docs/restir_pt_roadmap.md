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
invalid captures, and retained-root replay rejection. Together the 160-byte root, 176-byte mapped
snapshot, and 32-byte exact receiver sidecar cost about 302.32 MiB at 1280x673, are absent from
ordinary rendering, and are not committed
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

The receiver-material audit proves the existing cache is insufficient for exact diffuse remapping.
`gRestirAlbedoSss` stores deterministic `diffAlb` in RGBA16F and roughness is available separately,
but the sampled diffuse event uses `eventThroughput = diffAlb / (1 - ps)` and directional density
`(1 - ps) * |cos| / pi`. The lobe probability `ps` also depends on per-hit F0, which no receiver
guide stores; material identity alone cannot reconstruct texture-evaluated F0. A shader-independent
counterexample gives identical current guide terms but different technique mass/throughput for two
F0 values. The exact view-20 sidecar now uses two float4 lanes (32 B/pixel, about 26.29 MiB at
1280x673) alongside its source-root/mapped snapshot buffers. Primary visibility writes exact RGB
diffuse albedo plus diffuse technique mass deterministically before endpoint selection, with opaque
and particle rules matching the production path sampler. Pass A clears the second lane and Pass B
writes the exact biased outgoing-ray origin for the unsplit camera receiver before lobe or endpoint
selection. Mapping-replay dispatches are forbidden from overwriting it, and its BDA is zero outside
view 20.

The first consumer is comparison-only. Cross-frame replay counts surface-compatible receivers as
invalid guide, no stored positive diffuse edge, or stored-edge eligible, then compares eligible
records in the fixed order technique mass, throughput, and directional PDF. Exact accounting must
hold for both partitions. The guide is not yet an admission substitute or a remapping input, and it
does not enter committed history or the estimator until runtime comparison proves the contract.

Runtime comparison proved the material half of the contract but not the outgoing-direction half.
Across 29 stable readbacks, all 106191 stored-edge comparisons had a valid guide, throughput mismatch
was zero, technique-mass mismatch was 299 (0.2816%), and directional-PDF mismatch was 25180
(23.7120%). Both counter partitions were exact in every readback. A one-second camera move reduced
the eligible population fail-closed and stationary counts recovered on the next readback. The next
gate is therefore an exact replayable receiver outgoing direction/density, independent of endpoint
selection; persistent reuse remains forbidden.

The first outgoing-density correction is comparison-only. The prior check combined a selected
Pass-B second-hit vertex with an origin reconstructed from the separately traced Pass-A guide hit.
It now uses the exact Pass-B biased origin from the sidecar. Validity is evaluated before
stored-edge availability, so a `noStoredEdge` result still proves that exact material/origin state
was produced independently of endpoint selection. This does not add a sampled path, change
reservoir ABI/replay ABI, or authorize remapping/history; runtime must first demonstrate exact PDF
reconstruction and counter accounting.

Runtime exact-origin validation passed. Fifteen stationary readbacks contained 207213 attempted
receivers with zero invalid material/origin records; 164116 of them had no canonical edge, proving
that the sidecar is endpoint-independent. The 43097 stored-edge comparisons produced 42899 accepted,
175 technique-mass rejects, zero throughput rejects, and 23 PDF rejects (0.053368%, reduced from
23.7120%). Both partitions were exact in every frame. The strict tolerance remains unchanged and
the finite residual tails fail closed. Before any persistent admission, define a shader-independent
guide-only remap policy and a counter-only A/B over the no-edge population; do not silently remove
the existing topology/depth/transport/footprint gates.

The guide-only remap policy is now defined as an ordered pre-visibility chain: reconnection geometry,
receiver directional PDF/PSS Jacobian, then spectral throughput. View 20 evaluates that chain only
for valid-guide `noStoredEdge` records and publishes mutually exclusive `ready`, `geometry`, `pdf`,
and `throughput` counters. Runtime must prove both `guideRemap.eligible = receiverGuide.noStoredEdge`
and `guideRemap.eligible = ready + geometry + pdf + throughput`. This remains a counter-only shadow
experiment: it emits no visibility ray and cannot bypass strict admission, modify a reservoir or
history, change weights, or feed the estimator. A `ready` result therefore establishes mathematical
term availability, not permission for persistent reuse.

Runtime proved the entire pre-visibility partition: 23 stable readbacks contained 419559 eligible
no-edge records, all 419559 were `ready`, and geometry/PDF/throughput rejects were zero. Both counter
identities were exact in every readback. The next bounded gate is a view-20-only visibility shadow
audit for the ready population. It must remain observational and cannot turn ready records into
persistent mapped history or estimator inputs.

That bounded visibility audit now reuses the production shadow SBT from the exact stored Pass-B
receiver origin. It partitions every pre-visibility-ready record into clear, tinted, occluded, or
invalid transmittance. Required identities are `guideVisibility.eligible = guideRemap.ready` and
`guideVisibility.eligible = clear + tinted + occluded + invalid`. The traced result remains
counter-only: no shifted-radiance output, reservoir/history write, selection, weighting, strict-gate
relaxation, or estimator contribution is permitted.

Runtime passed this boundary across 26 stable readbacks. All 395053 ready records were accounted for:
394614 clear (99.8889%), 439 occluded (0.1111%), zero tinted in this scene, and zero invalid. Both
identities were exact in every frame. The next bounded gate may apply the returned RGB transmittance
to the already proven throughput ratio and audit the resulting shifted radiance/target with counters
only; it still cannot authorize persistent history or estimator use.

A targeted glass/water follow-up completed RGB-transmittance coverage: 48 readbacks contained
952802 eligible records, with 952446 clear, 7 tinted across five frames, 349 occluded, and zero
invalid. Both counter identities remained exact in every frame. The visibility gate is therefore
runtime-proven for clear, tinted, and occluded terminal paths.

The post-visibility target audit now multiplies the proven guide-only shifted radiance by every
valid production transmittance and evaluates the canonical luminance target. It partitions results
into positive, zero, or invalid and requires
`guideTarget.eligible = guideVisibility.clear + tinted + occluded` plus
`guideTarget.eligible = positive + zero + invalid`. This is still a counter-only view-20 experiment;
it cannot write a reservoir/history record, select or weight a sample, relax admission, or feed the
estimator.

Runtime passed this gate across 77 readbacks and 411047 eligible records: 406266 positive, 4781 zero,
and zero invalid. The corresponding visibility totals were 406131 clear, 135 tinted, and 4781
occluded. Both required partitions were exact in every frame, as were the stronger observed
equalities `positive = clear + tinted` and `zero = occluded`. The next bounded experiment may form
the GRIS candidate/merge-weight terms from the proven target, source count/final weight, and PSS
Jacobian, but must remain counter-only and non-persistent.

The counter-only GRIS readiness audit now forms exactly
`shiftedTarget * sourceFinalWeight * min(sourceEffectiveCount, 8) * pssJacobian` for every valid
guide target. It partitions the result into positive, zero, or invalid and requires
`guideWeight.eligible = guideTarget.positive + guideTarget.zero` plus
`guideWeight.eligible = positive + zero + invalid`. The PSS Jacobian is applied exactly once; the
audit does not form a selection probability or current weight sum and cannot mutate scratch,
reservoir history, persistent mapping, or the estimator. Runtime must prove this arithmetic boundary
before any stochastic merge experiment is considered.

Runtime passed the merge-weight boundary across 52 readbacks and 367160 eligible records: 362983
positive, 4177 zero, and zero invalid. Both required identities were exact in every frame. The
stronger observed equalities `weight positive = target positive` and
`weight zero = target zero` were also exact, proving that the source final weight, capped source
count, and single PSS Jacobian introduced no new zero or invalid arithmetic in this population.
Selection and all reservoir/history mutation remain disabled.

The next shadow gate evaluates selection readiness without performing selection. For every valid
guide merge weight it validates the current reservoir weight sum, computes
`mergeWeight / min(currentWeightSum + mergeWeight, 1e30)`, and partitions the result into positive,
zero, invalid-current-weight, probability-above-one, or arithmetic reject. Required identities are
`guideSelection.eligible = guideWeight.positive + guideWeight.zero` and
`guideSelection.eligible = positive + zero + current + high + arithmetic`. It consumes no random number,
does not copy a source sample, and does not update weight sum, M, final W, scratch, persistent
history, or the estimator.

Runtime proved exact accounting but rejected the capped-denominator formula as a complete selection
contract. Across 68 readbacks and 393809 eligible records, 379657 probabilities were positive, 5677
were valid zero, 8475 exceeded one, and both invalid-current-weight and general arithmetic rejects
were zero. Both required identities were exact in every frame, and every zero probability matched a
zero merge weight. Because all non-general rejects are mathematically equivalent here to
`mergeWeight > 1e30`, silently clamping the probability or admitting those samples would hide an
estimator-changing bias. The next bounded gate must define and audit an overflow-stable selection
ratio from the uncapped relative weights while keeping stored weight-sum capping and all reservoir
mutation disabled.

That bounded A/B now evaluates the mathematically equivalent uncapped ratio without first adding
the two weights: when the candidate is larger it uses
`1 / (1 + currentWeightSum / mergeWeight)`, otherwise it uses
`ratio / (1 + ratio)` with `ratio = mergeWeight / currentWeightSum`. This avoids both overflow and
the rejected `1e30` denominator cap without clamping the probability. The new counter-only partition
requires `guideStableSelection.eligible = positive + zero + invalid`; runtime should additionally
show `stable positive = old positive + old high`, `stable zero = old zero`, and zero invalid values.
Stored weight-sum capping, RNG, selection, reservoir/history writes, and the estimator remain
unchanged until this comparison passes.

Runtime passed the stable-ratio boundary across 45 readbacks and 261882 eligible records. The stable
partition contained 258020 positive, 3862 zero, and zero invalid probabilities. On the identical
population the rejected capped formula produced 252546 positive, 3862 zero, 5474 probability-high,
and zero current/arithmetic rejects. Every frame preserved exact terminal accounting, identical
eligibility and zero populations, and `stable positive = old positive + old high`. The uncapped
relative ratio is therefore accepted as the probability contract; RNG, sample selection, stored
weight-sum policy, scratch/persistent history, and the estimator are still unchanged. Any Bernoulli
selection experiment must be introduced as its own counter-only gate.

The next counter-only gate performs that Bernoulli comparison without touching either canonical
replay RNG stream. A local PCG hash of pixel and frame identity produces an exact `[0,1)` float,
which is compared with the proven stable probability. The required identities are
`guideStableBernoulli.eligible = guideStableSelection.eligible` and
`eligible = selected + retained + invalid`. These counters do not copy the source sample or update
weight sum, M, final W, scratch/persistent history, strict admission, or the estimator.

Runtime passed this Bernoulli boundary across 73 readbacks and 513545 eligible records: 495376 were
selected, 18169 retained, and none invalid. The identical stable-probability population contained
507819 positive and 5726 zero records, with zero invalid. Every frame preserved identical
eligibility, exact `selected + retained + invalid` accounting, `selected <= positive`, and
`retained >= zero`; 97.5497% of positive-probability records selected the source. The local draw is
therefore accepted for diagnostic selection. The next bounded gate may audit post-selection stored
weight/M/final-W arithmetic and selected/retained sample readiness, but still must not write a
reservoir/history record or affect the estimator.

That post-selection gate now shadows the future stored arithmetic in registers only. It keeps the
existing caps of `1e30` for weight sum and `16777216` for effective M, chooses either the shifted
source target or retained current target using the proven stable Bernoulli result, and forms final
`W = cappedWeightSum / (cappedM * selectedTarget)`. Its mutually exclusive counter partition is
`eligible = selectedReady + retainedReady + empty + sampleReject + arithmeticReject`; a zero final
weight sum is an intentional empty no-op, while every non-empty ready result requires finite positive
M, selected target, denominator, and final W. The shadow gate copies no sample metadata, writes no
scratch or persistent reservoir/history state, and does not change admission or the estimator.

Runtime passed this boundary across 38 readbacks and 268384 eligible records. All three upstream
eligibility totals matched exactly. The stable Bernoulli selected 257389 source samples and retained
10995 current samples; post-selection classified exactly 257389 as selected-ready, 7292 as
retained-ready, and 3703 as valid empty no-ops. Sample and arithmetic rejects were both zero, as were
stable-probability and Bernoulli invalids. Every frame preserved the full terminal partition,
selected-ready equalled Bernoulli-selected, and the remaining post-selection outcomes equalled
Bernoulli-retained. This proves the counter-only arithmetic/readiness contract, but still does not
authorize copying sample metadata or writing scratch/persistent history.

The next bounded gate copies a complete future reservoir sample into a shader-local register value.
Selected records use shifted radiance/target, retain the original canonical source replay metadata,
and replace only the receiver-dependent directional PDF, throughput, reconnection geometry, mapping
control and current-generation identity. Retained records must preserve every current sample lane
bit-for-bit; both outcomes receive only the already-proven post-selection weight/M/final-W values.
An exclusive `guideSampleCopy` counter partition reports selected, retained, empty, metadata reject,
or arithmetic reject. The local value is deliberately not stored in either reservoir slot or any
sidecar, so this gate cannot reach committed history, a later frame, or the active estimator.

Fresh Vulkan runtime passed the sample-copy boundary across 29 readbacks and 250194 eligible records:
240654 source-selected payloads, 7279 exactly retained current payloads, 2261 empty no-ops, and zero
metadata or arithmetic rejects. Every frame preserved equality with the post-selection eligible and
terminal categories as well as its own exact terminal sum. The register-only full-sample contract is
therefore proven; writing it to persistent mapped history remains a separate prohibited gate.

The write-isolated storage/lifetime gate is also proven. View 20 lazily allocates a distinct
full-resolution 176 B/pixel scratch, clears it every frame, writes either the complete guide sample or
an ABI/generation terminal sentinel, then reads that device memory in a separate raygen dispatch after
a Vulkan barrier. Across 17 readbacks and 257204 eligible records, storage classified 249186 selected,
7854 retained and 164 empty outcomes, with zero metadata, arithmetic or malformed/stale rejects. Every
frame had exact category equality with the register-only sample copy, all six deltas were zero, and
`eligible = terminal`. At 1280x673 the buffer is 151613440 bytes and the full lazy view-20 diagnostic
allocation is 446.91 MiB. It never aliases either committed history slot or the existing mapped
snapshot, and ordinary rendering receives a zero BDA. Before any persistent ping-pong history is
introduced, the next gate must pair both selected and retained stored records with the exact original
source-root provenance required for future one-frame replay; mapping composition and estimator use
remain prohibited.

That source-root provenance companion is now proven in same-frame scratch. View 20 owns a separate
cleared 160 B/pixel `PathSourceRoot` buffer paired by pixel with the 176 B/pixel guide reservoir
scratch. A selected mapping replaces the old source-pixel key with its independently reprojected
current texel, advances the retained camera-relative segment origins by the frame camera delta, and
stores current source/receiver guides. A retained current sample instead captures its current
wavefront queue root and current guides. The post-barrier storage validator reads both buffers and
classifies the root as selected-ready, retained-ready, missing, or malformed without replaying it or
making it persistent.

Fresh Vulkan/RTX 5060 Ti validation produced 18 readbacks and 245291 root-eligible pairs: 237688
selected-ready and 7603 retained-ready, with zero missing or metadata rejects. Every frame preserved
`root eligible = root terminal = selected-ready + retained-ready`, all three reservoir/root deltas
were zero, and the parent reservoir scratch independently classified 245409 eligible records with
118 empty outcomes and zero rejects. The additional root scratch raises the complete lazy view-20
allocation to 606453760 bytes (578.36 MiB) at 1280x673; ordinary rendering still receives zero scratch
addresses. The next bounded gate is an isolated one-frame ping-pong/replay experiment for the paired
reservoir and root. Neither buffer may enter committed history, become a spatial source, compose a
mapping/Jacobian, or affect the estimator before that replay gate passes.

The isolated one-frame replay gate is now proven without another full-resolution allocation. A
separate CPU lifecycle token allows the paired scratch only on the immediately following view-20
frame with the same history generation and valid path-history continuity. A read-before-clear
raygen pass reprojects and validates the current receiver, independently reprojects and validates
the retained source key, then exactly replays the stored original source root. Selected diffuse
mappings use the receiver-aware source replay contract, while retained identity samples use the
generic exact replay contract. A barrier completes all reads before the same reservoir/root pair is
cleared and rewritten for the current frame. The pass is counter-only: accepted samples are not
reconstructed, selected, weighted, exposed as spatial sources, or committed to history.

Fresh Vulkan/RTX 5060 Ti validation produced 25 readbacks and 21536000 attempted pixels. Every
readback preserved `attempted = terminal = 861440` and `delta = 0`; lifecycle, metadata, and source
surface rejects stayed zero. The deliberate camera movement failed closed through 98688 receiver
reprojection rejects and 11 receiver-surface rejects, then recovered. Across the 23 settled
readbacks, exact replay accepted 54810 selected mappings and 1688 retained identity samples; only 15
strict source-replay tails and one independent source-reprojection reject remained. The complete
run accepted 57542 selected and 1791 retained samples. The lazy view-20 allocation therefore remains
606453760 bytes (578.36 MiB), and ordinary rendering still receives zero scratch addresses. The next
bounded gate may reconstruct receiver-aware mapping, visibility, and target from these accepted
previous pairs, but must remain diagnostic-only and must not update GRIS weights, sample selection,
committed history, or the ordinary estimator.

The accepted-pair reconstruction gate is now proven in the same read-before-clear pass. Only selected
diffuse records that passed exact source replay enter the remap family; retained identity records are
still reported by the replay gate but never treated as mapped samples. The shader reconstructs the
already-proven receiver-aware geometry, directional-PDF ratio, PSS Jacobian, and endpoint-independent
throughput ratio, then uses the exact receiver guide origin and production shadow SBT to classify
clear/tinted/occluded visibility and the canonical shifted luminance target. Fifteen new counters
(indices 142--156) keep remap readiness, visibility, and target identities separate. No value is
written to either scratch buffer, history, reservoir weights, selection state, debug image, or the
ordinary estimator.

Fresh Vulkan/RTX 5060 Ti validation produced 27 readbacks and 23258880 replay attempts. Every replay
and reconstruction identity was exact: all replay `delta` values were zero, remap `eligible = terminal`,
visibility categories summed exactly to visibility eligible, and target categories summed exactly to
target eligible. The run accepted 101968 selected and 2998 retained replay records; all 101968 selected
records reached remap-ready, with 101967 clear and one occluded visibility result. The corresponding
target population contained 101967 positive and one zero target, with no invalid arithmetic. A short
camera movement remained fail-closed through receiver/source reprojection and replay rejects, and all
accepted records recovered afterward. The lazy view-20 allocation remains 606453760 bytes (578.36
MiB). The next bounded gate is a counter-only comparison of the reconstructed previous target against
stored source target/throughput metadata; merge weights, selection, history, and estimator use remain
prohibited.

The stored-metadata audit keeps that boundary and adds no new full-resolution storage or ABI lane.
After a selected previous diffuse record reaches finite visibility and target evaluation, seven
view-20 counters (indices 157--163) compare reconstructed visible RGB/target with the paired scratch
record's `sampleRadianceTarget`, and independently compare receiver-guide throughput with
`reconnectionThroughput.xyz`. Each comparison has match, mismatch, and invalid categories with its
own exact eligible identity. The audit does not generate RNG, compute a merge weight, select a sample,
write scratch/history, compose mappings, change the debug image, or contribute to the estimator.
Fresh Vulkan/RTX 5060 Ti validation produced 29 readbacks and 245581 metadata-eligible selected
records. Both target and throughput partitions preserved exact zero-delta accounting on every
readback, with no invalid arithmetic. Receiver throughput matched the stored lane for 245572 records;
all nine mismatches occurred in the two brief camera-motion readbacks, while the other 27 readbacks
had 245263/245263 exact throughput matches. Visible target matched for 245564 records. Its 17
mismatches split exactly into the same nine motion-dependent throughput changes and eight fresh
occlusions; the three tinted visibility results still matched. This proves why persistent reuse must
reconstruct the current receiver target instead of copying the stored one. No Vulkan device, GPU,
shader, or memory failure was logged. The next bounded gate may audit the previous-pair GRIS merge
weight from the freshly reconstructed target, stored source final weight/count, and directly
recomputed PSS Jacobian, but must remain counter-only with no RNG, selection, history write, mapping
composition, or estimator use.

That merge-weight arithmetic audit is now proven. Sixteen fresh view-20 readbacks covered 134487
eligible previous mappings. The exact partition was `zero = 6`, `positive = 134481`, `invalid = 0`,
with `eligible = terminal` and `delta = 0` on every readback. The six zero weights matched the six
zero-target/occluded reconstructions; every positive reconstructed target produced a finite positive
weight. Two stored-throughput mismatches occurred only in the brief camera-motion population and
did not create invalid weight terms. The audit still performs no selection, RNG, weight/history
write, mapping composition, or estimator work. The next bounded gate may audit hypothetical
`nextWeightSum`, effective-count cap, and final reservoir weight arithmetic, still without selecting
or writing a sample.

The post-weight arithmetic audit is now proven in view 20. Seventeen fresh readbacks covered 188935
eligible records. `nextWeightSum` and capped effective count were valid for all records with zero
deltas. The previous-source final-weight branch was valid for all 188935 records; the
current-retained branch was valid for 45060 and rejected 143875 because the current reservoir had no
positive target, matching the CPU reference's fail-closed selected-sample denominator rule. No RNG,
selection, reservoir/history write, mapping composition, estimator contribution, or invalid GPU
arithmetic was introduced.

The capped accumulated weight is a storage bound, not a probability denominator. Runtime A/B showed
finite cases where `mergeWeight / min(currentWeightSum + mergeWeight, 1e30)` exceeds one. Selection
therefore uses an overflow-stable form of the uncapped relative ratio: divide the smaller weight by
the larger before forming the final ratio, without summing the two large values or clamping the
result. The stored sum remains capped at `1e30`, effective M remains capped at `16777216`, and source
M remains limited to eight. This separation is part of the estimator contract and must not be
collapsed by an implementation shortcut.

The diagnostic Bernoulli, branch arithmetic, and lane-policy gates are now proven. A local hash
independent of both replay RNG streams partitions the stable probability into selected and retained
outcomes. Selected records require freshly reconstructed receiver radiance/target, directional PDF,
throughput, reconnection geometry, current source key and generation; they preserve the original
canonical proposal and replay root. Retained records preserve the current opaque sample payload.
Both branches receive only the already validated weight-sum, effective-count, final-weight and
confidence lane. The latest register-only record audit covered 48 Vulkan readbacks / 76722 eligible
previous pairs: 68605 selected records and 8116 retained records passed the common metadata validator;
there was one explicit retained reject caused by the already-known zero-current-weight fail-closed
branch. Selected/rejected and retained/rejected categories preserved exact per-frame accounting and
zero total deltas. The assembled record is still register-only: it has not been written to device
memory or committed to history.

The register-only reservoir/root pairing gate is now proven. The branch-specific `PathSourceRoot`
companion is assembled in registers and validated against the selected or retained record without
writing either scratch buffer. A fresh Vulkan/RTX 5060 Ti quick-play produced 251 readbacks covering
474215 eligible pairs: 424425 selected-ready, 0 selected-reject, 49789 retained-ready and one
retained reject. Every pair delta was zero; the one reject is the already-known upstream
zero-current-weight fail-closed record, while root capture, chain and identity rejects were zero
after recovery. Persistent mapped history remains blocked until the next isolated one-frame
ping-pong/replay gate passes exact replay, receiver remap, current visibility/target reconstruction,
lifecycle invalidation and exclusive accounting as one chain. Mapped records remain forbidden as new
spatial sources, previous mapping Jacobians are never composed, and ordinary estimator integration
remains out of scope. Detailed chronological evidence and superseded experiments live in the local
`tasks_archive.md`; this roadmap is the durable mathematical authority.

The pair's own register-only seeded replay is also proven before any device write. Selected pairs
retrace their branch root and compare through the receiver-aware mapped-source contract; retained
identity pairs use the generic exact replay comparison. Twenty-eight fresh Vulkan readbacks covered
50857 eligible pairs: all 45618 selected and 5239 retained records replayed exactly, with zero
rejects and zero accounting deltas. This added only five host-visible counters (229 uints / 916 B
total), no ABI lane and no full-resolution storage. The next bounded gate may write exactly this
validated pair to isolated view-20 scratch and read/replay it one frame later. It still may not enter
committed history, become a spatial source, compose a mapping/Jacobian, or affect the estimator.

The next isolated device scratch gate is now implemented and runtime-proven. Two ping-pong slots hold
the complete branch reservoir/root pair; the shader writes only after register replay succeeds, then
the following frame independently reprojects the receiver and source and replays the stored seeds.
Thirty-six fresh Vulkan readbacks covered 6554880 attempts: 24288 selected and 2587 retained pairs
were accepted, metadata rejects stayed zero, and every terminal partition had `delta = 0`. Strict
receiver/source reprojection and source-replay tails remain explicit fail-closed categories. This
diagnostic adds 32 B/pixel of ping-pong storage (1130.43 MiB total at 1280×673) and expands WorldPush
to 672 bytes; ordinary rendering still receives zero addresses. Before history integration, the
device-written payload/root therefore had to match the register pair bit-for-bit.

That exact storage gate is now proven without another full-resolution copy. Every replay-approved
write increments eligible/completed counters and contributes to a bounded 4096-record capture of the
expected register pair. Each capture stores 16 bytes of pixel/mapping/frame/generation metadata plus
the complete 176-byte reservoir and 160-byte root. A separate ray-generation pass runs only after a
Vulkan memory barrier and compares all reservoir and root fields as raw 32-bit lanes. Across 57 fresh
readbacks, all 142835 writes completed and all 142835 captured pairs matched both structures exactly
(128416 selected, 14419 retained); metadata rejects, reservoir/root mismatches, pair rejects, capture
overflow and accounting deltas were zero. The 352-byte capture record adds 1.375 MiB to the existing
diagnostic sample buffer, leaves WorldPush at 672 bytes and replay ABI at 10, and receives no address
outside view 20. This authorizes design of an isolated candidate-history promotion gate, but does not
yet authorize committed history, recursive spatial sourcing or ordinary-estimator use.

The isolated promotion boundary is now device-enforced. Each branch-root slot appends an 8 B/pixel
tag area that is cleared with the slot and written only by the post-barrier exact validator. A tag
binds the source frame index, 24-bit history generation and mapping kind; next-frame replay requires
the exact previous frame, current generation and stored identity/diffuse mapping kind before it may
read and replay the pair. Untagged records fail closed, so capture overflow can reduce diagnostic
coverage but can never silently promote an unvalidated pair. Forty-seven Vulkan readbacks wrote
68454 tags (61146 selected, 7308 retained); 8557592 next-frame checks partitioned into 8489576 empty
and 68016 admitted (60802 selected, 7214 retained), with zero tag-metadata rejects and exact write/
replay accounting. Two tag areas add 13.14 MiB at 1280×673 (1143.57 MiB full lazy view-20 storage),
while WorldPush stays 672 bytes and replay ABI stays 10. The pair remains diagnostic-only and still
cannot become a spatial source, compose mappings/Jacobians or contribute to the estimator.

The next boundary is a compact four-frame retention ring rather than another full-resolution
history allocation. Each of four 4096-entry slots stores the exact validated reservoir/root pair,
its capture metadata, and a separate 16-byte promotion marker written only by the post-barrier
bitwise validator. A metadata-only dispatch audits all 16384 entries and assigns one exclusive
state: empty, future frame, generation reject, mapping reject, current, age 1, age 2, age 3, or
expired. Live entries also partition exactly by identity versus diffuse-reconnection mapping.
Across 16 Vulkan readbacks, all 262144 attempts had exact terminal accounting: 115226 empty,
136707 live, 10211 expired, and zero future/generation/mapping rejects or deltas. All 136707 live
entries split into 13789 identity and 122918 mapped records with zero live delta. This proves bounded
retention and expiry only. It does not authorize replay from the ring, mapping/Jacobian composition,
committed history, or estimator contribution. The next replay gate must consume the original
source-root provenance directly and fail closed if a mapped result would become another source.

Age-aware original-root replay is now proven for every live ring age. A separate 16-byte lane stores
the cumulative capture-to-current camera translation plus the exact frame last advanced; it starts
at zero after exact promotion and can advance only from frame N to N+1. Ages 1–3 reconstruct the
immutable packed source-root origins in the current camera/terrain rebase and retrace their saved
seeds. Identity records use full replay comparison, while mapped records use only the original-source
comparator. No reconnection mapping is called, so a mapped result cannot become another spatial
source. Twenty Vulkan readbacks covered 107342 eligible records: age 1/2/3 accepted
35123/35179/35051 respectively, for 105353 total accepts and 1989 explicit fail-closed replay
rejects. Metadata/provenance rejects and terminal/replay/age accounting deltas were all zero. This
still authorizes no receiver reconnection, mapping composition, weight update, history write, or
estimator contribution.

The next isolated gate directly reprojects the retained receiver root for ages 1–3. It converts the
capture-time camera-relative root with the cumulative camera delta, projects it with the current
view-projection, and searches only a 3x3 current-guide footprint. It never chains motion vectors.
Strict position/normal/roughness/full-material comparison admitted 109946 of 110106 replay-approved
records across 22 Vulkan readbacks (age 1/2/3: 36917/36489/36540); 160 surface changes failed closed,
clip/bounds rejects were zero, and terminal plus mapping accounting were exact. The next gate is a
register/counter-only direct source-to-current-receiver remap for these admitted aged records. It
must recompute geometry, directional PDFs and Jacobian from the immutable original source root and
must not multiply or compose the previously stored mapping Jacobian. History and estimator writes
remain forbidden.

That pre-visibility remap gate is now proven. It combines the immutable replayed second-hit/source
state with the independently captured current receiver guide and recomputes
`J = J_geometry * p_receiver / p_original_source`; the previous mapping Jacobian is never read.
Across 24 Vulkan readbacks, 170695 of 174142 eligible records were ready across ages 1/2/3 as
57450/56189/57056. The remaining 3447 were explicit missing/unsupported diffuse-edge rejects;
guide, geometry, PDF and throughput rejects plus terminal/mapping deltas were zero. The next gate is
a counter-only visibility and shifted-target reconstruction for these aged direct remaps using the
exact current receiver ray origin and production shadow SBT.

That visibility/target gate is now proven as well. The production shadow query classifies clear,
tinted, occluded and invalid transmittance; valid occlusion proceeds to a zero target instead of
being treated as a reject. Across 42 Vulkan readbacks, all 284642 direct-remap-ready records reached
valid targets: 284636 clear/positive, 4 tinted/positive and 2 occluded/zero. Invalid visibility and
target arithmetic were zero. Ages 1/2/3 balanced as 95250/94636/94756, identity/mapped ownership as
22597/262045, and every gate/terminal/age/mapping delta was zero. The next isolated gate may form the
counter-only GRIS merge weight from the fresh shifted target, original source final weight,
`min(sourceM, 8)` and the newly recomputed direct Jacobian. Selection, reservoir/history writes and
estimator contribution remain forbidden.

That aged GRIS weight gate is now proven counter-only. Forty-two Vulkan readbacks classified all
166943 target-ready records as finite positive weights, with zero invalid/zero categories and exact
terminal, age, mapping and source-count accounting. Ages 1/2/3 were 56114/55285/55544 and mapping
ownership was 13682 identity + 153261 mapped. The current non-persistent source population was
entirely `M <= 8`; CPU reference tests separately prove that `M > 8` uses exactly 8 in the formula.
No stored previous Jacobian is composed. The next isolated gate may evaluate the overflow-stable
relative selection probability against the current receiver reservoir, but must not consume RNG or
mutate reservoir/history/estimator state.

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
