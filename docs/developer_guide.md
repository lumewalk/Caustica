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
View 20 also performs the first diagnostic spatial merge for finite accepted pairs. It writes the
merged weight sum, effective count, selected target, and final weight into a dedicated lazy mapped-
snapshot buffer; the normal current path slot is still the only slot committed by path history. A
source-selected snapshot record retains the original source seeds/states and carries the ABI-10
one-hop diffuse-reconnection descriptor together with the receiver PDF/throughput. Generic identity
replay rejects that mapping kind, and view 20 refuses an already mapped record as another spatial
source. Receiver-aware replay is mandatory before any mapped record may be committed.
The pass does not update committed reservoirs, history weights, or the active estimator.
The shader-independent `DiffuseMappingReplay` reference defines that boundary: replay ABI,
mapping kind, and first-edge event must match; receiver PDF and PSS Jacobian must be finite and
positive; replayed source RGB is divided by its exact source throughput, multiplied by the stored
receiver throughput and freshly traced RGB transmittance, and compared with the stored shifted RGB
using the same relative tolerance as GPU seeded replay.

View 20 dispatches that receiver-aware validation as a same-frame ray-generation pass after the
snapshot merge. A lazy view-20-only 160-byte-per-pixel sidecar retains both packed source queue
segments plus source and receiver position/material and normal/roughness guides. Segment origins and
positions are stored camera-relative, so they do not depend on the capture frame's terrain rebase.
Together with the separate 176-byte mapped snapshot and 32-byte exact receiver sidecar this costs
about 302.32 MiB at 1280x673 and is not allocated by ordinary rendering. The validation pass replays
the retained source root rather than dereferencing the transient queue, checks
the source topology and proposal state, reconstructs the receiver-side geometry, directional PDF,
PSS Jacobian and first-event throughput, traces receiver-to-second-hit visibility again, and compares
the resulting shifted RGB/target with the snapshot record. Source-root written/invalid counters and a
separate source-root replay reject make this lifetime boundary explicit. A valid frame has
`eligible == scratchSelected` and the sum of accepted plus all rejects equals eligible; the expected
steady-state result is near-total acceptance, while isolated finite-density/PDF outliers may be
rejected rather than admitted by weakening replay tolerances. ABI, source-state, receiver,
visibility, radiance, or retained-root rejects require investigation. The diagnostic image is
unchanged by this validation dispatch. Mapped records still do not enter committed history or the
normal estimator.

The same buffers now form a strict one-frame diagnostic snapshot without a second ping-pong copy.
After current guides are ready, the cross-frame pass first reads the previous snapshot. Receiver
motion reprojects the current receiver to its previous pixel. The retained source position is
independently projected into the current frame, a 3x3 search requires the candidate's motion vector
to return to the exact previous source pixel, and source/receiver position, normal, roughness, and
full material identity must match. Stored camera-relative positions and segment origins are rebuilt
with `currentCamOffset - camDelta`. This initial policy is deliberately static-surface-only: object
motion or changed geometry/material produces a fail-closed reprojection reject.

Once both roots pass, the original source seeds are replayed exactly. The pass recomputes the direct
source-to-current-receiver PDF and Jacobian, uses the current receiver throughput, traces current
visibility, and never multiplies by the stored previous-receiver Jacobian. Exclusive `crossFrame[...]`
counters enforce `attempted = receiverReprojection + mappedEmpty + eligible` and
`eligible = accepted + terminal rejects`. Only after that read completes is the mapped snapshot cleared and rewritten by
the current frame, followed by the existing same-frame replay. Snapshot continuity requires the
immediately preceding frame and matching history generation, and resets/toggle gaps invalidate it.
Mapped records still do not enter committed path history or the estimator. Vulkan validation on an
RTX 5060 Ti preserved both accounting equalities for all 108 cross-frame readbacks and produced no
ABI/root corruption. Ninety settled-camera readbacks accepted 40,803/342,717 eligible records
(11.905741%); the combined strict receiver-path check accounted for 301,409 terminal rejects, while
source reprojection/source replay accounted for 16/489 and all later geometry/PDF/visibility/radiance
reject categories stayed zero. The moving-camera interval remained fail-closed and acceptance
recovered after motion stopped. Before this boundary changes, split the receiver category into
surface and path-policy causes, validate that policy, and define explicit moving-surface support;
do not hide safe rejects by weakening replay tolerances. The diagnostic now reports those causes as
`receiverSurface`, `receiverSample`, `receiverEdge`, `receiverTopology`, `receiverDepth`,
`receiverTransport`, and `receiverFootprint`, plus their derived `receiverTotal`. Each eligible
record increments exactly one terminal counter in the same fail-closed order encoded by the CPU
reference. This is instrumentation only: it does not relax admission or write mapped history.

The initial split-counter capture found `receiverSample` to be the dominant settled-camera cause:
56,370 of 76,551 eligible records (73.637183%), versus topology 6,357, edge 2,477, footprint 1,105,
and surface 356. Acceptance was 8,594 (11.226503%); depth, transport, ABI/root, and all downstream
mapping rejects were zero. Camera motion moved the terminal population to the expected fail-closed
surface category. The counter-only `sampleRescue[...]` A/B now evaluates zero-current-sample records
through edge, topology, depth, transport, and footprint checks. Its exact shadow identity is
`eligible = rescued + edge + topology + depth + transport + footprint`, while strict accounting
still places every one of those records in `receiverSample`. Even a fully rescued shadow record
returns before geometry/radiance evaluation; it is never selected and cannot alter weights or
history. The shader-independent policy reference mirrors both orderings.

The first runtime A/B produced zero sample rescues: all 41,685 zero-current-sample candidates across
20 settled readbacks also lacked a valid receiver edge. Every shadow and strict accounting identity
was exact. Thus the positive-current-sample check is not an independent admission bottleneck here;
removing it would gain nothing and must not be treated as the fix. Diagnose the receiver-edge
subconditions next, especially whether exact receiver material/PDF/throughput data must be generated
independently of a positive current canonical endpoint before persistent remapping can be sound.

`edgeBreakdown[...]` is the next orthogonal shadow partition. It reports `missingValid`, `depth`,
`event`, `mapping`, `pdf`, and `finite`, with
`eligible = receiverEdge + sampleRescue.edge = sum(edge outcomes)`. It does not replace the strict
or sample-rescue terminal counters and cannot reach mapping geometry/radiance.

Runtime edge breakdown found only two populated causes: 139,525/146,307 missing-valid records, all
from the zero-current-sample shadow population, and 6,782 valid but non-diffuse events, all from the
strict receiver-edge category. Depth/mapping/PDF/finite failures were zero. Thus the stored edge is
not malformed; it is absent whenever no positive canonical endpoint published it. The next design
gate is a deterministic diffuse receiver-material contract independent of endpoint selection. Audit
the precision and semantics of the existing albedo/material guides before introducing a new exact
buffer or permitting persistence.

The audit shows why the current RGBA16F albedo guide cannot supply exact mapping state. Diffuse path
selection uses `ps(F0, diffAlb)`, so receiver throughput is `diffAlb/(1-ps)` and receiver PDF carries
the same `(1-ps)` technique mass. F0 is texture-evaluated at the hit and is absent from the guide
cache. `DiffuseReceiverMaterial` and `DiffuseReceiverGuide` now define the shader-independent
contract and an explicit same-albedo/different-F0 counterexample. View 20 lazily allocates a
two-lane FP32 sidecar (32 B/pixel, about 26.29 MiB at 1280x673). The first
`float4(exact diffuse RGB, diffuse technique mass)` lane is written by primary visibility before
canonical endpoint selection: opaque receivers use the same `ps(F0, diffAlb)` calculation as the
path tracer, particles use diffuse mass 1, and unsupported dielectric/water receivers write mass 0.
Pass A clears the second lane every frame; Pass B fills it with the exact biased outgoing-ray origin
for the unsplit camera receiver before lobe or endpoint selection. Replay dispatches cannot overwrite
that lane. Ordinary rendering keeps the BDA zero and allocates no corresponding memory.

Cross-frame replay currently uses the sidecar only as a shadow validation. `receiverGuide[...]`
partitions every surface-compatible attempt into invalid guide, no positive stored diffuse edge, or
stored-edge comparison; the latter is split into accepted, technique-mass, throughput, and PDF
mismatch. The accounting identities are `attempted = invalid + noStoredEdge + storedEligible` and
`storedEligible = accepted + mass + throughput + pdf`. These counters do not relax admission, feed
mapping terms, select samples, update history, or change the estimator.

The first runtime comparison covered 29 stable readbacks and 106191 stored-edge comparisons. Both
accounting identities held in every readback. The exact guide was always valid, throughput mismatch
was zero, and technique-mass mismatch was 299 records (0.2816%). Directional-PDF reconstruction
rejected 25180 records (23.7120%), isolating the remaining blocker to the sampled outgoing direction
or its exact density rather than to F0/material reconstruction. A brief camera move produced one
fail-closed readback with only 33 stored-edge comparisons; the stationary population recovered on
the next readback. The sidecar must not replace a current edge or enter persistent history until the
directional contract is exact.

The PDF follow-up does not store another sampled path or change the reservoir. Instead it removes a
mixed-coordinate reconstruction: the selected second-hit vertex comes from Pass B, while the old
origin was rebuilt from the separately traced Pass-A guide hit. The shadow comparison now forms the
sampled edge from Pass B's exact stored biased origin. Guide validity is checked before stored-edge
availability, so the `noStoredEdge` population also proves that the material/origin substrate exists
independently of a positive canonical endpoint. Runtime counters must show whether this removes the
PDF mismatch before the sidecar is allowed to feed remapping.

The fresh exact-origin run passed that gate. Across 15 stationary readbacks, all 207213 attempted
receivers had valid material and origin state, including 164116 records with no stored canonical
edge. Of 43097 stored-edge comparisons, 42899 were accepted, 175 failed the material-mass check,
throughput mismatch stayed zero, and only 23 failed PDF reconstruction (0.053368%, down from
23.7120%). Both accounting identities held in every frame. The remaining finite PDF-tail records
stay fail-closed under the existing tolerance. The next step is a reference/counter-only guide-remap
A/B for the no-edge population; it must not bypass strict history admission or write an estimator.

That guide-only A/B now evaluates the exact pre-visibility diffuse remap chain for every
`noStoredEdge` record: source/receiver reconnection geometry, receiver directional PDF and PSS
Jacobian, then spectral throughput support. The four terminal counters are mutually exclusive and
must satisfy `guideRemap.eligible = ready + geometry + pdf + throughput`; because the experiment is
attached directly to the valid-guide no-edge branch, `guideRemap.eligible` must also equal
`receiverGuide.noStoredEdge`. `ready` means only that finite mathematical terms can be formed. The
shadow path traces no visibility ray, does not relax the existing current-sample/edge or
topology/depth/transport/footprint gates, and cannot select a sample, write a reservoir, update
history, alter weights, or contribute to the estimator.

The first runtime audit passed this pre-visibility gate without a single reject. Across 23 stable
readbacks, all 419559 eligible no-edge records were `ready`; geometry, PDF, and throughput rejects
were zero. Both required identities held in every readback. Vulkan RT initialized on the RTX 5060
Ti without device, GPU, or shader failures. This proves availability of the finite remap terms, but
not shifted visibility. The next experiment may trace receiver-to-shared-vertex visibility for this
population in view 20 only; it must retain the same shadow-only isolation and must not write mapped
history or feed the estimator.

The visibility experiment now uses the exact Pass-B biased receiver origin and calls the same
production `visibility(...)` shadow SBT as the established shifted-radiance path. Every pre-visibility
`ready` record increments one mutually exclusive terminal counter: `clear`, `tinted`, `occluded`, or
`invalid`. Runtime must prove `guideVisibility.eligible = guideRemap.ready` and
`guideVisibility.eligible = clear + tinted + occluded + invalid`. The call observes alpha-tested,
translucent, and water traversal, but its result is not written to the debug image or any reservoir;
it cannot select a path, alter weights/history, relax strict admission, or feed the estimator.

Runtime passed the visibility gate. Across 26 stable readbacks, all 395053 pre-visibility-ready
records entered the visibility partition: 394614 were clear (99.8889%), 439 were correctly occluded
(0.1111%), and tinted/invalid were both zero in the tested scene. Both identities held in every
readback, and Vulkan RT reported no device, GPU, or shader failure. Zero tinted records describe this
capture rather than removing RGB transmittance from the contract. The next bounded experiment is a
counter-only post-visibility shifted-radiance/target audit for visible records; persistent history,
weights, and the estimator remain out of scope.

A targeted glass/water follow-up exercised the rare tinted branch. Across 48 readbacks and 952802
eligible records, 952446 were clear, 7 were tinted across five separate frames, 349 were occluded,
and invalid remained zero. Both identities stayed exact in every readback. RGB transmittance is
therefore runtime-covered as well as reference-tested; its low frequency is a property of the
receiver-to-shared-vertex geometry in the tested scene.

The next counter-only stage now applies every valid visibility result, including zero occlusion, to
the proven unoccluded shifted radiance and computes the canonical luminance target. Its mutually
exclusive terminals are `positive`, `zero`, and `invalid`. Runtime must prove
`guideTarget.eligible = guideVisibility.clear + tinted + occluded` and
`guideTarget.eligible = positive + zero + invalid`. Visibility-invalid records do not enter this
stage. The target is observed only through counters: it is not written to the debug image or a
reservoir and cannot affect selection, weights, history, strict admission, or the estimator.

Runtime passed the shifted-target gate across 77 readbacks and 411047 eligible records: 406266 were
positive (98.8369%), 4781 were zero (1.1631%), and invalid was zero. The visibility population was
406131 clear, 135 tinted, and 4781 occluded. Both required identities held in every readback; the
stronger scene-specific equalities `positive = clear + tinted` and `zero = occluded` also held in
all 77. This proves RGB-transmitted canonical target arithmetic without authorizing its use. The
next bounded stage may audit the finite GRIS candidate/merge-weight terms with counters only.

### Current T-050 diagnostic workflow

The current spatial path work remains isolated in debug view 20. It does not update committed path
history or the ordinary estimator. At 1280×673 its complete lazy diagnostic allocation is about
578.36 MiB and consists of the 176 B/pixel guide reservoir scratch, its 160 B/pixel source-root
companion, and the exact two-lane receiver guide. Ordinary rendering receives zero addresses for
these resources.

The proven one-frame chain is:

1. reproject current receiver and original source independently;
2. validate position, normal, roughness and full material identity;
3. replay the original source seeds exactly;
4. recompute direct source→current-receiver geometry, directional PDFs and PSS Jacobian;
5. rebuild receiver throughput, trace current visibility and reconstruct shifted target;
6. form the GRIS merge weight with source M limited to eight;
7. choose through the overflow-stable uncapped relative ratio;
8. audit selected/retained final arithmetic and future lane policy.

The accumulated reservoir sum may still be stored with the `1e30` cap, but that capped value must
never be reused as the Bernoulli denominator: runtime diagnostics demonstrated finite probabilities
above one in that formulation. Do not clamp those cases. Use the overflow-stable relative ratio
defined by the CPU reference and mirrored in `pathStableSelectionProbability`.

The register-only full-record gate is proven: selected and retained records are assembled in shader
registers and passed through `pathReservoirSampleMetadataReady`. The paired branch-specific
`PathSourceRoot` companion is now also proven in the same register-only pass. A fresh quick-play
produced 251 Vulkan readbacks with 474215 eligible pairs, 424425 selected-ready, 0 selected-reject,
49789 retained-ready and one explicit retained reject for the upstream zero-current-weight
fail-closed case. Exact pair and branch deltas stayed zero; root capture/chain/identity rejects were
zero after that record-level reject. The next gate is isolated one-frame ping-pong/replay of this
paired record. It must still not be stored, advance either replay RNG stream, commit history, expose a
mapped record as another spatial source, compose a prior Jacobian, or contribute to the estimator.

Before that write, the pair is now replayed once directly from its assembled register root. The
fresh runtime gate covered 50857 eligible pairs across 28 readbacks: 45618 selected and 5239 retained
records were exact, with zero rejects and zero deltas. This is still a counter-only check; the next
stage is the isolated device scratch write/read and one-frame replay of exactly the same pair.

That isolated write/read gate is now active in view 20. Two ping-pong `PathReservoir` and
`PathSourceRoot` slots are cleared before the previous replay pass; a branch pair is written only
after register replay succeeds, and the next frame reads/reprojects/replays the previous slot. The
latest 36 readbacks covered 6554880 attempts, with 24288 selected and 2587 retained accepts, zero
metadata rejects and exact `attempted = terminal` accounting on every readback. Reprojection and
source-replay rejects are expected strict categories, not repaired by tolerance loosening. The
diagnostic allocation is 1130.43 MiB at 1280×673 and WorldPush is 672 bytes; all four branch BDA
addresses are zero outside view 20. The next required boundary was explicit bit-for-bit equality
between the register pair and its device-written payload/root.

The post-barrier storage audit now follows that write pass. It captures at most 4096 complete
expected register pairs per ring slot. The record is now 384 bytes: metadata, the 176-byte
`PathReservoir`, the 160-byte `PathSourceRoot`, a 16-byte promotion marker written only after
the raw 32-bit lane comparison passes, and a 16-byte replay-camera provenance lane. The diagnostic
counter buffer is now 292 uints / 1168 bytes. Four capture slots use 6 MiB; together with the
pre-existing sample area the host-visible diagnostic sample buffer is 6422528 bytes (6.125 MiB).
WorldPush remains 672 bytes
and replay ABI remains 10. The runtime log line
`RT path guide branch scratch storage` must show `eligible == completed`, `attempted == captured`,
zero metadata/reservoir/root/pair rejects, and zero write/validation deltas. The proven run covered
57 readbacks and 142835 exact pairs (128416 selected, 14419 retained), with no capture overflow.

Exact samples now receive a device-side candidate tag after validation. The tag area is appended to
each root ping-pong slot (8 B/pixel), is cleared together with the current slot and records the source
frame, 24-bit history generation and mapping kind. Previous-frame replay first checks this tag; an
untagged pair or any frame/generation/mapping mismatch fails closed before source replay. The two tag
areas add 13.14 MiB at 1280×673, taking full lazy view-20 storage to 1143.57 MiB.

The runtime line `RT path guide branch candidate promotion` must have exact write and replay
partitions and zero metadata rejects. The proven run used 47 readbacks: 68454 writes split into
61146 selected and 7308 retained; 8557592 reprojected checks split into 8489576 empty, 60802
selected-admitted and 7214 retained-admitted, with both deltas zero. Admission remains diagnostic:
the tagged pair is not committed to ordinary history, cannot be reused as a spatial source, and does
not contribute to the estimator.

The compact promotion records now form a four-slot ring indexed by validation frame. A dedicated
4096×4 ray-generation audit classifies every entry as empty, future, generation/mapping reject,
current, age 1–3, or expired. It never traces an entry and never writes a reservoir/history lane.
The line `RT path guide branch candidate retention` must satisfy `attempted == terminal` and
`live == identity + mapped`; future/generation/mapping rejects and both deltas must remain zero in a
stable run. Sixteen fresh readbacks covered 262144 entries: 136707 live records (34278 current,
33855 age 1, 34263 age 2, 34311 age 3), 10211 expired, and 115226 empty. Mapping ownership split
exactly into 13789 identity and 122918 mapped live records.

Age-aware replay now runs in a separate post-retention pass. The compact provenance lane starts at
zero on validation and accumulates exactly one `camDelta` per subsequent live frame; skipped or
duplicated advancement fails closed. Ages 1–3 reconstruct the original packed queue origins with
`current camOffset - cumulative camera delta` and retrace only the stored original seeds. Identity
records use exact generic replay comparison. Diffuse-mapped records use the source-replay comparator;
the pass never calls reconnection mapping and therefore cannot use a mapped result as another spatial
source. Current records are skipped and age 4+ records are expired.

The runtime line `RT path guide branch age replay` must show nonzero eligible and accepted counts for
each age, `provenance=0`, and `delta=replayDelta=ageDelta=0`. The proven run covered 20 readbacks and
327680 attempts. All 107342 age-eligible records split across age 1/2/3 as 35612/35867/35863, with
accepted counts 35123/35179/35051. Overall acceptance was 105353 (10382 identity and 94971 mapped
original-root), while 1989 exact replay rejects remained explicit. Metadata/provenance rejects and
all accounting deltas were zero.

After source replay succeeds, the same age pass directly projects the immutable capture-time
receiver root through `curViewProj`. Camera-relative projection subtracts the accumulated capture-to-
current camera delta; comparison with the current guide buffers adds `current camOffset - cumulative
camera delta`. A 3x3 search tolerates pixel quantization, but position, normal, roughness and the full
24-bit material identity still use the strict replay predicates. Motion vectors are not read. The
runtime line `RT path guide branch receiver admission` must have exact `terminal=eligible`,
`delta=0`, `mappingDelta=0`, and nonzero admitted counts for ages 1–3. The proven static run covered
22 readbacks: 110106 eligible, 109946 admitted (36917/36489/36540 by age), 160 explicit surface
rejects, and zero clip/bounds rejects. Mapping ownership also balanced exactly as 10319 identity +
99627 mapped. This gate only classifies current receiver roots; it does not construct a new mapping,
change weights, write history, or affect the estimator.

Replay-approved and receiver-admitted aged records then run a pre-visibility direct-remap audit.
The immutable replay aggregate supplies the original second-hit vertex, geometric normal, source
directional PDF, source first-edge throughput and canonical radiance. The current receiver sidecar
supplies an independently captured diffuse technique mass, throughput and exact biased ray origin.
The pass recomputes solid-angle geometry, receiver directional PDF and
`geometry * receiverPdf / originalSourcePdf` directly from those terms. It never reads the stored
previous mapping Jacobian and does not trace visibility. `RT path guide branch direct remap` must
show exact terminal and mapping accounting, nonzero ready counts for ages 1–3, and zero
guide/geometry/PDF/throughput rejects in a stable static scene. The proven run covered 24 readbacks:
174142 eligible, 170695 ready (57450/56189/57056 by age), 3447 explicit unsupported/missing-edge
rejects, 12602 identity-ready plus 158093 mapped-ready, and zero other rejects or deltas. Counter
storage at that checkpoint was 314 uints / 1256 B. No reservoir, history, weight, selection or
estimator write occurs.

Each ready aged direct remap now also performs the production shadow query from the exact current
biased receiver origin to the replayed shared vertex, then multiplies the reconstructed shifted
radiance by the returned RGB transmittance and evaluates its luminance target. Occlusion is valid
visibility and must advance to a zero target; only non-finite/out-of-range visibility or target
arithmetic rejects. `RT path guide branch direct target` must show exact visibility, target, age and
mapping partitions plus `gateDelta=0`. The proven run covered 42 readbacks: all 284642 remap-ready
records reached a valid target, with 284636 clear, 4 tinted and 2 occluded/zero; invalid categories
and every delta were zero. Age 1/2/3 split as 95250/94636/94756 and mapping ownership as 22597
identity + 262045 mapped. Current counter storage is 330 uints / 1320 B; replay ABI, 384 B capture
record and 672 B WorldPush remain unchanged. At that checkpoint no GRIS weighting, selection,
history or estimator contribution was authorized.

The follow-on aged weight audit evaluates
`shiftedTarget * storedFinalWeight * min(storedM, 8) * directPssJacobian` entirely in registers.
`storedFinalWeight` and `storedM` belong to the retained source reservoir; the target and Jacobian
are freshly reconstructed for the current receiver, and the previous mapping Jacobian is never
read. `RT path guide branch direct weight` must close weight, source-count, age and mapping
partitions exactly. Across 42 readbacks all 166943 target-ready records produced finite positive
weights, split across ages 1/2/3 as 56114/55285/55544 and identity/mapped as 13682/153261. Invalid,
zero and every delta were zero. All runtime source counts were already at or below 8, as expected
for the current non-persistent producer; the `M > 8` capped branch is covered by the CPU reference
contract. Counter storage is now 342 uints / 1368 B. Selection, RNG, reservoir/history writes and
the estimator remain untouched.

The next aged gate evaluates only the overflow-stable relative selection probability. It reads the
current receiver reservoir's `weights.x`, permits zero for an empty reservoir, and calls
`pathStableSelectionProbability(currentWeightSum, mergeWeight)` without first adding the two
operands. `RT path guide branch stable selection` must satisfy all of the following:

- `weightReady == eligible`;
- `eligible == currentReject + probability.zero + probability.open + probability.one
  + probability.invalid`;
- `ready == current.zero + current.positive == age.one + age.two + age.three
  == mapping.identity + mapping.mapped`;
- all printed deltas are zero, with `currentReject == probability.invalid == 0` in a stable run.

Forty-three fresh Vulkan readbacks covered 188524 records: probability categories were zero/open/one/
invalid = 0/44733/143791/0, current zero/positive = 143760/44764, ages 1/2/3 =
63375/62492/62657, and identity/mapped = 15914/172610. The 31 additional exact-one outcomes with a
positive current weight are legitimate float rounding when that weight is negligible relative to the
merge weight; CPU tests preserve this boundary. Counter storage is now 356 uints / 1424 B. This pass
does not draw RNG, select or copy a sample, write any reservoir/history/scratch payload, or affect the
ordinary estimator.

The aged Bernoulli audit then derives a diagnostic draw from
`pathHash(receiverPixel ^ ringEntry * 747796405 ^ frame * 2891336453 ^ 0x94D049BB)`.
The ring-entry term prevents several aged candidates reprojected to one receiver from sharing the
same draw, while the seed remains independent of both stored replay RNG streams. The line
`RT path guide branch Bernoulli` must satisfy:

- `probabilityReady == eligible == selected + retained + invalid`;
- `ready == probability.zero + probability.open + probability.one`;
- `ready == age.one + age.two + age.three == mapping.identity + mapping.mapped`;
- `zeroViolation == oneViolation == invalid == 0`, and every printed delta is zero.

Twenty-one fresh Vulkan readbacks covered 245156 candidates: selected/retained were
222682/22474; probability zero/open/one were 111/53202/191843; ages 1/2/3 were
82442/81412/81302; identity/mapped were 17233/227923. Boundary violations, invalid and all deltas
were zero. Counter storage is now 371 uints / 1484 B. This audit performs no payload copy and writes
no reservoir, source root, scratch history, committed history, or estimator contribution.

The follow-on line `RT path guide branch post selection` audits only scalar register arithmetic.
It applies stored weight/M caps of `1e30` and `16777216` with pre-add saturation tests, adds
`min(sourceM, 8)` to M, selects shifted versus current target from the Bernoulli outcome, and computes
final `W = nextWeightSum / (nextM * selectedTarget)`. Zero weight sum is a valid empty result with
`W=0`; any non-empty result requires finite positive M, target, denominator and W. Required runtime
identities are:

- `BernoulliReady == eligible == terminal`;
- `terminal == currentReject + nextInvalid + selected.ready + selected.targetReject
  + selected.finalReject + retained.ready + retained.targetReject + retained.finalReject`;
- `eligible - currentReject - nextInvalid == weight.uncapped + weight.capped
  == count.uncapped + count.capped`;
- `ready == age.one + age.two + age.three == mapping.identity + mapping.mapped`;
- every printed delta is zero; stable runs should also have all reject categories at zero.

Sixty-one Vulkan readbacks covered 62392 candidates: selected/retained ready were 56362/6030,
ages 1/2/3 were 20683/20983/20726, and identity/mapped were 5169/57223. Every reject, cap and delta
was zero. CPU tests separately cover weight/M saturation, empty output, selected/retained target
choice, invalid inputs and non-finite final denominators. Counter storage is now 391 uints / 1564 B.
This gate does not copy or construct a reservoir payload and cannot write scratch, history, source
provenance or estimator state.

`RT path guide branch register record` is the next register-only boundary. For selected samples it
preserves the immutable source proposal/replay lanes and source key, while rewriting shifted
radiance/target, direct Jacobian, current receiver PDF/throughput, reconnection data, mapping kind and
generation. For retained samples, `pathReservoirSampleMetadataMatches(record, current)` must remain
true after replacing only the weights. Non-empty records must pass `pathReservoirSampleMetadataReady`;
empty results construct no sample. Required identities are:

- `postReady == eligible == selected.ready + selected.reject + retained.ready
  + retained.reject + empty`;
- selected population closes independently for rewrite and preserved-source lanes;
- retained non-empty population closes independently for preserved-current lanes;
- `eligible - empty == weights.ready + weights.reject`;
- `ready == age.one + age.two + age.three == source.identity + source.mapped`;
- every printed delta is zero. Replay-tail rejects remain valid terminal outcomes and must not be
  hidden by widening tolerance.

Across 42 Vulkan readbacks, 194062 outcomes produced 175134 selected-ready, 18919 retained-ready,
7 empty and 2 selected preservation rejects. All 175136 selected candidates passed receiver rewrite;
the two rejects were identity-source footprint tails caught only when converting the source into the
stricter mapped-record contract. Retained preservation and all 194055 non-empty weight lanes passed.
Ages 1/2/3 were 64760/65104/64196; source identity/mapped ownership was 15838/178222. All deltas were
zero and no Vulkan/GPU/shader error occurred. Counter storage is 411 uints / 1644 B. No local record
is written; direct source-key/root reprojection remains the next provenance gate.

`RT path guide branch register pair` proves that provenance gate without a device write. Selected
records directly reproject the immutable capture-time source root with cumulative camera provenance,
require an exact current 3x3 guide match, replace only the source screen key, and advance packed
queue origins once into current camera-relative coordinates. Retained records capture the current
queue root. The required identities are:

- `recordReady == eligible == selected.ready + selected.reject + retained.ready
  + retained.reject + empty`;
- selected record population closes across `sourceReady + clipReject + boundsReject
  + surfaceReject`; source-ready closes across key ready/reject; key-ready closes across selected
  root ready/reject;
- retained record population closes across retained root ready/reject;
- `ready == age.one + age.two + age.three == source.identity + source.mapped`;
- root chain/identity rejects stay explicit and every printed delta is zero.

Twenty-seven Vulkan readbacks covered 178331 pairs: 161268 selected, 17063 retained, and no empty
records in this run. Every pair was ready; all source/key/root/chain/identity rejects and all deltas
were zero. Ages 1/2/3 were 59322/59432/59577; original identity/mapped ownership was
14276/164055. Counter storage is 435 uints / 1740 B. This still does not write scratch or history;
the next diagnostic must retrace the assembled pair from its own root before storage is considered.

`RT path guide branch register pair replay` is that final register-only replay boundary. It retraces
the paired root with the paired source texel and saved seeds; selected records use the mapping-source
comparator while retained records require the full exact replay comparator. Required identities are:

- `pairReady - empty == eligible == selected.accepted + selected.reject
  + retained.accepted + retained.reject`;
- `accepted == age.one + age.two + age.three`;
- `accepted == source.identity + source.mapped`;
- `accepted == segments.one + segments.two`;
- every branch, terminal, age, source, segment and gate delta is zero.

Fifty-six Vulkan readbacks covered 32047 pairs: 29262 selected and 2785 retained, all accepted.
Ages 1/2/3 were 10912/10543/10592 and identity/mapped source ownership was 2571/29476. The runtime
scene produced only one-segment pairs; two-segment policy is covered by the CPU reference and shader
build. No replay reject, delta or Vulkan/GPU/shader error occurred. Counter storage is 448 uints /
1792 B. The gate performs no scratch/history write and does not affect the ordinary estimator.

`RT path guide branch aged storage` is the first device-write boundary for those replay-approved
aged pairs. It appends two isolated arrays to the host-visible view-20 sample buffer: an expected
capture and a stored capture, each bounded to 4096 entries with a 352-byte stride (16-byte
frame/generation/age/mapping metadata, 176-byte `PathReservoir`, and 160-byte `PathSourceRoot`). A
separate ray-generation pass reads both arrays only after a device barrier and requires exact
metadata plus bit-for-bit reservoir/root equality. The complete sample buffer is therefore
9306112 bytes (8.875 MiB), while the diagnostic counter buffer is 468 uints / 1872 B. Both BDAs
remain zero outside view 20; full-resolution scratch, committed history and the estimator are not
written by this gate.

Twenty-four Vulkan readbacks offered 158319 replay-approved pairs to the bounded capture. It stored
and validated the 98304-entry capacity (4096 per readback) and explicitly reported the remaining
60015 entries as bounded overflow. All 98304 stored reservoirs and roots matched bit-for-bit;
metadata, reservoir/root mismatch and pair rejects were zero. Accepted storage split into 88637
selected and 9667 retained records, ages 1/2/3 were 32643/32948/32713, original source ownership
was 7452 identity plus 90852 mapped, and every validated record had one replay segment in this
scene. All validation/partition/gate deltas were zero and no Vulkan/GPU/shader error occurred.

This proves isolated storage equality, not receiver-slot ownership. Several aged candidates may
reproject to the same current receiver, so the next diagnostic boundary must measure that fan-in
and define a deterministic single-writer winner before any full-resolution persistent branch slot
is written. The bounded arrays themselves must never be treated as committed path history.

`RT path guide branch receiver ownership` now proves that boundary without storing a path payload.
Every replay-approved aged pair writes one 16-byte frame/generation/receiver/control claim at its
unique four-slot ring index and atomically increments an 8-byte-per-receiver ownership sidecar. A
commutative `atomicMax` priority prefers age 1 over age 2 over age 3, then the smaller ring index;
the three disjoint 14-bit age bands make every live priority unique and non-zero. The sidecar is
cleared independently each view-20 frame, while stale claims fail the frame/generation checks.
A separate post-barrier full-resolution pass validates the winner against its exact ring entry and
partitions receiver fan-in, output branch, age, original source mapping and replay segment count.

The claim array adds 262144 bytes, bringing the host-visible sample buffer to 9568256 bytes
(9.125 MiB). The receiver ownership sidecar adds 8 B/pixel (6891520 bytes / 6.57 MiB at
1280x673), and the counter buffer is 491 uints / 1964 B. All storage remains lazy and view-20-only;
no `PathReservoir`, `PathSourceRoot`, committed history or estimator lane is written.

Across 36 Vulkan readbacks, all 130558 replay-approved claims were written and recovered through
the receiver sidecar. They occupied 129703 receivers: 128848 unique plus 855 collisions, all with
fan-in two (`128848 + 2*855 = 130558`); maximum fan-in was two. Every occupied receiver produced
one valid winner: 116715 selected and 12988 retained, ages 1/2/3 of 43947/43955/41801, original
source ownership of 10714 identity plus 118989 mapped, and 129703 one-segment winners. Metadata
rejects and every claim/receiver/collision/winner/partition delta were zero, with no Vulkan/GPU/
shader error. The next isolated gate may write only the winner pair to a separate view-20
full-resolution scratch slot and prove post-barrier equality; it still may not commit path history.

The winner-storage gate implements that copy without reusing either branch ping-pong slot. The
age-replay invocation first writes its complete 336-byte `PathReservoir` + `PathSourceRoot` pair at
the same unique ring index as its ownership claim. After the ownership barrier, exactly one
full-resolution invocation per occupied receiver reads the deterministic winner and copies it into
a separately cleared scratch allocation. The scratch contains three dense arrays: 176 B/pixel
reservoirs, 160 B/pixel roots and a 16 B/pixel frame/generation/priority/control commit tag. The tag
is written last and is the only validity gate; an empty receiver must observe an all-zero tag.

A distinct post-barrier pass reopens the winning claim and ring pair, checks frame, generation,
receiver, age, source/output mapping, replay segment count and exact tag ownership, then compares
every reservoir and root bit. It records empty-dirty, metadata, reservoir, root and pair rejects
separately and partitions accepted pairs by selected/retained branch, age, original source mapping
and segment count. The indexed owner-pair area adds 5505024 bytes, taking the host-visible sample
buffer to 15073280 bytes (14.375 MiB). The full-resolution winner scratch is 352 B/pixel or
303226880 bytes (289.18 MiB) at 1280x673. `WorldPush` is 688 bytes and the diagnostic counter
buffer is 513 uints / 2052 B; replay ABI 10 and the 120-byte inline push constants are unchanged.
All new addresses remain zero outside view 20, and this scratch is neither committed history nor an
estimator input.

Fresh Vulkan/RTX 5060 Ti runtime validation produced 46 readbacks. All 30471 winner writes
completed and all 30471 post-barrier pairs matched both reservoir and root bits. Empty receivers
totalled 8345209; empty-dirty tags, metadata rejects, reservoir/root mismatches, pair rejects and
every write/validation/partition delta were zero. The accepted split was 27605 selected plus 2866
retained, ages 1/2/3 were 10246/10164/10061, original source ownership was 2516 identity plus
27955 mapped, and all paths had one segment. The same readbacks included 60 true receiver
collisions, all fan-in two, so the exact-copy proof exercised the deterministic winner rather than
only unique claims. No Vulkan/device/shader failure occurred and the validation client was closed.

The follow-on winner replay gate replaces that single allocation with two independently owned
352 B/pixel ping-pong slots. `RtPathReservoirHistory` exposes the previous slot only for an adjacent
frame with matching generation and alternates slots independently from the branch scratch. The
current slot is cleared immediately before receiver ownership; the previous slot is read first and
is never modified. At 1280x673 the two slots total 606453760 bytes (578.36 MiB), and the complete
lazy shifted diagnostic GPU group is approximately 1728.51 MiB. `WorldPush` remains 688 B because
the previous-slot BDA occupies prior tail padding; counters are 528 uints / 2112 B, the diagnostic
sample buffer remains 14.375 MiB, inline push constants remain 120 B and replay ABI remains 10.

The log line `RT path guide branch winner previous replay` is the authority for this gate. Its
`attempted` count must equal the sum of receiver-reprojection reject, empty, metadata reject,
receiver-surface reject, source-reprojection reject, source-surface reject, source-replay reject,
selected accepted and retained accepted. Accepted output must also equal both the identity/mapped
source partition and the one/two-segment partition. Metadata reject and all printed deltas should
remain zero in a stable world; strict surface/reprojection/replay rejects are fail-closed and may
be non-zero during motion or scene changes. This diagnostic never performs another mapping,
updates committed history or affects the ordinary estimator.

The first fresh runtime proof used 27 readbacks at 569x320. Aggregate accounting was
4916160 attempted = 4849114 empty + 97 receiver-reprojection rejects + 50 receiver-surface
rejects + 52 source-reprojection rejects + 1 source-surface reject + 477 source-replay rejects +
66369 accepted. Metadata rejects and terminal delta were zero. Accepted output was 60520 selected
+ 5849 retained, original-source ownership was 5329 identity + 61040 mapped, and all 66369 records
had one replay segment; every printed partition delta was zero. No Vulkan/device/GPU/shader error
was logged. Do not relax replay tolerance to absorb the explicit 477 fail-closed rejects.

`RT path guide branch winner direct remap` is the next register/counter-only gate. It runs only
after a previous winner passed receiver/source reprojection and seeded replay. `remap` recomputes
current receiver guide validity, the replayed diffuse second-hit edge, geometry ratio, receiver/
source directional-PDF ratio, direct PSS Jacobian and receiver/source throughput ratio. `visibility`
uses the production shadow query; valid occlusion advances to a zero `target`. `weight` evaluates
the fresh shifted target against stored final weight and `min(sourceM, 8)`. The stored mapping
Jacobian is intentionally absent from the expression.

For a valid readback, replay accepted must equal remap eligible; each remap/visibility/target/weight
eligible count must equal its mutually exclusive terminal categories; remap ready must equal
visibility eligible; valid visibility must equal target eligible; target ready must equal weight
eligible; and weight ready must equal each M-cap, output-branch, source-mapping and segment
partition. Every printed delta and gateDelta should be zero. Edge rejects are expected for paths
without the supported continuous diffuse reconnection edge. The counter buffer is 557 uints /
2228 B; memory allocations, 688 B `WorldPush`, 120 B inline push constants and replay ABI 10 are
unchanged. No selection, RNG or payload/history/estimator write occurs.

Fresh runtime validation produced 23 readbacks: 209380 replay-accepted winners entered remap,
4099 unsupported edges rejected explicitly and 205281 reached ready. Guide/geometry/PDF/
throughput rejects were zero. Visibility was 205279 clear + 2 occluded; target and weight were
205279 positive + 2 zero, with zero invalid. M was uncapped for all 205281 weights. Output was
190021 selected + 15260 retained, original source was 15435 identity + 189846 mapped, and every
path had one segment. All printed deltas and gateDelta values were zero, with no Vulkan/device/GPU/
shader error. The validation client was closed and no Java process remains.

`RT path guide branch winner selection` continues directly from the valid winner weight. It reads
only the current receiver reservoir's weight sum and computes the uncapped overflow-stable relative
ratio; do not substitute the capped future weight sum as the denominator. Its deterministic
diagnostic draw hashes the receiver pixel, previous-winner pixel, stored frame tag, winner priority
and current frame. Neither the probability nor the draw mutates RNG, payload, history or estimator
state.

For a valid readback, weight ready must equal selection eligible and the mutually exclusive
selection terminal partition. Valid zero/open/one probabilities must equal both the current-weight
partition and Bernoulli eligible. Bernoulli eligible must equal selected + retained + invalid; after
excluding invalid, it must equal the probability-boundary, previous-output, source-mapping and
segment partitions. `violations[zero=0,one=0]` and every printed delta/gateDelta are mandatory.
The counter buffer is 582 uints / 2328 B; allocation sizes, 688 B `WorldPush`, 120 B inline push
constants and replay ABI 10 are unchanged.

Fresh runtime validation produced 36 readbacks / 116200 ready winners. Selection probabilities
were 28155 open + 88045 exact-one, with zero current rejects and zero invalid; current weights were
87999 zero + 28201 positive. Bernoulli produced 105180 selected + 11020 retained, zero invalid and
zero boundary violations. Previous output was 107039 selected + 9161 retained, original source was
9595 identity + 106605 mapped, and all records used one segment. All deltas were zero, no Vulkan/
GPU/shader error was logged, and the exact validation client was closed. The follow-on gate may
compute selected/retained post-selection weights and caps in registers only; it must still not write
winner scratch, committed path history or the ordinary estimator.

`RT path guide branch winner post-selection` mirrors the future weight lane but never constructs a
reservoir. The stored weight sum and M use pre-add saturation at `1e30` and `16777216`; do not reuse
either cap in the preceding Bernoulli denominator. Selected outcomes use the fresh shifted target,
retained outcomes use the current target, and non-empty records require a positive finite final
weight. Exact zero weight is valid only as an empty result with final weight zero.

For a valid readback, Bernoulli ready must equal post-selection eligible and its complete terminal
partition. Selected and retained terminal counts must individually equal their preceding Bernoulli
outcomes. After current/next validation, both weight-cap and count-cap partitions must equal the cap
eligible count. Final ready must equal each previous-output, source-mapping and segment partition.
Every `delta` and `gateDelta` must be zero. Counter storage is 603 uints / 2412 B; no allocation or
ABI size changes accompany this gate.

Fresh full-resolution runtime validation produced 12 readbacks / 137581 ready inputs: 125834
selected-ready, 11747 retained-ready and one retained empty result. All current/next/target/final
rejects were zero. Runtime weight and count caps were entirely uncapped; CPU reference tests cover
both saturation boundaries. Previous output was 128009 selected + 9572 retained, original source
was 9431 identity + 128150 mapped, and all records had one segment. All deltas were zero, no Vulkan/
GPU/shader error was logged, and the client was closed with no Java process remaining. The next
gate may assemble and validate only the future reservoir record in registers; root pairing,
scratch/history storage and ordinary-estimator use remain forbidden.

`RT path guide branch winner record` assembles the full future reservoir only in registers.
Selected output rewrites receiver-dependent lanes and the already independently reprojected source
key, while preserving source proposal/replay lanes. Retained output preserves current sample
metadata exactly. The common metadata validator is mandatory for both non-empty branches; selected
records must also pass the mapping-source replay comparator, and retained records must remain
identity mapped. The local record must never be written by this gate.

Valid accounting requires post-selection ready to equal record eligible and its selected/retained/
empty terminal partition. Rewrite, selected-preserve and source-key lane partitions each equal the
selected population; retained-preserve equals the non-empty retained population; weights and common
metadata each equal all non-empty records. Accepted record count must equal previous-output,
source-mapping and segment partitions. Every printed delta/gateDelta must be zero. The counter
buffer for this and the follow-on root-pair replay gate is 661 uints / 2644 B, with all allocation and ABI
sizes unchanged.

Fresh runtime validation produced 15 non-zero readbacks / 169988 eligible records, plus one excluded
zero readback during resize/reset. Results were 155362 selected-ready, 14622 retained-ready, 2
empty-ready and 2 explicit selected preserve-lane rejects. Rewrite, source-key, retained-preserve,
weight and metadata rejects were zero. Accepted ownership was 157843 previous-selected + 12143
previous-retained and 11805 identity + 158181 mapped original sources; every accepted record used
one segment. All deltas were zero and no Vulkan/GPU/shader error occurred. Keep the two preserve
rejects fail-closed. The next gate may construct the matching source root in registers; no storage,
history or estimator write is authorized.

`RT path guide branch winner pair` validates that register reservoir against its current-frame
`PathSourceRoot`. Selected records must have a ready current source key, exact selected-root
construction, valid segment chain and matching current source/receiver identities. Retained records
must capture a fresh current queue root and pass the same chain/identity checks. Valid accounting
requires `recordReady == eligible + empty`, `eligible == selected terminal + retained terminal`, each
root/key lane to cover its eligible branch exactly, and accepted count to equal previous-output,
source-mapping and segment partitions. All reported deltas must be zero.

The reference quick-play produced 18 readbacks / 100967 eligible pairs: 91637 selected-ready + 9330
retained-ready, with zero key/root/chain/identity rejects and exact ownership/segment accounting.
The client was then closed and no Java process remained. This gate performs no pair replay or write;
the next allowed step is direct register-only replay from the newly assembled root.

`RT path guide branch winner pair replay` retraces the pair directly from that root. Selected-ready
records must use the mapping-source comparator; retained-ready identity records must use the full
exact replay comparator. Valid accounting requires pair-ready to equal replay eligible and its
selected/retained terminal partition; accepted replay must equal every previous-output,
source-mapping and segment partition. All deltas and gate deltas must be zero.

The reference run produced 8 full-resolution readbacks / 89520 accepted pairs: 81254 selected +
8266 retained, zero replay rejects, and exact previous/source/segment partitions. Vulkan/device/GPU/
shader errors were absent; the client was closed and no Java process remained. The replay pair is
still not stored. The next allowed gate is bounded diagnostic storage plus exact post-barrier payload
equality, not committed history or estimator use.

`RT path guide branch winner pair storage` is that bounded storage gate. It captures at most 4096
replay-approved current-frame pairs per readback, then a separate ray-generation pass validates
metadata and compares every reservoir/root bit after a device barrier. `write.overflow` is expected
when eligible input exceeds 4096; correctness requires `completed + overflow == eligible`,
`validate.attempted == completed`, zero metadata/reservoir/root/pair rejects, exact branch/previous/
source/segment partitions, and every `delta`/`gateDelta` equal to zero.

The expected and stored 352-byte arrays intentionally alias the aged storage-audit arrays. Command
ordering guarantees that the winner-pair validator consumes them before the later aged writer may
reuse them. Do not reorder those passes without introducing independent storage or proving a new
lifetime. This reuse keeps the diagnostic sample buffer at 14.375 MiB; the counter buffer is 682
uints / 2728 B, `WorldPush` is 688 B, inline push constants are 120 B and replay ABI is 10.

The reference windowed/full-screen run produced 36 non-zero readbacks / 246013 eligible pairs:
125256 completed and validated, 120757 explicit capacity overflows, and zero metadata, reservoir,
root or pair rejects. Full-screen readbacks repeatedly reached exactly 4096 completed records. All
accounting deltas were zero; no Vulkan/device/GPU/shader error was logged, and the client process
tree was closed. This proves only bounded device storage equality. Before any dense candidate write,
define its ownership, clear/reset, generation and adjacent-frame lifetime contract; committed path
history and the ordinary estimator are still forbidden.

The original `RT path guide branch winner candidate owner` staging gate audited that ownership
contract without adding a buffer. The current winner write slot differs from the adjacent previous
read slot. Before replay,
only its 16 B/pixel tag tail is cleared; replay-approved output writes one tag at its own current
receiver index. The tag binds frame, 24-bit generation, receiver index, source/new-output/previous-
output mapping kinds and one/two-segment replay ownership. A full-resolution validator consumes the
tags before `beginCurrentBranchWinnerScratch` clears the complete slot for the existing aged-winner
pass. Do not move this validation after that clear or write payload before a separate storage gate.

Valid accounting requires replay accepted = write eligible = write completed = validate accepted;
validate attempted = empty + metadata reject + accepted; every accepted count must equal its branch,
previous-output, source-mapping and segment partitions. All deltas must be zero. The reference run
covered 8 full-screen readbacks / 7372800 attempted slots: 88355 tags accepted, 7284445 empty, zero
metadata rejects and zero deltas. Counter storage is 696 uints / 2784 B; allocations, `WorldPush`,
inline push constants and replay ABI are unchanged. The client was closed with no Java processes.

`RT path guide branch winner dense payload` extends that same temporary lifetime to the complete
176-byte `PathReservoir` and 160-byte `PathSourceRoot`. A replay-approved invocation stores both at
its receiver index and writes the owner tag last. The bounded storage validator then compares the
dense pair against the existing exact expected capture after a barrier. The full-screen owner pass
still validates every tag, and `beginCurrentBranchWinnerScratch` subsequently clears the entire
slot before its aged-winner use. Do not move the clear earlier, retain this payload after the clear,
or interpret it as committed history.

Correct accounting requires `write.eligible = write.completed = replayAccepted`, zero write delta
and gateDelta, and `validate.attempted = pair.accepted + pair.reject`. Metadata reject, reservoir
mismatch, root mismatch, pair reject and all validation deltas must be zero. The reference run wrote
11698/11698 pairs and matched all 4096 bounded dense samples bit-for-bit, split into 3759 selected
and 337 retained outputs. Counter storage is 714 uints / 2856 B; allocation sizes, 688 B `WorldPush`,
120 B inline push constants and replay ABI 10 are unchanged.

View 20 has two explicit operating modes. The default `visual` mode runs the shifted-radiance image
without the accumulated branch/winner proof chain, mapping validators or synchronous counter
readback. It is the mode for walking through a scene and inspecting the diagnostic. On the reference
RTX 5060 Ti full-screen responsiveness improved from about 1 FPS with the complete audit to about
16 FPS in visual mode. This is still a raw RT diagnostic without DLSS-RR, not an ordinary-render
benchmark.

The original proof workload is retained behind the startup property
`-Dcaustica.rt.view20FullAudit=true`. The supported project entry point is
`./runClient.ps1 -View20FullAudit`; the script appends the property to its intentionally rebuilt
`JAVA_TOOL_OPTIONS`. Supplying the environment variable before the script is not sufficient because
the script replaces it. `full audit` performs the multiple full-resolution replay/
validation passes, device barriers, readback waits and temporary full-payload writes needed for
counter evidence. Use only enough frames to obtain a clean readback, then close the client. The log
must say either `RT debug view 20 mode: visual` or `full audit`; never use the latter for manual scene
navigation. Both modes remain view-20-only, and ordinary rendering receives zero diagnostic scratch
addresses.

`RT path guide branch candidate arbitration` is the fresh-versus-aged ownership gate. It runs after
aged claims exist but before the temporary fresh payload is cleared. A valid adjacent-frame fresh
output always wins a collision; aged output is used only when fresh is absent. This prevents two
correlated merges that both consumed the current reservoir from being combined implicitly. The
original counter-only proof classified 861440 pixels into 845993 empty, 3652 fresh-only, 4003
aged-only and 7792 collisions won by fresh, with zero metadata rejects and deltas.

The mixed-payload gate applies that policy only inside the isolated current-winner
scratch. Fresh pairs remain in place; an aged-only owner copies its exact `PathReservoir` and
`PathSourceRoot` into the empty receiver slot. Every chosen pair is copied into the existing 4096-
entry bounded expected array, then a separate post-barrier pass compares the dense tag and every
payload bit. The high control bit and high generation byte record aged origin and exact age only in
that bounded expected copy. The dense tag is normalized for both origins to frame, 24-bit
generation, receiver owner, source/new-output/previous-output mapping kinds and replay segment
count; the temporary marker never reaches the dense slot.

Correct accounting requires chosen = capture cursor, capture cursor = completed + overflow, aged
chosen = aged write eligible = aged write completed, and validate attempted = metadata reject +
pair accepted + pair reject. Every accepted origin/branch/source/segment/age partition and all
deltas must be exact. Four reference full-audit readbacks offered 56201 chosen pairs, wrote all
16047 aged fallbacks, and checked 16384 bounded pairs (11552 fresh + 4832 aged) bit-for-bit. Metadata,
reservoir/root mismatch, pair reject and all deltas were zero. Counter storage is 756 uints / 3024 B;
allocation sizes, 688 B `WorldPush`, 120 B inline push constants and replay ABI 10 are unchanged.

The promotion gate removes the later full clear and the redundant aged-only winner rewrite. After
the mixed payload validator, the host commits that exact write slot directly to the independent
winner ping-pong. The adjacent frame reprojects the receiver, requires the stored owner to equal
the reprojected previous pixel, validates all mapping/segment bytes, independently reprojects the
immutable source root and repeats strict seeded replay. The deterministic winner draw is bound to
the complete promoted control word rather than to the retired aged priority. The three legacy
aged-only validators are not dispatched; their summaries are logged only if explicitly re-enabled,
so reused claim counters cannot create a false delta. This remains view-20 full-audit-only and
cannot write committed path history or the ordinary estimator.

Eleven reference readbacks attempted 2002880 adjacent-frame replays. They accepted 433019 records
(427210 selected and 5809 retained), including 14454 identity-source and 418565 mapped-source
records. Exactly 1275 strict source-replay rejects and one receiver-surface reject failed closed;
metadata rejects and every terminal/branch/source/segment delta were zero. In the producing frames,
435959 mixed candidates included 4344 aged fallback writes; all 45056 bounded samples matched every
reservoir/root bit, including 671 aged samples. No Vulkan/device/GPU/shader failure was logged.
Counter storage, allocation sizes, `WorldPush`, inline push constants and replay ABI are unchanged.

`RT path guide branch winner persistence policy` is the final current-frame boundary before the
host publishes the isolated winner ping-pong slot. It scans every receiver after the mixed payload
barrier and is read-only. For a valid static scene, all lifecycle/control/reservoir/source-key/
root-chain/source-root/receiver-root reject counters should be zero. Required identities are
`attempted = empty + rejects + accepted`, `accepted = selected + retained`, `accepted = identity +
mapped`, and `accepted = one + two`; every printed `delta` must be zero. Source and receiver root
checks are performed independently against the current guides, so a mapped reservoir cannot stand
in for missing source-root provenance.

The reference full-audit run produced five readbacks at 569x320: 910400 attempted slots, 735697
empty and 174703 accepted pairs (170662 selected, 4041 retained; 8364 identity-source, 166339
mapped-source), with zero rejects and exact accounting. Counter storage is 772 uints / 3088 B;
allocation sizes, 688 B `WorldPush`, 120 B inline push constants and replay ABI 10 are unchanged.
This line proves only the view-20 diagnostic policy. It does not indicate that committed path
history or the ordinary estimator consumes the mixed population.

For a fresh runtime check, use debug view 20 and inspect `run/logs/latest.log`. Normal operation
requires `RT bring-up OK`, Vulkan, the intended NVIDIA device, exact zero-delta counter partitions,
and no `DEVICE_LOST`, `VK_ERROR`, GPU fault or shader compilation error. Debug colors and sparse
white/colored pixels are diagnostic states, not final render quality. Mojang/Realms 401 messages are
unrelated authentication noise.

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
