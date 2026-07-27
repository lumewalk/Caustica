# Wavefront Split Plan — primary/guide pass + indirect pass

Plan of record started 2026-07-26, `pq` @ `b4a0e7b`. Supersedes nothing; complements
`docs/GPU_PERF_PLAN.md` (whose P-A/P-B/P-C ray-count and locality levers stay valid and orthogonal).

## 0. Why

Profile `run/nsight-profile/live-state-5.csv`, pq @ `c22ebf3`, against `main`:

| Callsite | main | pq @ c22ebf3 | contexts (pq) |
|---|---|---|---|
| primary radiance trace | 308 B / 30 v | **2932 B / 42 v** | 2 |
| `visibility()` (shadow) | 391 B / 39 v | 862 B / 77 v | 10 (5 sites x 2) |

Top stall flipped `LGSB` -> **`NOINST`** (instruction fetch), i.e. the shader outgrew its I-cache.

`b4a0e7b` fixed the duplication half of this: continuations became data (`PathSegment`), `tracePath`
is instantiated once behind a `[loop]`, and `world.rgen.spv` went **1017692 -> 645972 bytes (-36.5%)**.

### 0.1 `b4a0e7b` regressed: 21 ms -> 47 ms (`continuation b4a0e7b.csv`)

It made things **worse**, badly. Contexts did drop 2 -> 1 as designed, but live state at the primary
trace went **2932 B -> 3347 B**, and LGSB rose sharply. Top contributors:

| B | source | what |
|---|---|---|
| 504 | `world.rgen.slang:149` | medium extinction (was 336 B) |
| 432 | `:1316` | `n = payload.normal` |
| 324 | `:1516` | `gv_hitCamRel` |
| **288** | **`:1400`** | **`pending = makePathSegment(...)`** |
| 156 / 120 | `:1739` / `:1738` | `dir` / `origin`, now loop-carried |

The mechanism: **`pending` is created at the primary dielectric and cannot be consumed until
`tracePath` returns, so it survives every subsequent trace in registers.** `[loop]` additionally
blocks specialisation, forcing main's own state to become loop-carried. Halving the instruction
footprint bought nothing because occupancy — not instruction fetch — is the binding constraint.

**This is `GPU_PERF_PLAN.md` §0's rule confirmed a second time: in this kernel, reducing instructions
at the cost of live state is a losing trade.** The RIS wave-batching revert was the first instance.

It also converts this plan from a bet into a targeted fix. Pass A **writes the record to memory and
exits**; pass B **reads it once at entry**, where it becomes ordinary loop state. Neither pass holds a
continuation live across a trace, which is precisely what `b4a0e7b` got wrong. M1 must preserve that
property or it will reproduce the same regression through a more expensive mechanism.

The megakernel still contains two workloads with opposite characteristics and it keeps growing:

| | primary / guide | indirect / shading |
|---|---|---|
| runs | once per pixel | once per bounce, in a loop |
| coherence | screen-coherent | incoherent |
| owns | guides, spec surface + MV, transmission walk, wave temporal derivative | NEE, RIS, SSS, GGX continuation, RR |

Everything inlined into one entry point means the hot loop carries the primary code's instruction
footprint even though it is dead after bounce 0, and the primary code carries the loop's live state.

This also aligns with the standing lesson in `GPU_PERF_PLAN.md` §0: **reduce live state, never add
it.** rgen was at 126 live registers with occupancy as the binding constraint, and traversal (21.5%
of samples, 89% LGSB) is poorly hidden *because* of that. Splitting the kernel is the structural
version of that lesson; the RIS wave-batching attempt failed because it did the reverse.

### 0.2 M1 result: 21 ms -> 14.2 ms

M1 landed in `428922d` and was subsequently split into explicit per-pass modules (`62e8aae`,
`cb5a51d`, `27e10d0`). GPU timing:

| pass | time |
|---|---:|
| primary / guide | 4.4 ms |
| indirect / shading | 9.8 ms |
| **combined** | **14.2 ms** |

That is **6.8 ms / 32.4% faster** than the 21 ms `db6418b` baseline. The split therefore clears its
kill criterion and confirms the live-state diagnosis. Pass A now runs without SER; a full no-reorder
A/B for Pass B remains outside M2.

## 1. Design

Two raygen entry points in the same RT pipeline, selected by SBT record at dispatch, connected by the
existing guide images plus one new continuation buffer.

**Pass A — `world_primary.rgen`.** One invocation and exactly one camera sample per pixel, independent
of configured SPP. Trace the camera ray once, capture the foreground guides, and emit either the
pre-hit continuation or one/two post-interface continuations. At the first eligible dielectric, write
reflection and transmission records immediately and stop radiance traversal. Only deterministic
reflection/refraction guide probes may trace farther. Pass A uses ordinary `TraceRay`, with no
invocation-reorder capability or barrier.

**Pass B — `world.rgen`.** One invocation per pixel. Load each valid leaf and resample its terminal
continuation `worldPush.spp` times with decorrelated seeds, run NEE / RIS / SSS / GGX continuation,
sum the leaves, divide by SPP, and write the pixel once. It never touches `gv_*`.

`PathSegment` from `b4a0e7b` is already the record; that commit was deliberately shaped for this.

### What each pass does NOT contain

- Pass A: no NEE, no RIS, no SSS, no finite GGX sampling, and no radiance trace after the camera hit.
- Pass B: no guide capture, no auxiliary guide walks, no `specularReflectionMotion`, no
  `waterWaveGradTemporal`, no debug views.

### Accepted duplication

First-interface dielectric math and the non-temporal wave normal remain in both passes: Pass A creates
the first split, while Pass B owns all radiance transport after it and can hit later water/glass.

### 1.1 M2 transport boundary

The ownership rule is now **first dielectric interface only**.

- Water and `MATERIAL_DIELECTRIC` are modeled as exact Fresnel interfaces. Pass A splits the first
  non-TIR interface and queues both weighted post-interface states. Degenerate F=0/F=1 interfaces queue
  their sole continuation. Pass B owns every subsequent radiance trace and dielectric decision.
- All opaque surfaces, including non-emissive pure delta mirrors, terminate Pass A at the pre-hit
  continuation and are handled by Pass B.
- A polished opaque dielectric with both diffuse and exact-specular response is **not** the same split
  as glass. Glass has reflection `F` and transmission `1-F`; the current opaque BSDF is additive
  diffuse plus specular. Multiplying its diffuse branch by `1-F` would change energy. Mixed
  diffuse+delta opaque surfaces therefore remain in Pass B for M2 unless a record flag explicitly
  requests diffuse-only terminal shading.
- Every finite glossy lobe stays in Pass B. Its destination is a distribution, not a stable guide
  feature, and moving it would destroy Pass A's coherence without producing a valid sharp guide.

There is no carried `diffuseDepth`. Pass B starts an `indirectDepth` counter at zero for each queued
leaf and advances it on every hit it observes, regardless of material or selected lobe. The
primary/interface prefix consumed by Pass A remains outside this budget; once handed off, all hits are
treated uniformly for depth gating. RIS remains active with the full configured candidate count at every
Pass B hit. Pass B hits 0 and 1 retain full SSS, with SSS removed beginning at hit 2. This counter is
deliberately independent of packed `bounce`, so split records beginning at transport bounce 1 do not
lose first/second-hit SSS quality.

### 1.2 M2 guide ownership

Once Pass A deterministically owns the first dielectric split, dedicated deterministic guide probes
supply the endpoints without affecting queued radiance:

| leaf | ordinary guide | specular guide |
|---|---|---|
| deterministic refracted probe | `gAlbedo`, `gMotion`, destination depth/normal | — |
| deterministic reflected probe | — | reflected endpoint for `gSpecMotion` |

- The refracted guide accumulates a `guideFilter` only for **entering stained-glass/dielectric volumes**, multiplying the
  terminal diffuse albedo by `payload.albedo`. The closest-hit shader has already performed the desired
  texture-alpha blend (`lerp(white, texture*tint, alpha)`). Applying it once on entry avoids counting a
  block's front and back faces twice. Water is excluded: its payload tint parameterizes Beer extinction.
- Keep live guide state and the `rgba16f` / `rg16f` storage-image bindings FP32. The ray payload already
  packs hit attributes as `half3`, so changing `gv_*` only added another precision boundary. A
  half-typed image binding was also ineffective: Slang canonicalized it to `OpTypeImage %float` and
  widened values before `OpImageWrite`; the Vulkan image store owns the final format conversion.
- Keep distance-dependent Beer–Lambert extinction in radiance/lighting, not `gAlbedo`; otherwise the
  material guide varies with path length and disagrees with RR's material demodulation.
- The baseline `gSpecAlbedo` remains average view-dependent material reflectance
  (`rrSpecularAlbedo` / interface Fresnel). The reflected terminal supplies motion, not albedo.
  “Reflected diffuse content × reflection strength” is an experimental, flag-gated follow-up because it
  is not the documented guide identity and is unstable for emissive destinations, metals (`diffAlb=0`),
  sky, and mirror recursion. It requires a material-reflectance fallback and an in-place A/B.
- `resolveTransmissionGuide` deterministically refracts through later interfaces and never follows a
  reflected branch into ordinary albedo/depth. TIR freezes the ordinary tuple on that interface.
  Reflection motion uses its own one-ray guide probe. These are the only traces Pass A performs after
  queuing the radiance split. The refracted walk uses `worldPush.maxBounces` as its crossing limit; it
  has no separate guide-only cap.

## 2. Record layout

Target 48 B. Unpacked `PathSegment` is ~100 B; the packing below is lossless where it matters and the
`medium` outer slot is the only speculative squeeze.

| field | packed | B |
|---|---|---|
| `ro` | float3 (rebased world) | 12 |
| `rd` | octahedral unorm16x2 | 4 |
| `throughput` | rgb9e5 | 4 |
| `medium.current` | ior half + extinction rgb9e5 | 6 |
| `medium.outer` | ior half + extinction rgb9e5 | 6 |
| `rayConeWidth` / `rayConeSpread` | half x2 | 4 |
| `seed` | uint | 4 |
| `bounce`, `showCelestial`, medium/valid flags, next-record link | packed uint x2 | 8 |

M1 allocated one fixed record per render pixel **per SPP sample**:

`baseRecords = renderWidth * renderHeight * spp`

At 1280x720 and SPP 1 this is 921600 x 48 B = **44.2 MB (42.2 MiB)**. At SPP 8 it was
353.9 MB (337.5 MiB).

M2 keeps the race-free ownership model: one Pass B invocation writes one pixel. It does **not** launch
one indirect raygen per appended record, because two records for one pixel would race on `outImage`
without another radiance buffer and reduction pass.

Pass A is fixed at SPP 1 and can emit at most one second leaf. Configured SPP belongs solely to Pass B,
which resamples the stored leaf or leaves.

Use a base-plus-fixed-secondary queue:

1. Every pixel owns one fixed base record.
2. Every pixel owns one fixed secondary record at `baseRecords + pixelIndex`.
3. The base record's currently unused `pixelSample` word becomes a `nextRecord` index/sentinel.
4. Pass B loads the base record and then its optional linked record, sums both locally, and writes once.
5. No atomic allocation, queue header, reset, overflow path, or transfer-to-trace barrier is required.

The allocation is always `pixelCount * 2 * 48 B`: 88.5 MB (84.4 MiB) at 1280x720, independent of
configured SPP. A storage texture would need three `RGBA32UI` texels for the same 48-byte record and
would not reduce traversal work, so the queue remains a linear BDA buffer.

At the split, Pass A writes the transmission record to the fixed secondary slot and returns the
reflection record as the base. It does not trace or reload either branch. Pass B follows the link and
owns later stochastic interfaces, bounding the queue at two leaves without carrying branch state across
a Pass A radiance trace.

## 3. Phases

- **M0 — plumbing.** Generalize `RtPipeline.create` to take `String[] rgen` (same pattern the `rmiss`
  array already uses) and select the raygen SBT record at dispatch. Add a second raygen that is a
  copy of the current one. Play-test: pixel-identical output, one extra dispatch.
- **M1 — split, no branching.** Move primary/guide work to pass A, bounce loop to pass B, one record
  per pixel/sample, deterministic split temporarily disabled (falls back to stochastic). Play-test + profile.
  This is the milestone that proves or kills the approach.
- **M2.1 — transmitted-guide filter.** Apply entry-only stained-glass alpha/tint to the terminal
  ordinary albedo guide. Exclude water and Beer extinction. Validate in debug view 2.
- **M2.2 — linked secondary + dielectric split.** Give every pixel one directly indexed secondary
  record, split the single Pass A sample once at the first visually-primary Fresnel interface, write both
  post-interface continuations, and return without tracing either.
- **M2.3 — Pass-B transport/depth.** Keep opaque delta mirrors and every post-split radiance trace in
  Pass B, remove carried `diffuseDepth`, and count every hit observed by Pass B.
- **M2.4 — deterministic auxiliary guides.** Use dedicated reflected/refracted guide probes after the
  first hit. Never let stochastic radiance choices or reflected content enter ordinary guides.
- **M2.5 — experimental reflected-content guide.** Optional runtime A/B only. Modulate reflected
  terminal diffuse content by material reflectance with explicit fallbacks for sky/emissive/metal/mirror.
  Do not make this the default based only on another game's debug buffer.
- **M3 — validate and measure.** Play-test energy parity against the stochastic M1 reference, debug all
  six guides, profile Pass A/Pass B and Pass A live state, and compare combined time against both M1's
  **14.2 ms** and `db6418b`'s 21 ms. M2 must not give back the structural M1 win.
- **M4 — later.** ReSTIR spatial reuse becomes a third pass over the G-buffer. This is the reason the
  split is worth doing even if M3 is only neutral: spatial reuse is inherently a screen-space
  multi-pass algorithm and would otherwise be bolted onto a megakernel.

## 4. Risks / kill criteria

- **M1 regression budget.** M1 proved the structure at 14.2 ms. M2 is killed or redesigned if branch
  work/live state materially gives that win back; 21 ms is no longer an acceptable success threshold.
- **Measured M2 baseline.** The first linked-spill implementation measured **Pass A 8.6 ms + Pass B
  11.1 ms = 19.7 ms**, versus M1's 14.2 ms. The 4.2 ms Pass A increase identifies duplicated
  traversal as the primary cost; changing the 48 B linear buffer to a storage texture cannot remove it.
- **Reproducing the `b4a0e7b` failure.** Pass A must not trace either queued radiance branch. Check its
  live-state CSV as well as frame time.
- **Pass A tail latency.** Only deterministic guide probes may extend beyond the camera hit. Profile
  deep glass/water guide walks separately.
- **Queue footprint.** The fixed secondary mapping removes atomics and overflow but costs one extra
  48-byte record per pixel even when no split occurs. Watch VRAM at high render resolution and SPP.
- **Opaque energy mismatch.** Never reuse dielectric `(F, 1-F)` weights for an additive opaque
  diffuse+specular BSDF. Moving mixed surfaces requires an explicit diffuse-only terminal contract.
- **Rough dielectric gap.** Current water/glass transport is exact-delta in both passes. A real glossy
  dielectric is a separate BSDF feature, not something M2 gets merely by comparing roughness.
- **Spec-guide experiment.** Destination albedo is not material reflectance. Keep it flag-gated with
  robust fallbacks and judge RR stability, not only debug-view appearance.
- **SER is Pass B only.** Pass A is compiled without an invocation-reorder capability and uses ordinary
  `TraceRay`. A full no-reorder A/B for the larger indirect shader remains a separate experiment.
- **Two dispatches means a barrier**; trivial next to the trace cost, but it serialises pass A/B, so
  any pass A tail latency is exposed.

## 5. Status

- [x] Step 2 (single instantiation + guide hoist) — `b4a0e7b`, **GPU-tested: 21 ms -> 47 ms, REGRESSION**
      (kept deliberately; the split is its fix, see §0.1)
- [x] M0 plumbing — `d76f450`, multiple raygens per pipeline, no behaviour change
- [x] Pass A no-SER — ordinary `TraceRay`; emitted SPIR-V has no reorder capability/instructions
- [ ] Pass B no-reorder A/B (independent; intentionally deferred)
- [x] M1 split — 48 B packed records, primary/guide + indirect dispatches, stochastic dielectric
      fallback. **GPU-tested: primary 4.4 ms + indirect 9.8 ms = 14.2 ms**, versus the 21 ms baseline.
- [x] M2.1 transmitted-guide filter — implemented; shader validation passed, GPU guide-view check pending
- [x] M2.2 linked secondary + one deterministic dielectric split — implemented with one fixed slot
      per pixel, no atomics/overflow, Pass A fixed at one sample, and no post-split radiance trace in
      Pass A; GPU profile pending
- [x] M2.3 Pass-B transport + local per-hit depth — opaque delta and all post-split transport remain
      in Pass B; GPU parity check pending
- [x] M2.4 deterministic auxiliary guides — reflected/refracted guide probes are decoupled from
      stochastic radiance and reflected content never enters ordinary guides; GPU parity check pending
- [ ] M2.5 reflected-content `gSpecAlbedo` experiment (flagged, non-default)
- [ ] M3 GPU validation and measurement

Current worktree verification: Gradle test suite and SPIR-V validation pass. The emitted
`PackedPathSegment` array stride is still 48 B and `nextRecord` remains at byte offset 44.

Initial M2 measurement: **8.6 ms Pass A + 11.1 ms Pass B = 19.7 ms total**. Keep the queue buffer-backed:
mapping one record to a texture needs three `RGBA32UI` texels (the same 48 B payload) and does not
address the dominant extra traversal. Re-profile the leaf-owned guide path and fixed-SPP-1 Pass A.
