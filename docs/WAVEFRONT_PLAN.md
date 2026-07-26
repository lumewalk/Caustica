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

## 1. Design

Two raygen entry points in the same RT pipeline, selected by SBT record at dispatch, connected by the
existing guide images plus one new continuation buffer.

**Pass A — `world_primary.rgen`.** Camera ray; walk the dielectric chain while `diffuseDepth == 0`;
capture guides; resolve spec motion and the transmitted guide; write the six guide images. Emit one
continuation record per surviving path (two when a dielectric splits). Does **no** shading.

**Pass B — `world_indirect.rgen`.** One thread per record. Runs the bounce loop with NEE / RIS / SSS /
GGX continuation. Never touches `gv_*`.

`PathSegment` from `b4a0e7b` is already the record; that commit was deliberately shaped for this.

### What each pass does NOT contain

- Pass A: no NEE, no RIS, no SSS, no GGX sampling, no Russian roulette.
- Pass B: no guide capture, no `resolveTransmissionGuide`, no `specularReflectionMotion`, no
  `waterWaveGradTemporal`, no debug views.

### Accepted duplication

Dielectric transport (~90 lines) and the non-temporal wave normal appear in both passes: pass A walks
the primary chain, pass B must still handle a reflection ray hitting water. This is real and worth
paying — the block it buys separation from (RIS + NEE + SSS + GGX) is several times larger.

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
| `bounce`, `diffuseDepth`, `showCelestial`, `maySplit`, pixel index | packed uint x2 | 8 |

At 1280x720 render resolution: 921600 x 48 B = **44.2 MB** for one record per pixel. Use an **append
buffer with an atomic counter** sized ~1.25x pixel count (~55 MB) rather than a fixed 2-per-pixel
array — splits are the exception, and an append buffer also hands pass B a dense, coherent queue.
When the queue is full, fall back to the stochastic branch instead of splitting: graceful, and the
estimator stays unbiased.

**Open:** 44-55 MB is not free on 8 GB cards. If it bites, the fallback is to keep pass B in the same
dispatch for the common single-segment case and only spill splits — measure before deciding.

## 3. Phases

- **M0 — plumbing.** Generalize `RtPipeline.create` to take `String[] rgen` (same pattern the `rmiss`
  array already uses) and select the raygen SBT record at dispatch. Add a second raygen that is a
  copy of the current one. Play-test: pixel-identical output, one extra dispatch.
- **M1 — split, no branching.** Move primary/guide work to pass A, bounce loop to pass B, one record
  per pixel, deterministic split temporarily disabled (falls back to stochastic). Play-test + profile.
  This is the milestone that proves or kills the approach.
- **M2 — splits back.** Re-enable the deterministic Fresnel split as a second appended record.
  Play-test for energy parity against `b4a0e7b`.
- **M3 — measure.** Expect: raygen live state well under `main`'s 308 B baseline, NOINST gone,
  traversal LGSB better hidden via higher occupancy. **Baseline to beat is `db6418b` at 21 ms**, not
  `b4a0e7b` at 47 ms — beating the regression proves nothing.
- **M4 — later.** ReSTIR spatial reuse becomes a third pass over the G-buffer. This is the reason the
  split is worth doing even if M3 is only neutral: spatial reuse is inherently a screen-space
  multi-pass algorithm and would otherwise be bolted onto a megakernel.

## 4. Risks / kill criteria

- **Bandwidth vs occupancy.** The whole bet is that removing live state buys more than the record
  traffic costs. M1 must show it. If M1 does not beat `db6418b`'s 21 ms, revert to `db6418b`'s
  structure — `b4a0e7b` is not a fallback, it is a regression being carried deliberately while the
  split lands.
- **Reproducing the `b4a0e7b` failure through a buffer.** If pass A holds the record live across its
  own traces instead of writing and exiting, it pays the same 288 B live range *plus* the memory
  traffic. Check the live-state CSV for pass A, not just the frame time.
- **Pass B incoherence.** Records start incoherent, but they already are today; SER still applies
  inside pass B, and a dense queue is strictly better than a sparse screen dispatch.
- **SER interaction is an open question** independent of this work — `GPU_PERF_PLAN.md` §0 flags that
  the scheduler itself costs 7.7% and a no-reorder A/B has never been run. Do that A/B *before* M1 so
  the two changes are not entangled.
- **Two dispatches means a barrier**; trivial next to the trace cost, but it serialises pass A/B, so
  any pass A tail latency is exposed.

## 5. Status

- [x] Step 2 (single instantiation + guide hoist) — `b4a0e7b`, **GPU-tested: 21 ms -> 47 ms, REGRESSION**
      (kept deliberately; the split is its fix, see §0.1)
- [x] M0 plumbing — `d76f450`, multiple raygens per pipeline, no behaviour change
- [ ] no-reorder A/B (prerequisite, independent; intentionally skipped for this work)
- [x] M1 split — 48 B packed records, primary/guide + indirect dispatches, stochastic dielectric
      fallback. **GPU-tested: primary 4.4 ms + indirect 9.8 ms = 14.2 ms**, versus the 21 ms baseline.
- [ ] M2 splits restored
- [ ] M3 measure
