// Perspective path tracer. invViewProj = inverse(proj * viewRot) maps clip to camera-relative world.
// Primary rays are reconstructed from two unprojected points (near and far plane) rather than assuming
// the eye is at the origin: Minecraft bakes view bobbing into the projection, so the effective eye comes
// from the matrix. camOffset shifts camera-relative space into the terrain's rebased coordinates. Depth
// is reversed-Z (near=1, far=0).
//
// Lighting is an iterative path-tracing loop in raygen, not recursion from closest-hit, so
// maxRayRecursionDepth stays 1. At each surface vertex:
//   * NEE toward the sun — a single analytic directional light, gated by one hit-object visibility query
//     (TerminateOnFirstHit | SkipClosestHit). Lambertian BRDF = albedo/PI.
//   * a cosine-weighted Lambertian continuation ray (throughput *= albedo; the PI and cos cancel).
// Sky / ambient light needs no separate AO pass: continuation rays that escape land in world.rmiss,
// which returns sky radiance — so soft sky fill and contact AO both fall out of the path integral.
// Russian roulette terminates deep paths. Output is HDR (R16G16B16A16_SFLOAT); the HDR -> LDR tonemap
// happens later at the display.comp seam.
//
// SER: the build emits EXT and NV shader-invocation-reorder variants from this source.
//
// Each frame emits a single SPP-averaged (still noisy) estimate; temporal convergence is the denoiser's
// job. With RR disabled the output is the raw noisy path trace — a reference view, not a converged image.
import world_common;

[[vk::push_constant]] WorldPushConstants pc;

[[vk::binding(0, 0)]] RaytracingAccelerationStructure topLevelAS;
// HDR trace target: linear radiance, may exceed 1. Tonemap happens later at the display.comp seam.
[[vk::binding(1, 0)]] [format("rgba16f")] RWTexture2D<float4> outImage;
// Guide buffers — first-hit (primary-visibility) attributes consumed by the denoiser/DLSS-RR. Written
// every frame regardless of accumulation. gNormal.w carries LINEAR roughness (= GGX alpha, which is what
// DLSS-RR documents for this input); gDepth carries HW reversed-Z depth.
[[vk::binding(3, 0)]] [format("rgba16f")] RWTexture2D<float4> gNormal;     // xyz world normal, w linear roughness
[[vk::binding(4, 0)]] [format("rgba16f")] RWTexture2D<float4> gAlbedo;     // rgb diffuse albedo
[[vk::binding(5, 0)]] [format("r32f")]    RWTexture2D<float>  gDepth;      // HW reversed-Z depth
[[vk::binding(6, 0)]] [format("rg16f")]   RWTexture2D<float2> gMotion;     // screen-space motion (render px)
[[vk::binding(7, 0)]] [format("rgba16f")] RWTexture2D<float4> gSpecAlbedo; // specular albedo (0 — diffuse-only)
[[vk::binding(8, 0)]] [format("rg16f")]   RWTexture2D<float2> gSpecMotion; // reflection MVs for RR

// Per-frame push data loaded once from the BDA buffer at the top of main() (same pattern as the GLSL
// `worldPush` global). Layout constants generated from this module's SPIR-V — see world_common.WorldPush.
static WorldPush worldPush;

// The radiance payload remains module-level so tracePath and its guide helpers can share it across
// HitObject trace/invoke calls, mirroring the GLSL rayPayloadEXT global.
static Payload payload;

struct VisibilityResult {
    float3 transmittance;
    float waterHitT;
};

// First-hit (primary-visibility) guide attributes, captured at bounce 0 of tracePath. The primary ray
// is deterministic (no AA jitter yet), so every SPP sample's bounce 0 yields identical values.
static float3 gv_normal;
static float3 gv_albedo;
static float gv_rough;
static float gv_emission;
static uint gv_emissionSource;
// Linear VIEW-SPACE depth guide inputs; sky resolves to a large far value (no negative sentinel).
static float gv_depth;
static float3 gv_hitCamRel;       // primary-surface hit position relative to the current camera
static float3 gv_motionHitCamRel; // motion-guide hit position; water tracks the refracted hit instead
static bool gv_motionUseRefracted; // true when the MV tracks the refracted hit (its own reprojection delta)
// World-space displacement of the motion-guide surface since the previous frame (0 for static
// terrain/sky, an entity's per-vertex/object motion for a dynamic hit).
static float3 gv_motionObjDisp;
// Transmission replaces the ordinary guide tuple with the visible destination, so RR's specular inputs
// (specular albedo + reflection MV) need the foreground glass/water interface kept separately. This is
// the surface the reflection physically lives on; for an opaque primary hit it is that hit itself.
struct SpecSurface {
    float3 camRel;         // interface position relative to the current camera
    float3 normal;         // shading normal (wave-perturbed for water)
    float3 previousNormal; // the same normal one frame ago; equal to `normal` unless the surface animates
    float3 biasNormal;     // geometric normal, used only to offset a ray origin off the surface
    float roughness;
    float3 albedo;         // specular albedo fed to DLSS-RR for demodulation (0 = pure diffuse)
};
static SpecSurface gv_spec;

// Static surfaces reuse one normal for all three roles. Water overrides them (wave-displaced shading
// normal now and last frame, flat geometric normal for the bias) via the six-argument form.
SpecSurface makeSpecSurface(float3 camRel, float3 normal, float3 previousNormal, float3 biasNormal,
                            float roughness, float3 albedo) {
    SpecSurface s;
    s.camRel = camRel;
    s.normal = normal;
    s.previousNormal = previousNormal;
    s.biasNormal = biasNormal;
    s.roughness = roughness;
    s.albedo = albedo;
    return s;
}

SpecSurface makeSpecSurface(float3 camRel, float3 normal, float roughness, float3 albedo) {
    return makeSpecSurface(camRel, normal, normal, normal, roughness, albedo);
}

// Per-call control/results for the primary dielectric estimator. Ordinary pixels still trace once;
// a primary glass/water hit traces both Fresnel continuations and recombines them exactly.
static uint pathPrimaryDielectricBranch;
static bool pathPrimaryDielectricHit;
static float pathPrimaryDielectricF;

uint payloadMaterial() { return payload.flags & PAYLOAD_MATERIAL_MASK; }
uint payloadEmissionSource() {
    return (payload.flags >> PAYLOAD_EMISSION_SOURCE_SHIFT) & PAYLOAD_EMISSION_SOURCE_MASK;
}
// Set by world.rchit on any dielectric hit (see PAYLOAD_DIELECTRIC_ENTERING) — whether the ray was
// travelling into the volume (vs. out of it) at this hit's face.
bool payloadDielectricEntering() { return (payload.flags & PAYLOAD_DIELECTRIC_ENTERING) != 0u; }
bool payloadEmitterInList() { return (payload.flags & PAYLOAD_EMITTER_IN_LIST) != 0u; }
// LINEAR roughness, i.e. GGX alpha directly — NOT perceptual roughness. This is the one convention used
// end to end: LabPBR defines roughness = (1 - perceptualSmoothness)^2 and RtLabPbr.decode stores exactly
// that, RtMaterials.Profile carries the same units, and DLSS-RR wants linear roughness in its guide. So
// nothing squares it: ggxD/ggxG1/sampleGGXVNDF/rrSpecularAlbedo all take this value as alpha as-is, and
// any threshold compared against it is an alpha threshold.
float payloadRoughness() { return unpackHalf2(payload.roughMetal).x; }
float payloadMetalness() { return unpackHalf2(payload.roughMetal).y; }
float payloadEmission() { return unpackHalf2(payload.emissionSss).x; }
float payloadSss() { return unpackHalf2(payload.emissionSss).y; }
float payloadIor() { return unpackHalf2(payload.iorTransmission).x; }
float payloadTransmission() { return unpackHalf2(payload.iorTransmission).y; }

static const float PI = 3.14159265359;
static const float INV_PI = 0.31830988618;

// The analytic sky (gradient + sun/moon discs + stars) is computed in world.rmiss. On a miss, raygen
// reads payload.albedo and accumulates that sky radiance. The directional NEE light is pushed separately
// as worldPush.lightDir / worldPush.lightRadiance / worldPush.sunDir.
static const float SURF_BIAS = 0.005;  // offset secondary-ray origins along the normal (anti-acne)
static const float RAY_TMIN = 0.001;   // tiny tmin: the normal offset already clears the surface

// Water is a smooth dielectric. IOR 1.333; absorption is per-block Beer-Lambert extinction.
static const float WATER_IOR = 1.333;
static const float3 SKY_SPEC_ALBEDO = float3(0.5, 0.5, 0.5); // RR guide default for sky pixels.
static const float3 SKY_DIFF_ALBEDO = float3(0.5, 0.5, 0.5);

// Transmitted-ray origin offset when crossing a face from the TRANSLUCENT terrain layer (glass, ice).
// Must be SMALLER than the mesher-side TRANSLUCENT_INSET (RtTerrainMesher, 2e-4 blocks): those quads are
// recessed into their block by that inset, so a face touching a slab/stair leaves a tiny gap before the
// neighbour's surface. A large bias (SURF_BIAS) would restart the ray past that neighbour and see
// through it; this lands the ray inside the gap so the neighbour is still hit, while still clearing the
// surface it left to avoid self-intersection. Water is meshed by RtFluidMesher with no inset, so it uses
// the ordinary bias.
static const float INSET_TRANSMIT_BIAS = 1.0e-4;

// Ray-cone texture LOD: raygen tracks a one-pixel footprint along the path, and closest-hit converts it
// to texture mip levels from the local UV/world-space scale. Diffuse/glossy bounces widen the cone so
// indirect texture fetches don't keep sampling mip 0 through broad lobes.
static const float RAY_CONE_MIN_SPREAD = 1.0e-5;
static const float RAY_CONE_MIN_WIDTH = 1.0e-5;
static const float RAY_CONE_DIFFUSE_SPREAD = 0.25;
static const float RAY_CONE_GLOSSY_SPREAD_SCALE = 0.35;

// Per-channel Beer–Lambert extinction from a water body's biome tint (carried in payload.albedo for
// water hits, or worldPush.waterParams.xyz for the camera's own biome when starting submerged). A blue ocean
// tint (low red) absorbs red fastest → bluer with depth; swamp green-brown shifts the hue. The floor
// keeps even a white-tinted body very slightly absorbing so deep water never reads as clear vacuum.
static const float WATER_DENSITY = 0.05;
static const float3 WATER_ABSORB_FLOOR = float3(0.015, 0.010, 0.008);
float3 waterExtinction(float3 tint) {
    return WATER_ABSORB_FLOOR + WATER_DENSITY * (float3(1.0, 1.0, 1.0) - clamp(tint, 0.0, 1.0));
}

// A non-water dielectric's tint is a filter over one block of travel rather than a biome water colour, so
// it maps to extinction by inverting Beer-Lambert over that reference distance: transmittance t after one
// block means sigma = -ln(t). `transmission` scales that transmittance, so it keeps the sense the
// material's authored factor has everywhere else — 1 passes the tint through unchanged, 0 is opaque.
// Clamped away from zero so a fully saturated texel stays finite instead of producing an infinite sigma.
static const float VOLUME_TINT_REFERENCE_DISTANCE = 1.0; // blocks
float3 volumeExtinction(float3 tint, float transmission) {
    float3 blockTransmittance = clamp(clamp(tint, 0.0, 1.0) * clamp(transmission, 0.0, 1.0),
            1.0e-3, 1.0);
    return -log(blockTransmittance) / VOLUME_TINT_REFERENCE_DISTANCE;
}

// ---- Participating medium the path is currently inside.
//
// Refraction needs the RELATIVE index across an interface, so the renderer has to know what it is
// travelling through, not just what it is hitting. `ior` drives Snell/Fresnel; `extinction` drives the
// per-segment Beer-Lambert attenuation.
//
// The stack is depth 2 (current + the one it will return to) held in named fields, NOT an array. A
// dynamically indexed local array lands in scratch memory, and this raygen is already register-bound —
// paying an occupancy hit for nesting that Minecraft does not produce would be a bad trade. Depth 2
// covers air->water->glass and air->glass->water, which is the realistic worst case; anything deeper
// degrades to air on the way out, and because `entering` is re-derived per face from geometry rather
// than toggled, the path re-synchronises at the next crossing instead of staying corrupted.
struct Medium {
    float ior;
    float3 extinction;
    bool water; // drives the wave-refraction caustic on submerged receivers; nothing else is water-specific
};

struct MediumStack {
    Medium current;
    Medium outer;
};

Medium airMedium() {
    Medium m;
    m.ior = 1.0;
    m.extinction = float3(0.0, 0.0, 0.0);
    m.water = false;
    return m;
}

MediumStack makeMediumStack(Medium start) {
    MediumStack s;
    s.current = start;
    s.outer = airMedium();
    return s;
}

void mediumPush(inout MediumStack stack, Medium entered) {
    stack.outer = stack.current;
    stack.current = entered;
}

void mediumPop(inout MediumStack stack) {
    stack.current = stack.outer;
    stack.outer = airMedium();
}

// Water's tint is a biome colour whose absorption is calibrated per block of depth; any other volume
// dielectric's tint is a filter over a reference block of travel. Both end up as per-channel extinction.
Medium makeDielectricMedium(bool isWater, float ior, float3 tint, float transmission) {
    Medium m;
    m.ior = ior;
    m.extinction = isWater ? waterExtinction(tint) : volumeExtinction(tint, transmission);
    m.water = isWater;
    return m;
}

// Animated surface waves — a directional wave spectrum, gradient-only (the surface stays geometrically
// flat; only the normal tilts). Three things make it read as water rather than animated glass:
//  * Sharp-crested profile: each component is h = a·exp(S·(sin φ − 1)) — peaked crests, wide flat
//    troughs (pure sines look like rolling jelly). Still C∞ smooth with an exact analytic gradient.
//  * Real deep-water dispersion ω = √(g·k): long swell strides past (λ=14 m → ~4.7 m/s) while short
//    chop flutters nearly in place — wrong relative speeds are the biggest tell of fake water.
//  * Wind + meander: headings spread around a fixed wind direction (tight for the swell, wide for the
//    chop), and the largest components' phase is modulated along their crest (chain-ruled into the
//    gradient) so wavefronts wander instead of ruling straight periodic lines across the sea.
// The domain is world-anchored (see applyWaterWaves) so the pattern is pinned in world space, and the
// caustics (waterCaustic) differentiate this same field, so they stay in sync with the visible ripples.
static const float WATER_WAVE_STRENGTH = 0.3;  // slope scale (higher = choppier glints / more broken reflections)
static const float WAVE_SPEED = 0.8;           // global time scale; dispersion keeps relative speeds physical
static const int   WAVE_COUNT = 10;            // λ = 14 m … 0.38 m (each octave 1.5× shorter)
static const float WAVE_G = 9.81;
static const float2 WAVE_WIND = normalize(float2(1.0, 0.35)); // dominant travel direction
static const float WAVE_MEANDER = 1.1;         // crest phase-wobble amplitude (rad) on the long components
float waterWaveLodWeight(float wavelength, float footprint) {
    float cyclesPerPixel = max(footprint, 0.0) / max(wavelength, 1.0e-4);
    return 1.0 - smoothstep(0.25, 0.5, cyclesPerPixel);
}

// One walk of the wave spectrum, shared by every consumer so the spectrum constants exist once. WITH_DT
// is a generic so the derivative terms are compiled out entirely for callers that only want the
// gradient — the trigonometry is shared, but the chain-rule work is not free.
void waterWaveSpectrum<let WITH_DT : int>(float2 p, float t, float footprint,
                                          out float2 grad, out float2 gradDt) {
    // Per-wave steepness a·k: gentle in the long swell, concentrated in the mid/short chop.
    const float Q[WAVE_COUNT] = { 0.030, 0.035, 0.042, 0.050, 0.060, 0.065, 0.065, 0.058, 0.048, 0.038 };
    t *= WAVE_SPEED;
    grad = float2(0.0, 0.0);
    gradDt = float2(0.0, 0.0);
    float lambda = 14.0;
    for (int i = 0; i < WAVE_COUNT; i++) {
        float lodWeight = waterWaveLodWeight(lambda, footprint);
        if (lodWeight <= 0.0) break;
        float k = 2.0 * PI / lambda;
        float w = sqrt(WAVE_G * k);   // deep-water dispersion: phase speed √(g/k)
        // Heading: deterministic irregular spread about the wind, widening down the spectrum.
        float spread = (0.15 + 0.11 * float(i)) * sin(float(i) * 2.4 + 1.3);
        float cs = cos(spread), sn = sin(spread);
        float2 d = float2(WAVE_WIND.x * cs - WAVE_WIND.y * sn, WAVE_WIND.x * sn + WAVE_WIND.y * cs);
        float ph = k * dot(d, p) - w * t + 1.7 * float(i) * float(i);
        float2 dph = k * d;
        float phDt = -w;                      // ∂φ/∂t
        float2 dphDt = float2(0.0, 0.0);      // ∂(∇φ)/∂t
        if (i < 4) { // meander only the components whose fronts are long enough to read as lines
            float2 pd = float2(-d.y, d.x);
            float wph = 0.4 * k * dot(pd, p) + 0.5 * w * t;
            ph += WAVE_MEANDER * sin(wph);
            dph += (WAVE_MEANDER * cos(wph) * 0.4 * k) * pd;
            if (WITH_DT != 0) {
                float wphDt = 0.5 * w;
                phDt += WAVE_MEANDER * cos(wph) * wphDt;
                dphDt = (-WAVE_MEANDER * sin(wph) * wphDt * 0.4 * k) * pd;
            }
        }
        float S = 1.2 + 0.12 * float(i); // crest sharpness: the short chop is choppier
        float sinPh = sin(ph);
        float cosPh = cos(ph);
        float e = exp(S * (sinPh - 1.0));
        // a = Q/k; ∇h = a·S·cos φ·e^{S(sin φ−1)}·∇φ
        float amplitude = lodWeight * (Q[i] / k) * S;
        grad += amplitude * cosPh * e * dph;
        if (WITH_DT != 0) {
            // ∂/∂t of the term above, chain-ruled through φ and ∇φ.
            gradDt += amplitude * e
                    * (phDt * (-sinPh + S * cosPh * cosPh) * dph + cosPh * dphDt);
        }
        lambda /= 1.5;
    }
    grad *= WATER_WAVE_STRENGTH;
    gradDt *= WATER_WAVE_STRENGTH;
}

float2 waterWaveGrad(float2 p, float t, float footprint) {
    float2 grad, gradDt;
    waterWaveSpectrum<0>(p, t, footprint, grad, gradDt);
    return grad;
}

float2 waterWaveGrad(float2 p, float t) {
    return waterWaveGrad(p, t, 0.0);
}

// Evaluate the current gradient and linearly rewind it to the previous phase in one spectrum walk.
// Frame deltas are bounded on the CPU, so this avoids a second set of trigonometric/exponential work
// while remaining accurate over the sub-frame interval used by reflection reprojection.
void waterWaveGradTemporal(float2 p, float currentT, float previousT, float footprint,
                           out float2 currentGrad, out float2 previousGrad) {
    float2 gradDt;
    waterWaveSpectrum<1>(p, currentT, footprint, currentGrad, gradDt);
    previousGrad = currentGrad - (currentT - previousT) * WAVE_SPEED * gradDt;
}

// Perturb a (near-)horizontal water surface normal by the wave field. Flowing side faces (|n.y| small)
// keep their flat geometric normal. `worldXZ` must be a world-stable coordinate (rebased hit + the pushed
// anchor) so the ripples stay pinned in the world. `nGeo` is already oriented toward the viewer, so the
// down-viewed underside (n.y < 0) takes the mirrored slope (-up).
float3 applyWaterWaves(float3 nGeo, float2 worldXZ, float t, float footprint) {
    if (abs(nGeo.y) < 0.5) return nGeo;
    float2 grad = waterWaveGrad(worldXZ, t, footprint);
    float3 up = normalize(float3(-grad.x, 1.0, -grad.y));
    return nGeo.y >= 0.0 ? up : -up;
}

// ---- Water caustics. Sunlight refracting through the waved surface converges/diverges before it
// reaches an underwater receiver; the analytic wave field makes the true focusing factor computable
// instead of faked with a scrolling texture. Where the shadow ray of an underwater NEE vertex crossed
// water (returned by visibility() as VisibilityResult.waterHitT), evaluate the horizontal landing
// position of the refracted sun ray as a function of surface position and finite-difference it. The
// caustic intensity is the inverse Jacobian determinant of that surface→floor mapping (area compression = brightening, real
// fold caustics where det → 0). Because this uses the SAME wave field as the visible surface normals,
// the caustic pattern stays in sync with the ripples, and the per-sample sun-quad jitter (sampleSquare)
// shifts the pattern per sample → caustics physically blur with depth under DLSS-RR accumulation.
static const float CAUSTIC_EPS = 0.25; // finite-difference step (m); also low-passes sub-step wave detail
static const float CAUSTIC_MAX = 6.0;  // focusing clamp: keeps fold-line singularities from becoming fireflies

// Horizontal landing offset of a refracted sun ray entering the surface at world-stable xz, intersected
// with the receiver plane h metres below. inc = sunlight travel direction (downward, unit).
float2 causticLanding(float2 xz, float t, float3 inc, float h) {
    float2 g = waterWaveGrad(xz, t);
    float3 nw = normalize(float3(-g.x, 1.0, -g.y));
    float3 rdw = refract(inc, nw, 1.0 / WATER_IOR); // air→water: never TIR
    return xz + rdw.xz * (h / max(-rdw.y, 0.05));
}

float waterCaustic(float3 exitPos, float3 lightDir, float waterDist) {
    // Grazing light: the receiver-plane intersection degenerates and caustics vanish in reality anyway
    // (long slant paths absorb + the pattern smears out) — fade the effect to neutral 1.0.
    float fade = smoothstep(0.06, 0.18, lightDir.y);
    if (fade <= 0.0) return 1.0;
    float h = max(waterDist * lightDir.y, 0.05); // vertical depth of the receiver below the exit point
    float t = worldPush.waterParams.w;
    float3 inc = -lightDir;
    float2 base = exitPos.xz + worldPush.waterAnchor.xy;  // world-stable wave domain (same as applyWaterWaves)
    float2 p0 = causticLanding(base, t, inc, h);
    float2 px = causticLanding(base + float2(CAUSTIC_EPS, 0.0), t, inc, h);
    float2 pz = causticLanding(base + float2(0.0, CAUSTIC_EPS), t, inc, h);
    float det = abs((px.x - p0.x) * (pz.y - p0.y) - (px.y - p0.y) * (pz.x - p0.x));
    // Flat water ⇒ all landings displace identically ⇒ det = eps² ⇒ focus = 1 (energy-neutral); the
    // clamp bounds the fold-line spikes whose energy the surrounding darkening already paid for.
    float focus = (CAUSTIC_EPS * CAUSTIC_EPS) / max(det, 1.0e-5);
    return lerp(1.0, min(focus, CAUSTIC_MAX), fade);
}

// Below this GGX alpha the lobe is treated as a delta mirror. The threshold is not about detecting an
// authored zero — it is about the point where a finite lobe stops being distinguishable from a mirror.
// A GGX lobe's half-angle is ~alpha, and the sun's angular radius is ~0.0047 rad, so alpha = 4e-4 is
// ~12x tighter than the sun disc and far under a pixel footprint at any reflection distance. Nothing
// below it can be resolved, and the delta path is both sharper and better conditioned there.
//
// Two things follow from taking the delta path, and both are why the threshold is set here rather than
// at the 8-bit quantum:
//   * ggxD(_, 0) is exactly 0, so the NEE specular term vanishes by construction and the sun arrives
//     solely via the mirror ray hitting the disc (showCelestial). That is the correct, unbiased
//     accounting — see the double-count note on ggxD below.
//   * There is no floor to apply above the threshold: alpha > 4e-4 is well conditioned for both the
//     VNDF sample and the NDF, so authored roughness passes through unmodified and the mapping stays
//     continuous. A separate MIN_ROUGH would only reintroduce a cliff.
//
// Stored roughness IS alpha (see payloadRoughness), so these compare against it directly.
static const float MIRROR_ALPHA_MAX = 4.0e-4; // LabPBR perceptual smoothness >= 0.98
// Above this the specular lobe is too broad for a single mirrored reprojection to describe, so the
// reflection MV is left at zero and RR falls back to the ordinary MV.
static const float SPEC_MOTION_ALPHA_MAX = 0.25; // perceptual smoothness <= 0.5
static const uint MAX_TRANSMISSION_GUIDE_HITS = 8u;
static const uint PRIMARY_DIELECTRIC_RANDOM = 0u;
static const uint PRIMARY_DIELECTRIC_TRANSMISSION = 1u;
static const uint PRIMARY_DIELECTRIC_REFLECTION = 2u;

float luminance(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

// GGX normal distribution.
// NOTE: the 1e-7 denominator guard is not merely a divide-by-zero epsilon, and it is load-bearing. At
// the lobe peak d == a2, so the guard dominates PI*d*d until a2 >> 1.8e-4, i.e. rough >> 0.116; every
// smoother material has its NEE highlight peak truncated. That truncation is what keeps the sun from
// being counted twice: a specular continuation sets showCelestial, so the BSDF-sampled ray may hit the
// sun disc in world.rmiss while NEE is also adding a sun term for the same lobe, with no MIS weighting
// between them. Removing the guard restores the physical peak AND the double count, and makes the
// single-sample sun estimator explode wherever the lobe is tighter than the disc (~0.0047 rad). The
// real fix is MIS (or a disc-aware specular light sample), not an epsilon edit — left as-is.
float ggxD(float ndh, float alpha) {
    float a2 = alpha * alpha;
    float d = ndh * ndh * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d + 1.0e-7);
}

// Smith GGX masking term for one direction (separable form; G2 = G1(v) * G1(l)).
float ggxG1(float ndx, float alpha) {
    float a2 = alpha * alpha;
    return 2.0 * ndx / (ndx + sqrt(a2 + (1.0 - a2) * ndx * ndx) + 1.0e-7);
}

float3 fresnelSchlick(float cosT, float3 f0) {
    float m = clamp(1.0 - cosT, 0.0, 1.0);
    float m2 = m * m;
    return f0 + (1.0 - f0) * (m2 * m2 * m);
}

// Henyey–Greenstein phase function (with 4π normalisation so it integrates to 1 over the sphere).
// g=0 → isotropic; g>0 → forward-scatter peak at cosT=1 (light and view aligned through a thin slab).
static const float SSS_G        = 0.6; // forward-scatter anisotropy; leaves/grass are quite directional
static const float SSS_STRENGTH = 1.0; // 1.0 ≈ 2.5× the Lambertian at peak (sss=1, cosT=1, backNdl=1)
// Deepest SHADED surface at which the SSS transmission shadow ray fires. diffuseDepth, not `bounce`: the
// glass/water continue branches don't shade a surface, so foliage seen through a window is still
// diffuseDepth 0 and should backlight like any other primary leaf — see RIS_MAX_DIFFUSE_DEPTH for the
// same distinction and why it matters.
static const int   MAX_SSS_DIFFUSE_DEPTH = 1; // fire on the first-shaded surface + one indirect hit
float hg(float cosT, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(max(0.0, 1.0 + g2 - 2.0 * g * cosT), 1.5));
}

// NVIDIA RR guide-buffer helper: integrated/view-dependent specular reflectivity, not raw F0.
// alpha is GGX alpha (== the linear roughness materials store), NoV is the first-hit view cosine.
float3 rrSpecularAlbedo(float3 specularColor, float alpha, float NoV) {
    NoV = abs(NoV);
    float NoV2 = NoV * NoV;
    float alpha2 = alpha * alpha;
    float4 X = float4(1.0, NoV, NoV2, NoV * NoV2);
    float4 Y = float4(1.0, alpha, alpha2, alpha * alpha2);

    float2 m1 = float2(
            dot(float2(0.99044, -1.28514), X.xy),
            dot(float2(1.29678, -0.755907), X.xy));
    float3 Xxyw = float3(X.x, X.y, X.w);
    float3 m2 = float3(
            dot(float3(1.0, 2.92338, 59.4188), Xxyw),
            dot(float3(20.3225, -27.0302, 222.592), Xxyw),
            dot(float3(121.563, 626.13, 316.627), Xxyw));
    float bias = dot(m1, Y.xy) / max(dot(m2, float3(Y.x, Y.y, Y.w)), 1.0e-7);

    float2 m3 = float2(
            dot(float2(0.0365463, 3.32707), X.xy),
            dot(float2(9.0632, -9.04756), X.xy));
    float3 Xxzw = float3(X.x, X.z, X.w);
    float3 m4 = float3(
            dot(float3(1.0, 3.59685, -1.36772), Xxzw),
            dot(float3(9.04401, -16.3174, 9.22949), Xxzw),
            dot(float3(5.56589, 19.7886, -20.2123), Xxzw));
    float scale = dot(m3, Y.xy) / max(dot(m4, float3(Y.x, Y.y, Y.w)), 1.0e-7);

    bias *= clamp(specularColor.g * 50.0, 0.0, 1.0);
    return specularColor * max(0.0, scale) + float3(max(0.0, bias), max(0.0, bias), max(0.0, bias));
}

// Exact (unpolarized) Fresnel reflectance for a dielectric interface. cosI is the incidence cosine on
// the incoming side; etaI/etaT are the IORs the ray travels from / into. Returns 1.0 on total internal
// reflection (sin of the transmitted angle >= 1), which matches refract() returning the zero vector —
// so reflect/refract selection and TIR stay consistent.
float fresnelDielectric(float cosI, float etaI, float etaT) {
    float sinT2 = (etaI * etaI) / (etaT * etaT) * max(0.0, 1.0 - cosI * cosI);
    if (sinT2 >= 1.0) {
        return 1.0;
    }
    float cosT = sqrt(1.0 - sinT2);
    float rs = (etaI * cosI - etaT * cosT) / (etaI * cosI + etaT * cosT);
    float rp = (etaI * cosT - etaT * cosI) / (etaI * cosT + etaT * cosI);
    return 0.5 * (rs * rs + rp * rp);
}

// PCG hash RNG.
uint pcg(inout uint s) {
    s = s * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
float rndf(inout uint s) {
    // Convert the high 24 bits, which are exactly representable as float. Converting all 32 bits first
    // lets the top 128 uint values round to 2^32, incorrectly returning 1.0 and breaking `< probability`
    // tests (most seriously the F == 1 total-internal-reflection branch).
    return float(pcg(s) >> 8u) * (1.0 / 16777216.0);
}

float3 primaryRayDir(float2 ndc) {
    float4 nearH = mul(worldPush.invViewProj, float4(ndc.x, ndc.y, 1.0, 1.0));
    float4 farH = mul(worldPush.invViewProj, float4(ndc.x, ndc.y, 0.0, 1.0));
    float3 nearP = nearH.xyz / nearH.w;
    float3 farP = farH.xyz / farH.w;
    return normalize(farP - nearP);
}

float primaryRayConeSpread(float2 ndc, float2 size, float3 dir) {
    float2 onePixelNdc = 2.0 / size;
    float3 dx = primaryRayDir(ndc + float2(onePixelNdc.x, 0.0));
    float3 dy = primaryRayDir(ndc + float2(0.0, onePixelNdc.y));
    return max(max(length(cross(dir, dx)), length(cross(dir, dy))), RAY_CONE_MIN_SPREAD);
}

// Cosine-weighted hemisphere sample about n (Malley's method). For a Lambertian BRDF this is the
// importance-sampling match: BRDF*cos/pdf = (albedo/PI)*cos / (cos/PI) = albedo, so the continuation
// throughput is just *= albedo with no PI/cos bookkeeping left over.
float3 cosineDir(float3 n, inout uint s) {
    float u1 = rndf(s);
    float u2 = rndf(s);
    float r = sqrt(u1);
    float phi = 6.2831853 * u2;
    float3 t = normalize(abs(n.x) > 0.9 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0));
    float3 b = normalize(cross(n, t));
    t = cross(b, n);
    float3 local = float3(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u1)));
    return normalize(local.x * t + local.y * b + local.z * n);
}

// Soft shadows: sample a direction within the light's SQUARE angular extent about `axis`. MC's
// sun/moon are square quads, so the NEE shadow ray samples a square (not a cone) — the same square the
// visible disc in world.rmiss uses, built from the same celestial tangent frame (right = arc-travel
// direction, from worldPush.celestial.xyz). Averaged over frames by DLSS-RR this yields soft penumbrae that
// widen with occluder distance (contact-hardening) for free. halfAngle <= 0 ⇒ exact direction (hard).
float3 sampleSquare(float3 axis, float halfAngle, inout uint s) {
    float3 right = normalize(cross(axis, worldPush.celestial.xyz));
    float3 up = cross(right, axis);
    float t = tan(halfAngle);
    float u = (rndf(s) * 2.0 - 1.0) * t;
    float v = (rndf(s) * 2.0 - 1.0) * t;
    return normalize(axis + u * right + v * up);
}

// Sample a GGX visible-normal (VNDF, Heitz 2018) about geometric normal n for view ve, returning the
// sampled microfacet normal (half-vector) in world space. Paired with the separable Smith term, the
// importance-sampling weight reduces to F * G1(NdotL) (applied by the caller).
float3 sampleGGXVNDF(float3 n, float3 ve, float alpha, inout uint s) {
    float3 t = normalize(abs(n.x) > 0.9 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0));
    float3 b = normalize(cross(n, t));
    t = cross(b, n);
    float3 Ve = float3(dot(ve, t), dot(ve, b), dot(ve, n)); // view in tangent space (z = n)
    float a = alpha;
    float3 Vh = normalize(float3(a * Ve.x, a * Ve.y, Ve.z));
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    float3 T1 = lensq > 0.0 ? float3(-Vh.y, Vh.x, 0.0) * rsqrt(lensq) : float3(1.0, 0.0, 0.0);
    float3 T2 = cross(Vh, T1);
    float u1 = rndf(s);
    float u2 = rndf(s);
    float r = sqrt(u1);
    float phi = 6.2831853 * u2;
    float p1 = r * cos(phi);
    float p2 = r * sin(phi);
    float ss = 0.5 * (1.0 + Vh.z);
    p2 = (1.0 - ss) * sqrt(1.0 - p1 * p1) + ss * p2;
    float3 Nh = p1 * T1 + p2 * T2 + sqrt(max(0.0, 1.0 - p1 * p1 - p2 * p2)) * Vh;
    float3 Ne = normalize(float3(a * Nh.x, a * Nh.y, max(0.0, Nh.z))); // microfacet normal, tangent space
    return normalize(Ne.x * t + Ne.y * b + Ne.z * n);                 // -> world space
}

// ===== RIS emitter NEE =================================================================
// Block emitters (torches, glowstone, lava, ...) get explicit NEE: per diffuse vertex, stream M uniform
// candidates from the global light buffer through a weighted reservoir keyed by the UNSHADOWED target
// p-hat = luminance(f * Le * G * area), then spend ONE shadow ray on the survivor. Sun/moon NEE stays
// separate and additive. Lights are footprint bounding RECTANGLES (RtLightCollector): sampling a point
// uniformly over the rectangle has pdf 1/area, so `area` rides the contribution like any area light.
// Single-frame RIS only (no temporal reservoir reuse): DLSS-RR's own history already denoises this well
// enough that a persistent per-pixel reservoir wasn't worth the added complexity and VRAM.
// One resample serves three mutually exclusive receiver terms (see evalSampleContrib): front BRDF,
// two-sided particle billboards, and LabPBR SSS backscatter from lights BEHIND the surface — all share
// the reservoir and its single shadow ray, whose origin follows the survivor's side of the surface.

// A reservoir in registers. pos/lnrm/le/area describe the chosen emitter sample; W is its unbiased
// contribution weight, M the effective sample count. wSum/phat are streaming scratch (not persisted).
struct Reservoir {
    float3 pos;  // light sample point (rebased world space)
    float3 lnrm; // emitter outward normal
    float3 le;   // emitter radiance
    float area;  // emitter rectangle area (the area-light pdf factor)
    float M;
    float W;
    float wSum;
    float phat;
};

Reservoir resEmpty() {
    Reservoir r;
    r.pos = float3(0.0, 0.0, 0.0);
    r.lnrm = float3(0.0, 0.0, 0.0);
    r.le = float3(0.0, 0.0, 0.0);
    r.area = 0.0;
    r.M = 0.0;
    r.W = 0.0;
    r.wSum = 0.0;
    r.phat = 0.0;
    return r;
}

// Unshadowed contribution of a fixed light sample point at surface (hitPos,n,v); out p-hat = its
// luminance (the resampling target). Evaluated from the UNBIASED hitPos — SURF_BIAS is applied only to
// the shadow-ray origin in shadeReservoir, on whichever side the survivor sample lands. Three receiver
// modes share one target function because they are mutually exclusive per candidate:
//   - front hemisphere (ndl>0): the same Lambert(+GGX) BRDF split the sun NEE uses;
//   - twoSided (particles): billboards have no back face, ndl=abs(ndl) folds both sides into the front;
//   - back hemisphere with sss>0 (LabPBR leaves/grass): HG transmission phase — cosT = dot(wi,rd) peaks
//     when the ray points from the light through the slab toward the camera (same as the sun SSS term).
float3 evalSampleContrib(float3 sp, float3 lnrm, float3 le, float area, float3 hitPos, float3 n,
                         float3 v, float3 rd, float3 diffAlb, float3 F0, float rough,
                         bool twoSided, float sss, out float phat) {
    phat = 0.0;
    float3 toL = sp - hitPos;
    float dist2 = dot(toL, toL);
    if (dist2 < 1.0e-6) {
        return float3(0.0, 0.0, 0.0);
    }
    float dist = sqrt(dist2);
    float3 wi = toL / dist;
    float cosL = max(0.0, dot(lnrm, -wi));
    if (cosL <= 0.0) {
        return float3(0.0, 0.0, 0.0); // receiver behind the emitter face
    }
    float ndl = dot(n, wi);
    if (twoSided) {
        ndl = abs(ndl);
    }
    float3 contrib = float3(0.0, 0.0, 0.0);
    if (ndl > 0.0) {
        float G = ndl * cosL / dist2; // area-light geometry term (x area)
        float3 brdf = diffAlb * INV_PI;
        // twoSided callers are particle billboards, which pass rough=1/F0=0 and want diffuse-only
        // shading: fresnelSchlick still returns up to full white at grazing angles even with F0=0 (that
        // IS the Fresnel effect), so skipping the term here — not just zeroing F0 — is what keeps
        // billboards from picking up a rim highlight they were never meant to have.
        if (!twoSided) {
            float3 h = normalize(wi + v);
            float ndh = max(0.0, dot(n, h));
            float ndv = max(1.0e-4, dot(n, v));
            float vdh = max(0.0, dot(v, h));
            float D = ggxD(ndh, rough);
            float Gs = ggxG1(ndv, rough) * ggxG1(ndl, rough);
            float3 Fs = fresnelSchlick(vdh, F0);
            brdf += (D * Gs) * Fs / (4.0 * ndv * ndl);
        }
        contrib = brdf * le * (G * area);
    } else if (sss > 0.0) {
        float backNdl = -ndl;
        float G = backNdl * cosL / dist2;
        float cosT = dot(wi, rd);
        contrib = diffAlb * sss * SSS_STRENGTH * hg(cosT, SSS_G) * le * (G * area);
    }
    phat = luminance(contrib);
    return contrib;
}

// Select one light from the emitted-power distribution in O(1).
void selectGlobalLight(inout uint proposalSeed, out uint lightIndex) {
    float aliasSample = rndf(proposalSeed) * float(worldPush.lightCount);
    uint column = min(uint(aliasSample), worldPush.lightCount - 1u);
    if (pc.lightAliasAddr != 0) {
        LightAlias a = ConstPtr<LightAlias>(pc.lightAliasAddr)[column];
        bool self = aliasSample - float(column) < a.accept;
        lightIndex = self ? column : a.aliasIndex;
    } else {
        lightIndex = column;
    }
}

bool findLightGridCell(float3 p, out LightGridCell cell, out int3 cellCoord) {
    cell.spanOffset = 0u;
    cell.spanCount = 0u;
    cell.invWeightSum = 0.0;
    cellCoord = int3(0, 0, 0);
    if (pc.lightGridCellAddr == 0 || pc.lightGridSpanAddr == 0) return false;
    int3 coord = int3(floor((p - worldPush.lightGridOrigin.xyz) / worldPush.lightGridOrigin.w));
    if (coord.x < 0 || coord.y < 0 || coord.z < 0
            || coord.x >= worldPush.lightGridDims.x
            || coord.y >= worldPush.lightGridDims.y
            || coord.z >= worldPush.lightGridDims.z) return false;
    uint linear = (uint(coord.z) * uint(worldPush.lightGridDims.y) + uint(coord.y))
            * uint(worldPush.lightGridDims.x) + uint(coord.x);
    cell = ConstPtr<LightGridCell>(pc.lightGridCellAddr)[linear];
    cellCoord = coord;
    return cell.spanCount > 0u;
}

void selectSectionLight(uint firstLight, uint lightCount, inout uint proposalSeed,
                        out uint lightIndex) {
    float aliasSample = rndf(proposalSeed) * float(lightCount);
    uint column = min(uint(aliasSample), lightCount - 1u);
    LightAlias alias = ConstPtr<LightAlias>(pc.lightLocalAliasAddr)[firstLight + column];
    bool self = aliasSample - float(column) < alias.accept;
    uint localIndex = self ? column : alias.aliasIndex;
    lightIndex = firstLight + localIndex;
}

float unpackUnsignedFloat(uint bits, uint mantissaBits) {
    uint mantissaMask = (1u << mantissaBits) - 1u;
    uint mantissa = bits & mantissaMask;
    uint exponent = (bits >> mantissaBits) & 31u;
    if (exponent == 0u) {
        return float(mantissa) * exp2(float(1 - 15 - int(mantissaBits)));
    }
    return (1.0 + float(mantissa) / float(1u << mantissaBits))
            * exp2(float(int(exponent) - 15));
}

float3 lightRadiance(Light light) {
    uint packed = light.le;
    return float3(unpackUnsignedFloat(packed & 0x7ffu, 6u),
            unpackUnsignedFloat((packed >> 11u) & 0x7ffu, 6u),
            unpackUnsignedFloat((packed >> 22u) & 0x3ffu, 5u));
}

int3 lightSectionCoord(Light light) {
    uint packed = light.section;
    return int3(int(packed & 0x3ffu), int((packed >> 10u) & 0x3ffu),
            int((packed >> 20u) & 0x3ffu));
}

// The half axes share three lanes two-at-a-time; see the Light record in world_common.
float3 lightHalfU(Light light) {
    return float3(unpackHalf2(light.halfUxy), unpackHalf2(light.halfUzVx).x);
}

float3 lightHalfV(Light light) {
    return float3(unpackHalf2(light.halfUzVx).y, unpackHalf2(light.halfVyz));
}

// U x V drives both the emitter normal and the rectangle area, so it is computed once and shared.
float3 lightCrossUV(Light light) {
    return cross(lightHalfU(light), lightHalfV(light));
}

// The rect spans 2U x 2V, so its area is 4|U x V| — exactly the rectArea the collector used to store.
float lightArea(Light light) {
    return 4.0 * length(lightCrossUV(light));
}

float3 lightGeometricNormal(Light light) {
    float3 normal = normalize(lightCrossUV(light));
    return (light.section & 0x40000000u) != 0u ? -normal : normal;
}

float proposalPdf(Light light, float3 le, LightGridCell cell, int3 cellCoord,
                  float localProbability) {
    float power = max(0.0, lightArea(light) * luminance(le));
    float globalPdf = pc.lightAliasAddr != 0
            ? power * worldPush.lightRebase.w : 1.0 / float(worldPush.lightCount);
    float localPdf = 0.0;
    if (localProbability > 0.0 && power > 0.0) {
        int3 delta = lightSectionCoord(light) - cellCoord;
        if (all(abs(delta) <= int3(2, 2, 2))) {
            float3 deltaF = float3(delta);
            localPdf = power * cell.invWeightSum / max(1.0, dot(deltaF, deltaF));
        }
    }
    return localProbability * localPdf + (1.0 - localProbability) * globalPdf;
}

void selectLightGridSpanLight(LightGridCell cell, inout uint proposalSeed,
                              out uint lightIndex) {
    ConstPtr<LightGridSpan> spans = ConstPtr<LightGridSpan>(pc.lightGridSpanAddr);
    float aliasSample = rndf(proposalSeed) * float(cell.spanCount);
    uint column = min(uint(aliasSample), cell.spanCount - 1u);
    LightGridSpan span = spans[cell.spanOffset + column];
    bool self = aliasSample - float(column) < span.accept;
    uint firstLight = self ? span.firstLight : span.aliasFirstLight;
    uint lightCount = self ? (span.packedLightCounts & 0xffffu)
            : (span.packedLightCounts >> 16u);
    selectSectionLight(firstLight, lightCount, proposalSeed, lightIndex);
}

// Sample the exact mixture q = alpha*qCell + (1-alpha)*qGlobal. qCell first selects a nearby section
// by emitted power and section distance, then its section-local power alias. Every light in
// the neighborhood is represented without a fixed cap; qGlobal gives distant lights full support.
// Alias spans make weighted local selection O(1). The mixture PDF is reconstructed after the single
// Light48 load: section power cancels with the section-local light PDF.
void selectLightGridLight(LightGridCell cell, bool useLocal, inout uint proposalSeed,
                          out uint lightIndex) {
    if (useLocal) {
        selectLightGridSpanLight(cell, proposalSeed, lightIndex);
    } else {
        selectGlobalLight(proposalSeed, lightIndex);
    }
}

// ---- RIS cost budget. Both constants below come from a set of temporary probes that pinned the
// candidate walk out of the shader to bound what memory-side RIS work could ever be worth. Against an
// 18.0ms frame at M=8, removing every dependent load (span -> alias -> record) and all lane divergence
// saved 5.9ms — a third of the frame. Split by vertex that was ~4.3ms at secondary vertices against
// ~1.1ms at the primary hit, and forcing coherent selection recovered at most 1.3ms of it. So the cost
// was chasing DEPTH, not divergence, and it lived at secondary vertices. The two knobs below spend that
// finding; a presampled candidate pool would attack the same 5.9ms structurally, but see the note on
// proposalSeed in tracePath for why it should share the pool rather than the seed.

// Candidate-count divisor for RIS at secondary vertices (bounce > 0). Secondary vertices carried ~78% of
// the cost while being much the more forgiving place to spend variance: that radiance is already
// integrated over a diffuse lobe before the denoiser sees it. 18.0 -> 16.1ms, and indistinguishable from
// 1 by eye — whereas reducing M globally to 2 was clearly worse, because that also degraded the primary
// hit. The primary hit always keeps the full worldPush.risCandidates.
//
// 4 is the floor worth using rather than a tuning accident. The stratification below keeps at least one
// global candidate, so M=2 is the smallest count that still schedules one local AND one global; at M=1
// localProbability collapses to 0 and the light-grid proposal that makes nearby emitters sample well is
// gone entirely.
static const uint SECONDARY_RIS_DIVISOR = 4u;

// Deepest SHADED surface at which RIS emitter NEE runs. Past it, emitters are gathered only when a path
// hits one directly — the pre-RIS behaviour, still unbiased because gateEmitter lifts in lockstep, so the
// failure mode is noisier emitter light at depth rather than missing light. Unlike the divisor this also
// drops shadeReservoir's per-vertex shadow ray, which never scaled with M. Gating at all: 16.1 -> 15.7ms.
//
// This counts diffuseDepth, not `bounce`, and the difference is not cosmetic. MATERIAL_DIELECTRIC and
// MATERIAL_WATER continue without shading anything, so looking through a pane spends bounce 0 on the
// glass and lands the first shaded surface at bounce 1. Gating on `bounce` therefore starved surfaces
// seen through glass or water — visually primary, but counted as depth — and they went black, since only
// a path randomly striking an emitter could light them. Keying on shaded surfaces removes that whole
// class of bug and lets the budget go lower than the two bounces of headroom the dielectric prefix used
// to cost.
//
// Caveat this does not solve: a smooth specular bounce IS a shaded surface, so a wall seen in a metal
// block counts as depth 1. Accumulated roughness rather than a vertex count is the principled fix, and
// is what a radiance cache would need anyway to decide cache-vs-trace.
static const uint RIS_MAX_DIFFUSE_DEPTH = 2u;

// Initial RIS over M power-weighted candidates -> one resampled sample, no shadow ray yet. The chosen
// sample's W = wSum / (M * p-hat); M counts every candidate, including zero-weight ones.
Reservoir risInitial(float3 hitPos, float3 n, float3 v, float3 rd, float3 diffAlb, float3 F0,
                     float rough, bool twoSided, float sss, uint shadedDepth,
                     inout uint seed, inout uint proposalSeed) {
    Reservoir r = resEmpty();
    LightGridCell gridCell;
    int3 gridCellCoord;
    float3 gridLookup = hitPos;
    if (pc.lightGridCellAddr != 0) {
        // Stochastically blend across hard section boundaries. Every cell proposal retains full global
        // support, so conditioning on this independently jittered lookup remains unbiased.
        gridLookup += (float3(rndf(proposalSeed), rndf(proposalSeed), rndf(proposalSeed)) - 0.5)
                * worldPush.lightGridOrigin.w;
    }
    bool hasGridCell = findLightGridCell(gridLookup, gridCell, gridCellCoord);
    // shadedDepth, not the bounce index: a surface behind glass or water is the first thing this path has
    // shaded and is visually primary, so it keeps the full candidate count.
    uint candidateCount = shadedDepth == 0u
            ? worldPush.risCandidates
            : max(1u, worldPush.risCandidates / SECONDARY_RIS_DIVISOR);
    // Deterministically stratify the proposal mixture. At the default M=8 this schedules exactly six
    // local and two global candidates, with every lane taking the same branch for a given candidate.
    // Counts that are not divisible by four use the nearest practical split with at least one global
    // candidate; M=1 is global-only so the estimator retains full support.
    uint globalCandidateCount = hasGridCell
            ? max(1u, (candidateCount + 2u) / 4u) : candidateCount;
    uint localCandidateCount = candidateCount - globalCandidateCount;
    float localProbability = float(localCandidateCount) / float(candidateCount);
    for (uint c = 0u; c < candidateCount; c++) {
        r.M += 1.0;
        uint li;
        uint globalsBefore = (c * globalCandidateCount) / candidateCount;
        uint globalsAfter = ((c + 1u) * globalCandidateCount) / candidateCount;
        bool useLocal = hasGridCell && globalsAfter == globalsBefore;
        selectLightGridLight(gridCell, useLocal, proposalSeed, li);
        Light lg = ConstPtr<Light>(pc.lightBufAddr)[li];
        // Soft shadows: uniform point on the emitter rectangle ((s,t) in [-1,1]^2, pdf 1/area).
        float s = rndf(seed) * 2.0 - 1.0;
        float t = rndf(seed) * 2.0 - 1.0;
        float3 sp = lg.pos + worldPush.lightRebase.xyz
                + s * lightHalfU(lg) + t * lightHalfV(lg);
        float3 le = lightRadiance(lg);
        float3 lightNormal = lightGeometricNormal(lg);
        float area = lightArea(lg);
        float phat;
        evalSampleContrib(sp, lightNormal, le, area, hitPos, n, v, rd,
                diffAlb, F0, rough, twoSided, sss, phat);
        if (phat <= 0.0) {
            continue;
        }
        float sourcePdf = proposalPdf(lg, le, gridCell, gridCellCoord, localProbability);
        float w = phat / max(sourcePdf, 1.0e-20);
        r.wSum += w;
        if (rndf(seed) * r.wSum < w) { // weighted reservoir update
            r.pos = sp;
            r.lnrm = lightNormal;
            r.le = le;
            r.area = area;
            r.phat = phat;
        }
    }
    r.W = r.phat > 0.0 ? (r.wSum / (r.M * r.phat)) : 0.0;
    return r;
}

// Shade a finalized reservoir: recompute the survivor's contribution at (hitPos,n,v), cast ONE shadow
// ray, and return throughput-free radiance contrib*vis*W. The one ray serves whichever term fired for
// the survivor (front BRDF, twoSided billboard, or SSS backscatter) — the origin is biased toward the
// sample's side of the surface.
float3 shadeReservoir(Reservoir s, float3 hitPos, float3 n, float3 v, float3 rd, float3 diffAlb,
                      float3 F0, float rough, bool twoSided, float sss, out float3 vis) {
    vis = float3(0.0, 0.0, 0.0);
    if (s.W <= 0.0 || s.phat <= 0.0) {
        return float3(0.0, 0.0, 0.0);
    }
    float phat;
    float3 contrib = evalSampleContrib(s.pos, s.lnrm, s.le, s.area, hitPos, n, v, rd,
            diffAlb, F0, rough, twoSided, sss, phat);
    if (phat <= 0.0) {
        return float3(0.0, 0.0, 0.0);
    }
    // Pick the shadow-ray origin on the sample's side of the surface FIRST, then measure the ray from
    // that origin — measuring dist from hitPos while tracing from the biased origin shifts the ray tip
    // up to SURF_BIAS*ndl toward the emitter plane, beating the 0.001*dist stop-short margin for lights
    // closer than ~SURF_BIAS/0.001 blocks -> shadow ray hits the emitter's own face -> per-texel black.
    float side = dot(n, s.pos - hitPos) >= 0.0 ? 1.0 : -1.0;
    float3 origin = hitPos + side * n * SURF_BIAS;
    float3 toL = s.pos - origin;
    float dist = length(toL);
    // Stop just short of the sample point so the ray doesn't self-occlude on the emitter's own face.
    VisibilityResult shadow = visibility(origin, toL / dist, dist * 0.999);
    vis = shadow.transmittance;
    return contrib * vis * s.W;
}

// TLAS instance-mask bits (set on the Java side, ANDed against these per-ray cull masks):
//   bit 0 (0x01) — secondary rays: shadows, GI bounces, reflections.
//   bit 1 (0x02) — the primary camera ray.
// Terrain / entities / block entities keep 0xFF (both bits). Particles use 0x02: primary-visible
// receiver billboards only. The first-person camera owner uses 0x01 (secondary only).
static const uint CULL_SECONDARY = 0x01u;
static const uint CULL_PRIMARY = 0x02u;
static const uint TERRAIN_BUCKETS = 4u;
static const uint SBT_RADIANCE = 0u;
static const uint SBT_SHADOW = TERRAIN_BUCKETS;
static const uint SBT_STRIDE_BUCKET = 1u;
static const uint MISS_RADIANCE = 0u;
static const uint MISS_GUIDE = 1u;

RayDesc makeRay(float3 origin, float tmin, float3 dir, float tmax) {
    RayDesc r;
    r.Origin = origin;
    r.TMin = tmin;
    r.Direction = dir;
    r.TMax = tmax;
    return r;
}

float3 offsetSurfaceOrigin(float3 position, float3 surfaceNormal,
                           float3 outgoingDirection, float bias) {
    float3 sideNormal = dot(surfaceNormal, outgoingDirection) >= 0.0
            ? surfaceNormal : -surfaceNormal;
    return position + sideNormal * bias;
}

// A radiance payload carrying only the two words the invoked hit/miss shader READS: `flags`
// (show-celestial gate, consumed by world.rmiss) and the packed ray cone (consumed by world.rchit for
// texture LOD). Every other member is an OUTPUT that world.rchit/world.rmiss write, so it carries no
// information into a trace and is initialized to a neutral value here.
Payload makeRadiancePayload(uint flags, uint rayCone) {
    Payload p;
    p.albedo = half3(0.0h, 0.0h, 0.0h);
    p.normal = half3(0.0h, 0.0h, 0.0h);
    p.hitT = -1.0; // miss sentinel: world.rmiss writes only albedo, so a miss leaves this negative
    p.motionPrev = half3(0.0h, 0.0h, 0.0h);
    p.f0 = half3(0.0h, 0.0h, 0.0h);
    p.flags = flags;
    p.roughMetal = 0u;
    p.emissionSss = 0u;
    p.iorTransmission = 0u;
    p.rayCone = rayCone;
    return p;
}

// SER radiance trace: trace into a hit object, reorder threads by hit coherence, THEN invoke the
// hit/miss shader — this is where the divergent Section/Prim shading work (world.rchit/world.rahit)
// actually happens, so reordering before it is what pays for the reorder's own cost.
//
// The payload is built TWICE from the trace state rather than carried across ReorderThread. Traversal
// runs only world.rahit, which never writes a RADIANCE payload: its terrain tint paths live in the
// translucent/water buckets, which carry no any-hit record for radiance rays, and its entity paths are
// gated on RAY_FLAG_SKIP_CLOSEST_HIT_SHADER. So whatever TraceRay leaves in the payload holds no
// information. Reading it back and carrying it over would pin all ten members — 72 bytes — in registers
// across the one point where SER must spill every live value, which profiling showed to be the peak
// live state in this shader. Rebuilding costs a few constant moves and leaves only these two words
// spanning the reorder.
void traceRadianceReordered(uint cullMask, float3 ro, float tmin, float3 rd, float tmax,
                            bool showCelestial, float rayConeWidth, float rayConeSpread) {
    uint flags = showCelestial ? PAYLOAD_SHOW_CELESTIAL : 0u;
    uint rayCone = packHalf2(float2(rayConeWidth, rayConeSpread));
    Payload tracePayload = makeRadiancePayload(flags, rayCone);
    HitObject hObj = HitObject::TraceRay(topLevelAS, RAY_FLAG_NONE, cullMask,
            SBT_RADIANCE, SBT_STRIDE_BUCKET, MISS_RADIANCE,
            makeRay(ro, tmin, rd, tmax), tracePayload);
    ReorderThread(hObj);
    payload = makeRadiancePayload(flags, rayCone);
    HitObject::Invoke(topLevelAS, hObj, payload);
}

// Auxiliary RR probes deliberately use ordinary TraceRay. They are one-off, incoherent rays whose
// result is consumed immediately, so paying a reorder barrier and carrying guide state across it only
// adds latency. The dedicated miss record also avoids the atmosphere march: guides need hit-vs-miss,
// not sky radiance.
void traceGuide(uint cullMask, float3 ro, float tmin, float3 rd, float tmax,
                float rayConeWidth, float rayConeSpread) {
    payload = makeRadiancePayload(0u, packHalf2(float2(rayConeWidth, rayConeSpread)));
    TraceRay(topLevelAS, RAY_FLAG_NONE, cullMask, SBT_RADIANCE, SBT_STRIDE_BUCKET,
            MISS_GUIDE, makeRay(ro, tmin, rd, tmax), payload);
}

Payload makeShadowPayload() {
    Payload shadowPayload;
    shadowPayload.albedo = half3(1.0h, 1.0h, 1.0h);
    shadowPayload.hitT = -1.0; // water-crossing sentinel, filled by the water any-hit
    shadowPayload.normal = half3(0.0h, 0.0h, 0.0h);
    shadowPayload.motionPrev = half3(0.0h, 0.0h, 0.0h);
    shadowPayload.f0 = half3(0.0h, 0.0h, 0.0h);
    shadowPayload.flags = 0u;
    shadowPayload.roughMetal = 0u;
    shadowPayload.emissionSss = 0u;
    shadowPayload.iorTransmission = 0u;
    shadowPayload.rayCone = 0u;
    return shadowPayload;
}

VisibilityResult visibility(float3 origin, float3 dir, float tmax) {
    // Vulkan requires an identical payload structure for every stage reachable by this trace. The shadow
    // path uses only albedo as accumulated transmittance and hitT as the nearest-water crossing.
    Payload shadowPayload = makeShadowPayload();
    // Shadow SBT records run any-hit only for cutout/translucent/water. Cutout alpha-tests; translucent
    // and water tint shadowPayload.albedo and pass through. Solid blocks terminate traversal. There is no
    // closest or miss shader worth executing, so use a hit object only as the traversal result.
    HitObject hObj = HitObject::TraceRay(topLevelAS,
            RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
            CULL_SECONDARY, SBT_SHADOW, SBT_STRIDE_BUCKET, 0u,
            makeRay(origin, RAY_TMIN, dir, tmax), shadowPayload);
    VisibilityResult result;
    result.transmittance = hObj.IsMiss() ? shadowPayload.albedo : float3(0.0, 0.0, 0.0);
    result.waterHitT = shadowPayload.hitT;
    return result;
}

// `valid` is false when the point projects behind the previous camera, where the perspective divide
// would otherwise manufacture an arbitrarily large NDC. Mirror images reach that case easily: a virtual
// image sits as far behind the reflector as the source is in front, so a reflection seen at a grazing
// angle can land behind the eye even though the reflector itself is comfortably in view.
float2 projectPrevNdc(float3 worldPos, out bool valid) {
    float4 clip = mul(worldPush.prevViewProj, float4(worldPos - worldPush.camOffset + worldPush.camDelta, 1.0));
    valid = clip.w > 0.0;
    return valid ? clip.xy / clip.w : float2(0.0, 0.0);
}

// Previous-frame screen position of a planar reflection. The reflected image of a world point P seen
// in a planar reflector is the MIRROR image V = mirror(P) across the surface plane: the eye sees V
// along a straight line, so the reflection appears at proj(V). Project the mirror image directly.
// Gotcha: intersecting a reflection ray with the surface plane divides by dot(mirrorDir, n), which
// explodes at grazing angles and produces non-zero MV in a static scene. Mirroring is division-free, so
// static MV is bit-exact zero at any incidence. The reflector is assumed static (terrain / water);
// reflectedMotionPrev carries the reflected content's own displacement.
float2 previousReflectionNdc(float3 surfacePos, float3 surfaceNormal, float3 reflectedWorldPos,
                             float3 reflectedMotionPrev, out bool valid) {
    float3 n = normalize(surfaceNormal); // mirror(P) is orientation-independent, so no viewer flip needed
    float3 prevHit = reflectedWorldPos - reflectedMotionPrev;
    float3 mirroredHit = prevHit - 2.0 * n * dot(prevHit - surfacePos, n);
    return projectPrevNdc(mirroredHit, valid);
}

// Reflection motion remains owned by the physical foreground interface even when transmission later
// replaces the ordinary guide tuple. Guide tracing is intentionally non-SER (traceGuide above).
float2 specularReflectionMotion(SpecSurface surface, float3 primaryDir,
                                float2 currentNdc, float2 size, float primaryConeSpread) {
    if (length(surface.normal) < 0.5
            || max(surface.albedo.r, max(surface.albedo.g, surface.albedo.b)) <= 0.001
            || surface.roughness > SPEC_MOTION_ALPHA_MAX) {
        return float2(0.0, 0.0);
    }

    float3 surfacePos = surface.camRel + worldPush.camOffset;
    float3 n = normalize(surface.normal);
    float3 specDir = reflect(primaryDir, n);
    // biasNormal is a closest-hit normal, already unit length, and only its sign matters here.
    traceGuide(CULL_SECONDARY,
            offsetSurfaceOrigin(surfacePos, surface.biasNormal, specDir, SURF_BIAS),
            RAY_TMIN, specDir, 10000.0,
            max(length(surface.camRel) * primaryConeSpread, RAY_CONE_MIN_WIDTH),
            max(primaryConeSpread, RAY_CONE_MIN_SPREAD));

    float3 reflectedHit;
    float3 reflectedMotionPrev;
    if (payload.hitT > 0.0) {
        reflectedHit = surfacePos + specDir * payload.hitT;
        reflectedMotionPrev = payloadMaterial() == MATERIAL_OPAQUE
                ? payload.motionPrev : float3(0.0, 0.0, 0.0);
    } else {
        reflectedHit = surfacePos + specDir * 1.0e6;
        reflectedMotionPrev = float3(0.0, 0.0, 0.0);
    }
    float3 previousN = length(surface.previousNormal) >= 0.5
            ? normalize(surface.previousNormal) : n;
    bool prevValid;
    float2 prevNdc = previousReflectionNdc(
            surfacePos, previousN, reflectedHit, reflectedMotionPrev, prevValid);
    return prevValid ? (prevNdc - currentNdc) * 0.5 * size : float2(0.0, 0.0);
}

// Deterministic transmitted destination for clear-interface RR guides. The first interface has already
// been crossed. Continue through every crossed dielectric until opaque/particle content or sky is reached;
// TIR changes the walk into a reflected secondary ray instead of producing a false foreground guide.
void setTransmissionGuide(float3 hitCamRel, float3 motionPrev, float3 normal,
                          float roughness, float3 diffuseAlbedo) {
    gv_hitCamRel = hitCamRel;
    gv_motionHitCamRel = hitCamRel;
    gv_motionObjDisp = motionPrev;
    gv_motionUseRefracted = true;
    gv_normal = normal;
    gv_rough = roughness;
    gv_albedo = diffuseAlbedo;
}

void resolveTransmissionGuide(float3 surfacePos, float3 transmittedDir,
                              float3 surfaceBiasNormal,
                              float currentIor, float rayBias,
                              float rayConeWidth, float rayConeSpread) {
    if (dot(transmittedDir, transmittedDir) <= 0.0) return;

    float3 direction = normalize(transmittedDir);
    // surfaceBiasNormal is a closest-hit normal, already unit length, and only its sign matters here.
    float3 ro = offsetSurfaceOrigin(surfacePos, surfaceBiasNormal, direction, rayBias);
    currentIor = max(currentIor, 1.0);

    for (uint crossing = 0u; crossing < MAX_TRANSMISSION_GUIDE_HITS; ++crossing) {
        traceGuide(CULL_PRIMARY, ro, RAY_TMIN, direction, 10000.0,
                rayConeWidth, rayConeSpread);
        if (payload.hitT <= 0.0) {
            setTransmissionGuide((ro + direction * 1.0e6) - worldPush.camOffset,
                    float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0), 1.0, SKY_DIFF_ALBEDO);
            return;
        }

        uint material = payloadMaterial();
        float3 interfacePos = ro + direction * payload.hitT;
        if (material == MATERIAL_OPAQUE || material == MATERIAL_PARTICLE) {
            float endpointRoughness = material == MATERIAL_PARTICLE
                    ? 1.0 : clamp(payloadRoughness(), 0.0, 1.0);
            float3 endpointAlbedo = material == MATERIAL_PARTICLE
                    ? payload.albedo
                    : payload.albedo * (1.0 - clamp(payloadMetalness(), 0.0, 1.0));
            setTransmissionGuide(interfacePos - worldPush.camOffset, payload.motionPrev,
                    payload.normal, endpointRoughness, endpointAlbedo);
            return;
        }

        if (material != MATERIAL_WATER && material != MATERIAL_DIELECTRIC) return;

        // Mirror the radiance path's refraction. Absorption inside the crossed media is deliberately NOT
        // folded into the albedo guide: RR demodulates by albedo, and a distance-dependent attenuation
        // baked into it would disagree with the colour, which is where the extinction actually lands.
        float3 interfaceNormal = normalize(payload.normal);
        float3 interfaceBiasNormal = interfaceNormal;
        if (material == MATERIAL_WATER && (worldPush.flags & 16u) != 0u) {
            float waterFootprint = rayConeWidth
                    / max(abs(dot(-direction, interfaceBiasNormal)), 0.2);
            interfaceNormal = applyWaterWaves(interfaceNormal,
                    interfacePos.xz + worldPush.waterAnchor.xy,
                    worldPush.waterParams.w, waterFootprint);
        }
        // Exiting returns to air: the walk keeps only the current index, which is the same depth-1
        // approximation the guide already made and is exact for a single enclosing volume.
        float targetIor = payloadDielectricEntering() ? max(payloadIor(), 1.0) : 1.0;
        float3 nextDirection = refract(direction, interfaceNormal, currentIor / targetIor);
        if (dot(nextDirection, nextDirection) <= 0.0) {
            // Total internal reflection: there is no transmitted destination to describe, and following
            // the reflection instead would be actively harmful. setTransmissionGuide marks its endpoint
            // as its own feature, whose MV is that point's own reprojection delta — valid for a
            // refracted point (the near-constant refraction offset cancels between frames) but WRONG for
            // a mirror image, which sweeps at roughly twice the camera's rate and in the opposite sense.
            // That produced motion vectors that grew with camera motion.
            //
            // Bailing leaves the foreground interface tuple the caller already wrote, which is what TIR
            // physically deserves: F is 1, so specular albedo is 1 and diffuse albedo is 0, the pixel is
            // pure reflection, and gSpecMotion already describes it with a proper mirror-image
            // reprojection. Depth stays on the interface, which is the only real surface here.
            return;
        }
        direction = normalize(nextDirection);
        currentIor = targetIor;
        ro = offsetSurfaceOrigin(
                interfacePos, interfaceBiasNormal, direction, SURF_BIAS);
    }
}

// One Monte Carlo path from (ro, rd). Returns accumulated radiance along the path.
float3 tracePath(float3 ro, float3 rd, float primaryConeSpread, uint2 pix, uint sampleIndex,
                 bool captureGuides, inout uint seed) {
    float3 L = float3(0.0, 0.0, 0.0);
    float3 throughput = float3(1.0, 1.0, 1.0);
    pathPrimaryDielectricHit = false;
    pathPrimaryDielectricF = 0.0;
    float rayConeWidth = 0.0;
    float rayConeSpread = max(primaryConeSpread, RAY_CONE_MIN_SPREAD);
    int maxBounces = int(worldPush.maxBounces);
    int rrStart = maxBounces <= 3 ? 1 : 2;
    // RIS emitter NEE: direct lighting from block emitters is active when lights are published and the
    // candidate count is non-zero. Off => emitters are gathered only on path hits (legacy behaviour).
    bool risOn = worldPush.risCandidates > 0u && worldPush.lightCount > 0u && pc.lightBufAddr != 0;
    // Independent light-proposal RNG. Screen-space coherent selection is intentionally deferred until
    // there is a real presampled RIS tile buffer; direct sharing made tile-shaped lighting noise visible.
    // Measured: making a 16x16 tile share this seed recovers at most 1.3ms, so a future pool should share
    // the POOL rather than the seed — the win is in removing dependent loads, not in coherence.
    uint proposalSeed = pix.x * 1973u + pix.y * 9277u + 26699u
            ^ worldPush.frameIndex * 2654435761u ^ sampleIndex * 2246822519u;
    proposalSeed = pcg(proposalSeed);
    // Start submerged if the camera is in water, so the first segment already carries the right relative
    // index and absorption when the eye is underwater. The camera-biome tint is the correct extinction
    // until the first water surface is crossed, after which each hit supplies its own body's tint.
    MediumStack medium = makeMediumStack((worldPush.flags & 1u) != 0u
            ? makeDielectricMedium(true, WATER_IOR, worldPush.waterParams.xyz, 1.0)
            : airMedium());
    bool waterWaves = (worldPush.flags & 16u) != 0u; // animated wave-normal perturbation
    // Sky-disc gate (see world.rmiss): the sun/moon disc is shown to the primary ray and to rays spawned
    // by a specular/dielectric bounce (mirror reflections + refractions of the sun/moon), but hidden from
    // diffuse-sampled continuation rays — those are already covered by the sun NEE, so the tiny disc
    // would be a double-counted firefly. Reset true on every specular/water bounce, false on a diffuse one.
    bool showCelestial = true;
    // Number of surfaces this path has SHADED so far, as distinct from `bounce`, which also counts
    // dielectric interfaces. Glass and water continue without shading anything, so a wall behind a pane
    // is bounce 1 but diffuseDepth 0 — visually primary, and it must be treated as such. See
    // RIS_MAX_DIFFUSE_DEPTH.
    int diffuseDepth = 0;
    for (int bounce = 0; bounce <= maxBounces; bounce++) {
        // Radiance SBT records run any-hit only for true alpha cutout. Translucent/water go straight to
        // closest-hit for dielectric handling. Geometry is double-sided; the chit flips the normal.
        // Primary (bounce 0) is the camera ray (CULL_PRIMARY): sees particles but not the first-person
        // player. Bounce rays are secondary (CULL_SECONDARY): exclude particles, include the player.
        traceRadianceReordered(bounce == 0 ? CULL_PRIMARY : CULL_SECONDARY, ro, 0.0, rd, 10000.0,
                showCelestial, rayConeWidth, rayConeSpread);

        if (payload.hitT < 0.0) {
            // world.rmiss writes sky radiance (gradient + sun/moon disc + stars) into payload.albedo. The
            // packed show-celestial flag decides whether this path segment may see the bright disc.
            float3 sky = payload.albedo;
            if (bounce == 0 && captureGuides) { // primary ray hit sky: guides describe "no surface"
                gv_normal = float3(0.0, 0.0, 0.0);
                // Flat mid-grey albedo guide for sky (NOT the bright sky radiance): demodulating DLSS-RR
                // by the huge sun-disc radiance rings/artifacts around the sun. A constant albedo keeps
                // the brightness in the lighting channel where RR resolves it cleanly.
                gv_albedo = SKY_DIFF_ALBEDO;
                gv_rough = 1.0;
                gv_emission = 0.0;
                gv_emissionSource = EMISSION_SOURCE_NONE;
                gv_hitCamRel = rd * 1.0e6; // sky at infinity: only rotation parallax (-> large far depth)
                gv_motionHitCamRel = gv_hitCamRel;
                gv_motionUseRefracted = false;
                gv_motionObjDisp = float3(0.0, 0.0, 0.0); // sky is static
                // Zero normal: specularReflectionMotion rejects this and returns a zero reflection MV.
                gv_spec = makeSpecSurface(gv_hitCamRel, gv_normal, 1.0, SKY_SPEC_ALBEDO);
            }
            L += throughput * sky; // escaped to sky
            break;
        }

        // Beer–Lambert: attenuate along the segment just travelled by the medium it lay inside. Applies
        // to every hit reached while inside a volume dielectric (its own exit face, or whatever content
        // lies within it), shifting the transmitted radiance with distance. Air's extinction is zero, so
        // the ordinary case costs one comparison.
        if (any(medium.current.extinction > 0.0)) {
            throughput *= exp(-medium.current.extinction * payload.hitT);
        }

        float3 n = payload.normal; // oriented toward the incoming ray by the closest-hit
        float3 hitPos = ro + rd * payload.hitT;
        rayConeWidth = max(rayConeWidth + rayConeSpread * max(payload.hitT, 0.0), RAY_CONE_MIN_WIDTH);
        uint material = payloadMaterial();
        if (bounce == 0 && captureGuides) {
            gv_emission = payloadEmission();
            gv_emissionSource = payloadEmissionSource();
        }

        // ---- Dielectric interface: water, stained glass, ice. ONE handler and one behaviour — every
        // dielectric is a volume. Fresnel from the relative index across the face, then reflect or
        // refract; no diffuse response and no sun NEE, because the surface is specular. Transmission
        // pushes or pops a participating medium, and the segment travelled inside it is attenuated by
        // that medium's extinction rather than by a tint multiply at the interface.
        //
        // Water is the only material that adds anything: the animated wave normal, and a medium marked
        // for the caustic term. Which side of the face we are on comes from the geometry
        // (payloadDielectricEntering), so a stray or missing face cannot desync the medium for the rest
        // of the path.
        if (material == MATERIAL_WATER || material == MATERIAL_DIELECTRIC) {
            bool isWater = material == MATERIAL_WATER;
            bool entering = payloadDielectricEntering();
            float3 tint = payload.albedo;
            float transmission = clamp(payloadTransmission(), 0.0, 1.0);

            // Ripple the (near-flat) surface normal before any Fresnel/guide use → glints, broken
            // reflections, wobbling refraction. The wave domain is world-stable: rebased hit XZ plus the
            // pushed anchor (terrain origin mod 4096) survives the per-move terrain rebase.
            float3 geometricNormal = n; // unrippled; ray origins offset along this
            float3 previousNormal = n;  // wave normal one frame ago, for the reflection MV
            if (isWater && waterWaves) {
                float2 waterDomain = hitPos.xz + worldPush.waterAnchor.xy;
                float waterFootprint = rayConeWidth / max(abs(dot(-rd, geometricNormal)), 0.2);
                if (bounce == 0 && captureGuides && abs(geometricNormal.y) >= 0.5) {
                    float2 currentGrad;
                    float2 previousGrad;
                    waterWaveGradTemporal(waterDomain, worldPush.waterParams.w,
                            worldPush.waterAnchor.z, waterFootprint, currentGrad, previousGrad);
                    float orientation = geometricNormal.y >= 0.0 ? 1.0 : -1.0;
                    n = orientation * normalize(float3(-currentGrad.x, 1.0, -currentGrad.y));
                    previousNormal = orientation
                            * normalize(float3(-previousGrad.x, 1.0, -previousGrad.y));
                } else {
                    n = applyWaterWaves(geometricNormal, waterDomain,
                            worldPush.waterParams.w, waterFootprint);
                }
            }

            // The medium this face opens into, and the one it returns to on the way out — which is what
            // the stack remembers.
            Medium entered = makeDielectricMedium(isWater, max(payloadIor(), 1.0), tint, transmission);
            float etaI = medium.current.ior;
            float etaT = entering ? entered.ior : medium.outer.ior;

            float cosI = clamp(dot(-rd, n), 0.0, 1.0);
            float F = fresnelDielectric(cosI, etaI, etaT);
            float3 transmittedDir = refract(rd, n, etaI / etaT);
            // The translucent terrain layer is recessed by TRANSLUCENT_INSET (RtTerrainMesher), so a
            // glass/ice face touching a slab or stair sits a hair behind that neighbour's surface. A full
            // SURF_BIAS would restart the transmitted ray past the neighbour and see through it. Water
            // comes from the fluid mesher, which applies no inset, so it takes the ordinary bias.
            float transmitBias = isWater ? SURF_BIAS : INSET_TRANSMIT_BIAS;

            if (bounce == 0 && pathPrimaryDielectricBranch != PRIMARY_DIELECTRIC_RANDOM) {
                pathPrimaryDielectricHit = true;
                pathPrimaryDielectricF = F;
            }
            if (bounce == 0 && captureGuides) { // feed RR a smooth dielectric specular surface
                gv_normal = n;
                gv_rough = 0.0; // an exact Fresnel interface, not a finite GGX lobe
                gv_albedo = float3(0.0, 0.0, 0.0); // fallback when TIR leaves no transmitted layer
                gv_hitCamRel = hitPos - worldPush.camOffset;
                gv_motionHitCamRel = gv_hitCamRel;
                gv_motionUseRefracted = false;
                gv_motionObjDisp = isWater ? float3(0.0, 0.0, 0.0) : payload.motionPrev;
                gv_spec = makeSpecSurface(gv_hitCamRel, n, previousNormal, geometricNormal,
                        gv_rough, float3(F, F, F));

                if (dot(transmittedDir, transmittedDir) > 0.0) {
                    resolveTransmissionGuide(hitPos, transmittedDir, geometricNormal,
                            entering ? entered.ior : medium.outer.ior,
                            transmitBias, rayConeWidth, rayConeSpread);
                }
            }

            bool chooseReflection = bounce == 0
                    && pathPrimaryDielectricBranch != PRIMARY_DIELECTRIC_RANDOM
                    ? pathPrimaryDielectricBranch == PRIMARY_DIELECTRIC_REFLECTION
                    : rndf(seed) < F;
            if (chooseReflection) {
                rd = reflect(rd, n);
                ro = offsetSurfaceOrigin(hitPos, geometricNormal, rd, SURF_BIAS);
            } else {
                if (dot(transmittedDir, transmittedDir) <= 0.0) break; // TIR with F < 1 cannot happen
                rd = normalize(transmittedDir);
                ro = offsetSurfaceOrigin(hitPos, geometricNormal, rd, transmitBias);
                // Crossing into or out of the volume. Absorption is the medium's job from here, so the
                // tint is NOT also multiplied into the throughput — that would double-count it.
                if (entering) {
                    mediumPush(medium, entered);
                } else {
                    mediumPop(medium);
                }
            }
            showCelestial = true; // specular interface: the continuation ray may see the sun/moon disc
            if (bounce >= rrStart) {
                float q = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.02, 1.0);
                if (rndf(seed) > q) {
                    break;
                }
                throughput /= q;
            }
            continue;
        }

        // Particles (material == 2): camera-facing receiver billboard. The instance mask keeps particles
        // off secondary rays, so this only fires at bounce 0; they receive direct light and one diffuse
        // path continuation into the scene, but never appear in shadows/reflections/GI or emit radiance.
        if (material == MATERIAL_PARTICLE) {
            float3 albedo = payload.albedo;
            if (captureGuides) {
                gv_normal = n;
                gv_albedo = albedo; // RR demodulation target = the particle texel (keeps it sharp)
                gv_rough = 1.0;
                gv_hitCamRel = hitPos - worldPush.camOffset;
                gv_motionHitCamRel = gv_hitCamRel;
                gv_motionUseRefracted = false;
                gv_motionObjDisp = payload.motionPrev; // per-particle MV: de-ghost moving particles
                // Zero specular albedo: no reflection MV is traced for a diffuse billboard.
                gv_spec = makeSpecSurface(gv_hitCamRel, n, 1.0, float3(0.0, 0.0, 0.0));
            }

            float3 lightDir = worldPush.lightDir.xyz;
            float lightHalfAngle = worldPush.lightDir.w;
            if (lightHalfAngle > 0.0) {
                lightDir = sampleSquare(lightDir, lightHalfAngle, seed);
            }
            // Billboards are effectively two-sided receivers. Bias from the side facing the light; shadow
            // rays exclude particles anyway, but this keeps the origin sane when the light is behind the
            // camera side.
            float signedNdl = dot(n, lightDir);
            float ndl = abs(signedNdl);
            if (ndl > 0.0) {
                float3 shadowOrigin = hitPos + (signedNdl >= 0.0 ? n : -n) * SURF_BIAS;
                float3 vis = visibility(shadowOrigin, lightDir, 10000.0).transmittance;
                if (max(vis.r, max(vis.g, vis.b)) > 0.0) {
                    L += throughput * albedo * INV_PI * worldPush.lightRadiance.xyz * ndl * vis;
                }
            }

            // RIS direct lighting from block emitters (lava, glowstone, torches, ...) — same reservoir
            // sampler as terrain/entities, but two-sided: a billboard has no back face, so light
            // striking either side of the quad should still land (matches the sun/moon NEE above).
            if (risOn && diffuseDepth <= int(RIS_MAX_DIFFUSE_DEPTH)) {
                float3 v = -rd;
                Reservoir r = risInitial(hitPos, n, v, rd, albedo, float3(0.0, 0.0, 0.0), 1.0,
                        true, 0.0, uint(diffuseDepth), seed, proposalSeed);
                float3 risVis;
                L += throughput * shadeReservoir(r, hitPos, n, v, rd, albedo, float3(0.0, 0.0, 0.0), 1.0,
                        true, 0.0, risVis);
            }

            if (bounce >= maxBounces) {
                break;
            }
            throughput *= albedo;
            ro = hitPos + n * SURF_BIAS;
            rd = cosineDir(n, seed);
            rayConeSpread = max(rayConeSpread, RAY_CONE_DIFFUSE_SPREAD);
            showCelestial = false; // diffuse receiver: direct sun/moon was handled by NEE above
            diffuseDepth++;        // a billboard is a shaded surface, unlike the dielectric branches
            continue;
        }

        float3 albedo = payload.albedo;
        float sss = payloadSss(); // LabPBR SSS strength (0 when absent)
        float3 p = hitPos + n * SURF_BIAS;
        float3 v = -rd; // view direction (toward the camera / incoming ray)

        // Material split. Metals tint F0 by albedo and have no diffuse lobe; dielectrics use a fixed
        // 0.04 F0, sourced from the chit (LabPBR custom/metal F0, or the dielectric default it applies).
        float rough = clamp(payloadRoughness(), 0.0, 1.0);
        float metal = clamp(payloadMetalness(), 0.0, 1.0);
        // Unresolvably tight lobes become exact delta mirrors; everything else keeps its authored value.
        // Snapping to 0 also propagates into the guide buffers, which is what tells RR to hold the
        // reflection sharp instead of accumulating it as a glossy lobe.
        bool exactSpecular = rough <= MIRROR_ALPHA_MAX;
        rough = exactSpecular ? 0.0 : rough;
        float3 diffAlb = albedo * (1.0 - metal);
        float3 F0 = payload.f0;

        if (bounce == 0 && captureGuides) { // primary-visibility surface: capture the denoiser guide buffers
            gv_normal = n;
            gv_albedo = diffAlb;  // RR diffuse-albedo demodulation target
            gv_rough = rough;
            gv_hitCamRel = ro + rd * payload.hitT - worldPush.camOffset; // camera-relative hit position
            gv_motionHitCamRel = gv_hitCamRel;
            gv_motionUseRefracted = false;
            gv_motionObjDisp = payload.motionPrev; // per-vertex world motion (0 for static terrain/sky)
            gv_spec = makeSpecSurface(gv_hitCamRel, n, rough,
                    rrSpecularAlbedo(F0, rough, dot(n, v)));
        }

        // Emissive surfaces (lava, glowstone, torches, ...) add radiance directly, colored by albedo.
        // RIS emitter NEE: the direct-hit emission term is gated only for emitters RIS actually samples
        // (payload emitter-in-list bit) — and then only off diffuse continuation rays (showCelestial
        // false), which the previous vertex's RIS already accounted for; the primary ray and
        // specular/dielectric bounces (showCelestial true) still add it, since no emitter NEE happened
        // along those. Emitters NOT in the light buffer (sparse/sub-threshold footprints, entities) are
        // never NEE-sampled, so they always gather — bit-identical to the no-NEE path, no energy lost.
        // Emission is already the full HDR strength (EMISSIVE_STRENGTH baseline * any override
        // multiplier baked in Java at material-compile time — see MaterialHeader.features packing).
        float emission = payloadEmission();
        // One gate drives both the RIS call below and the direct-hit emission suppression here. They must
        // agree: gateEmitter exists only because RIS already accounted for this emitter, so a bounce where
        // RIS is skipped must also gather emission directly or that light is lost outright.
        bool risActive = risOn && diffuseDepth <= int(RIS_MAX_DIFFUSE_DEPTH);
        bool gateEmitter = risActive && payloadEmitterInList();
        if (emission > 0.0 && (!gateEmitter || showCelestial)) {
            L += throughput * albedo * emission;
        }

        // NEE: direct light from the dominant celestial body (sun by day, moon by night) — Lambert
        // diffuse + GGX specular, one shadow ray. Jitter the light direction within its square angular
        // extent (worldPush.lightDir.w half-angle) so the shadow ray samples the light's quad — soft penumbrae
        // over accumulation. The same sampled direction drives ndl, the shadow ray, and the specular
        // half-vector.
        float3 lightDir = worldPush.lightDir.xyz;
        float lightHalfAngle = worldPush.lightDir.w;
        if (lightHalfAngle > 0.0) {
            lightDir = sampleSquare(lightDir, lightHalfAngle, seed);
        }
        float ndl = max(0.0, dot(n, lightDir));
        if (ndl > 0.0) {
            VisibilityResult shadow = visibility(p, lightDir, 10000.0);
            float3 vis = shadow.transmittance;
            // Underwater receiver whose shadow ray crossed a water surface: scale the direct light by
            // the wave-refraction caustic at the exit point. Focusing scales the incident irradiance,
            // so it applies to the whole NEE term (diffuse + specular).
            if (medium.current.water && waterWaves && shadow.waterHitT > 0.0) {
                vis *= waterCaustic(p + lightDir * shadow.waterHitT, lightDir, shadow.waterHitT);
            }
            if (max(vis.r, max(vis.g, vis.b)) > 0.0) {
                float3 brdf = diffAlb * INV_PI; // Lambertian diffuse (f = albedo/PI)
                float3 h = normalize(lightDir + v);
                float ndh = max(0.0, dot(n, h));
                float ndv = max(1.0e-4, dot(n, v));
                float vdh = max(0.0, dot(v, h));
                float D = ggxD(ndh, rough);
                float G = ggxG1(ndv, rough) * ggxG1(ndl, rough); // separable Smith masking
                float3 F = fresnelSchlick(vdh, F0);
                brdf += (D * G) * F / (4.0 * ndv * ndl); // Cook–Torrance specular
                L += throughput * brdf * worldPush.lightRadiance.xyz * ndl * vis;
            }
        }

        // RIS direct lighting from block emitters (single-frame only, no temporal reservoir reuse).
        // Front lobe + SSS backlighting are resampled together and share ONE shadow ray — a light can
        // only be in front of n or behind it, never both, so combining them into one target function is
        // exact, not an approximation. The SSS term is gated off past MAX_SSS_DIFFUSE_DEPTH by passing
        // sss=0 there (falls back to plain front-only RIS). Keyed on diffuseDepth, not `bounce`, for the
        // same reason as RIS_MAX_DIFFUSE_DEPTH above: leaves seen through a window are still the first
        // surface this path has shaded, and should still backlight.
        if (risActive) {
            float activeSss = diffuseDepth <= MAX_SSS_DIFFUSE_DEPTH ? sss : 0.0;
            Reservoir r = risInitial(hitPos, n, v, rd, diffAlb, F0, rough, false, activeSss,
                    uint(diffuseDepth), seed, proposalSeed);
            float3 risVis;
            L += throughput * shadeReservoir(r, hitPos, n, v, rd, diffAlb, F0, rough, false,
                    activeSss, risVis);
        }

        // Thin-surface SSS transmission. Light entering from the back face scatters through toward the
        // viewer via a forward-biased HG phase (leaves/grass backlit by the sun glow when you look
        // through them toward the light source). Shadow ray fires from the back face to avoid
        // self-occlusion. cosT = dot(lightDir, rd): rd tracks the ray toward the sun when the sun is
        // behind the slab, giving cosT ≈ 1 → forward-scatter peak.
        if (sss > 0.0 && diffuseDepth <= MAX_SSS_DIFFUSE_DEPTH) {
            float backNdl = max(0.0, dot(-n, lightDir));
            if (backNdl > 0.0) {
                VisibilityResult shadowBack = visibility(hitPos - n * SURF_BIAS, lightDir, 10000.0);
                float3 visB = shadowBack.transmittance;
                // Same caustic as the front-face NEE — underwater kelp/seagrass transmission should
                // flicker with the same light bands as the floor around it.
                if (medium.current.water && waterWaves && shadowBack.waterHitT > 0.0) {
                    visB *= waterCaustic(hitPos + lightDir * shadowBack.waterHitT,
                            lightDir, shadowBack.waterHitT);
                }
                if (max(visB.r, max(visB.g, visB.b)) > 0.0) {
                    float cosT = dot(lightDir, rd);
                    L += throughput * diffAlb * sss * SSS_STRENGTH * hg(cosT, SSS_G) * backNdl * worldPush.lightRadiance.xyz * visB;
                }
            }
        }

        // Every shading term for this surface is now accounted for, so it counts toward the shaded depth.
        // The dielectric branches above return before reaching here, which is exactly the distinction
        // between this and `bounce`.
        diffuseDepth++;

        // Indirect continuation: importance-sample the diffuse OR the GGX specular lobe, chosen by their
        // relative reflectance.
        float ps = exactSpecular && luminance(diffAlb) <= 1.0e-6
                ? 1.0
                : clamp(luminance(F0) / (luminance(F0) + luminance(diffAlb) + 1.0e-4), 0.1, 0.9);
        if (rndf(seed) < ps) {
            float3 l;
            if (exactSpecular) {
                // Authored zero is a delta distribution, not a narrow finite GGX lobe. This avoids the
                // degenerate VNDF at alpha=0 and preserves a genuinely sharp reflection without a floor.
                l = reflect(rd, n);
                throughput *= fresnelSchlick(clamp(dot(n, v), 0.0, 1.0), F0) / ps;
            } else {
                float3 h = sampleGGXVNDF(n, v, rough, seed);
                l = reflect(rd, h); // rd = -v: reflect the incoming ray about the microfacet normal
                float ndl2 = dot(n, l);
                if (ndl2 <= 0.0) {
                    break; // microfacet reflects below the surface: terminate this path
                }
                // VNDF + separable Smith ⇒ weight = F · G2/G1(v) = F · G1(l); divide by the lobe pdf ps.
                float3 F = fresnelSchlick(max(0.0, dot(v, h)), F0);
                throughput *= F * ggxG1(ndl2, rough) / ps;
                rayConeSpread = max(rayConeSpread, rough * RAY_CONE_GLOSSY_SPREAD_SCALE);
            }
            ro = p;
            rd = l;
            showCelestial = true; // specular lobe: this ray is a (glossy) mirror — show the disc
        } else {
            throughput *= diffAlb / (1.0 - ps);
            ro = p;
            rd = cosineDir(n, seed);
            rayConeSpread = max(rayConeSpread, RAY_CONE_DIFFUSE_SPREAD);
            showCelestial = false; // diffuse lobe: disc covered by NEE, hide it (anti-firefly)
        }

        // Russian roulette on deeper bounces: survive with prob = max channel, rescale survivors.
        if (bounce >= rrStart) {
            float q = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.02, 1.0);
            if (rndf(seed) > q) {
                break;
            }
            throughput /= q;
        }
    }
    return L;
}

[shader("raygeneration")]
void main() {
    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0]; // load the per-frame push data from the BDA buffer
    int2 pix = int2(DispatchRaysIndex().xy);
    float2 size = float2(DispatchRaysDimensions().xy);
    float2 uv = (float2(pix) + 0.5) / size;
    float2 ndc = uv * 2.0 - 1.0; // unjittered pixel centre — the motion vector reprojects this

    // Offset the primary ray within the pixel by the sub-pixel jitter reported to DLSS-RR, so
    // successive frames sample different positions and RR can resolve detail above render resolution.
    // The guide buffers below come from this jittered first hit (they must match the jittered colour);
    // only the motion vector stays on the unjittered centre, which DLSS expects.
    float2 jndc = (uv + worldPush.jitter / size) * 2.0 - 1.0;

    float4 nearH = mul(worldPush.invViewProj, float4(jndc.x, jndc.y, 1.0, 1.0));
    float4 farH = mul(worldPush.invViewProj, float4(jndc.x, jndc.y, 0.0, 1.0));
    float3 nearP = nearH.xyz / nearH.w;
    float3 farP = farH.xyz / farH.w;

    float3 origin = nearP + worldPush.camOffset;
    float3 dir = normalize(farP - nearP);
    float rayConeSpread = primaryRayConeSpread(jndc, size, dir);

    uint seed = (uint(pix.x) * 1973u + uint(pix.y) * 9277u + 26699u) ^ (worldPush.frameIndex * 2654435761u);

    uint spp = max(worldPush.spp, 1u);
    float3 frameRadiance = float3(0.0, 0.0, 0.0);
    for (uint s = 0u; s < spp; s++) {
        seed = pcg(seed); // decorrelate per-sample paths (shared camera ray, no AA jitter yet)
        uint transmissionSeed = seed;
        pathPrimaryDielectricBranch = PRIMARY_DIELECTRIC_TRANSMISSION;
        // Guides describe one deterministic surface, so only the first sample's transmission branch
        // writes them; the reflection branch and every later sample reuse what it captured.
        float3 transmissionRadiance = tracePath(origin, dir, rayConeSpread, uint2(pix), s,
                s == 0u, transmissionSeed);

        bool primaryDielectricHit = pathPrimaryDielectricHit;
        float primaryDielectricF = pathPrimaryDielectricF;
        float3 sampleRadiance = transmissionRadiance;

        if (primaryDielectricHit) {
            uint reflectionSeed = seed ^ 0xa511e9b3u;
            reflectionSeed = pcg(reflectionSeed);
            pathPrimaryDielectricBranch = PRIMARY_DIELECTRIC_REFLECTION;
            float3 reflectionRadiance = tracePath(origin, dir, rayConeSpread, uint2(pix), s + spp,
                    false, reflectionSeed);

            float dielectricF = saturate(primaryDielectricF);
            sampleRadiance =
                    dielectricF * reflectionRadiance
                    + (1.0 - dielectricF) * transmissionRadiance;
            seed = transmissionSeed ^ reflectionSeed;
            seed = pcg(seed);
        } else {
            seed = transmissionSeed;
        }

        frameRadiance += sampleRadiance;
    }
    pathPrimaryDielectricBranch = PRIMARY_DIELECTRIC_RANDOM;
    frameRadiance /= float(spp);

    // Reflection remains owned by the foreground interface even when glass/water replaced the ordinary
    // tuple with the transmitted destination.
    float2 specMotion = specularReflectionMotion(gv_spec, dir, jndc, size, rayConeSpread);

    // Hardware (non-linear, reversed-Z) depth for DLSS-RR + Frame Generation: project the camera-relative
    // primary hit through the forward view-projection and take ndc z/w — exactly the value a rasterizer
    // would write. Reversed-Z (near=1, far=0), so DLSS gets the DepthInverted flag; sky's far hit -> ~0.
    // Clear transmission uses destination depth so the entire ordinary tuple describes one layer.
    float4 curClip = mul(worldPush.curViewProj, float4(gv_hitCamRel, 1.0));
    gv_depth = curClip.w > 0.0 ? curClip.z / curClip.w : 0.0;

    // Motion vector: reproject this guide point through the previous frame's view-projection.
    // The primary ray was cast through the JITTERED ndc (jndc), so this hit's current screen position is
    // jndc, NOT the pixel centre. Subtract jndc so the MV is jitter-free (= 0 when static), which is
    // what DLSS expects with MVJittered unset. Scaled from NDC into render-pixel space.
    // For a dynamic surface, previous-frame position is the current point minus its world-space
    // displacement; subtracting gv_motionObjDisp de-cameras the MV.
    float4 prevClip = mul(worldPush.prevViewProj, float4(gv_motionHitCamRel + worldPush.camDelta - gv_motionObjDisp, 1.0));
    float2 curNdc = jndc; // primary hit's current screen position is exactly the jittered ray ndc
    // Transmitted content is its own feature: compare its previous and current projections rather than
    // subtracting the primary interface's jittered NDC.
    float4 curClipMotion = gv_motionUseRefracted
            ? mul(worldPush.curViewProj, float4(gv_motionHitCamRel, 1.0))
            : float4(0.0, 0.0, 0.0, 1.0); // unused; keeps the guard below uniform
    // Guard both perspective divides the way the depth above does. A guide point can land behind either
    // camera — most easily a transmitted destination seen at a grazing angle, or one that leaves the
    // frustum as the camera turns — and w at or below zero turns the divide into an arbitrarily large
    // vector that DLSS-RR reprojects with. No valid previous position exists in that case, so report no
    // motion and let RR fall back instead of handing it garbage.
    float2 motion;
    if (prevClip.w > 0.0 && (!gv_motionUseRefracted || curClipMotion.w > 0.0)) {
        if (gv_motionUseRefracted) curNdc = curClipMotion.xy / curClipMotion.w;
        motion = (prevClip.xy / prevClip.w - curNdc) * 0.5 * size;
    } else {
        motion = float2(0.0, 0.0);
    }

    // Guide buffers: first-hit or coherently replaced attributes consumed by the denoiser/DLSS-RR.
    gNormal[pix] = float4(gv_normal, gv_rough);
    gAlbedo[pix] = float4(gv_albedo, 1.0);
    gDepth[pix] = gv_depth;
    gMotion[pix] = motion;
    gSpecAlbedo[pix] = float4(gv_spec.albedo, 1.0);
    gSpecMotion[pix] = specMotion;

    // Debug guide-buffer visualization: bypass accumulation and show a guide directly.
    if (pc.debugView != 0u) {
        float3 dbg;
        if (pc.debugView == 1u) {
            dbg = gv_normal * 0.5 + 0.5;            // world normal -> [0,1]
        } else if (pc.debugView == 2u) {
            dbg = gv_albedo;                        // diffuse albedo
        } else if (pc.debugView == 3u) {
            dbg = float3(gv_depth, gv_depth, gv_depth); // HW reversed-Z depth (near=white, far/sky=black)
        } else if (pc.debugView == 4u) {
            dbg = float3(gv_rough, gv_rough, gv_rough); // roughness
        } else if (pc.debugView == 6u) {
            dbg = gv_spec.albedo;                   // exact value bound to DLSS-RR
        } else if (pc.debugView == 7u) {
            dbg = float3(clamp(0.5 + specMotion * 0.05, 0.0, 1.0), 0.5); // reflection motion
        } else if (pc.debugView == 8u) {
            dbg = float3(gv_emission, gv_emission, gv_emission); // resolved emission mask/strength
        } else if (pc.debugView == 9u) {
            float3 sourceColor = gv_emissionSource == EMISSION_SOURCE_LAB_PBR ? float3(0.15, 0.8, 1.0)
                    : gv_emissionSource == EMISSION_SOURCE_HEURISTIC ? float3(1.0, 0.45, 0.05)
                    : gv_emissionSource == EMISSION_SOURCE_UNIFORM ? float3(1.0, 1.0, 1.0)
                    : float3(0.0, 0.0, 0.0);
            dbg = sourceColor * (gv_emissionSource == EMISSION_SOURCE_NONE
                    ? 0.0 : max(0.2, clamp(gv_emission, 0.0, 1.0)));
        } else {
            dbg = float3(clamp(0.5 + motion * 0.05, 0.0, 1.0), 0.5); // motion: red=+x, green=+y
        }
        outImage[pix] = float4(dbg, 1.0);
        return;
    }

    // Single noisy estimate; the denoiser (DLSS-RR) handles temporal convergence.
    outImage[pix] = float4(frameRadiance, 1.0);
}
