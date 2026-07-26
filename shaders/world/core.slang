// Bindings, per-frame push state, the ray payload, and the constants every pass shares.
// Include first: everything below depends on pc, worldPush and payload.


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



uint payloadMaterial() { return payload.flags & PAYLOAD_MATERIAL_MASK; }
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

// Stored roughness IS alpha (see payloadRoughness), so these compare against it directly.
static const float MIRROR_ALPHA_MAX = 4.0e-4; // LabPBR perceptual smoothness >= 0.98
// Above this the specular lobe is too broad for a single mirrored reprojection to describe, so the
// reflection MV is left at zero and RR falls back to the ordinary MV.
static const float SPEC_MOTION_ALPHA_MAX = 0.25; // perceptual smoothness <= 0.5
static const uint MAX_TRANSMISSION_GUIDE_HITS = 8u;
// Continuations one sample may carry: the camera segment, plus the reflection branch a primary
// dielectric splits off. Raising this needs a real queue, which is what the continuation buffer is for.
static const uint MAX_PATH_SEGMENTS = 2u;

