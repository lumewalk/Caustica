// Primary/guide pass (pass A of the wavefront split — see docs/WAVEFRONT_PLAN.md).
//
// Traces the camera ray through the dielectric chain to the first surface worth shading, captures the
// DLSS-RR guide buffers, and writes one resumable continuation per sample to the path queue. It shades
// nothing: no NEE, no RIS, no BSDF sampling. That is why rt_lighting and rt_indirect are absent below,
// and their absence is the point — the pass separation is enforced by this include list rather than
// left to dead-code elimination.
//
// Perspective setup: invViewProj = inverse(proj * viewRot) maps clip to camera-relative world. Primary
// rays are reconstructed from two unprojected points (near and far plane) rather than assuming the eye
// is at the origin, because Minecraft bakes view bobbing into the projection, so the effective eye comes
// from the matrix. camOffset shifts camera-relative space into the terrain's rebased coordinates. Depth
// is reversed-Z (near=1, far=0).
//
// SER: the build emits EXT and NV shader-invocation-reorder variants from this source.
import world_common;

#include "rt_core.slanginc"
#include "rt_math.slanginc"
#include "rt_medium.slanginc"
#include "rt_segment.slanginc"
#include "rt_water.slanginc"
#include "rt_trace.slanginc"
#include "rt_guides.slanginc"
#include "rt_primary.slanginc"

[shader("raygeneration")]
void main() {
    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    uint2 dispatchIndex = DispatchRaysIndex().xy;
    uint2 dimensions = DispatchRaysDimensions().xy;
    int2 pix = int2(dispatchIndex);
    float2 size = float2(dimensions);
    float2 uv = (float2(pix) + 0.5) / size;
    float2 jndc = (uv + worldPush.jitter / size) * 2.0 - 1.0;
    float4 nearH = mul(worldPush.invViewProj, float4(jndc.x, jndc.y, 1.0, 1.0));
    float4 farH = mul(worldPush.invViewProj, float4(jndc.x, jndc.y, 0.0, 1.0));
    float3 nearP = nearH.xyz / nearH.w;
    float3 farP = farH.xyz / farH.w;
    float3 origin = nearP + worldPush.camOffset;
    float3 dir = normalize(farP - nearP);
    float rayConeSpread = primaryRayConeSpread(jndc, size, dir);
    uint seed = (dispatchIndex.x * 1973u + dispatchIndex.y * 9277u + 26699u)
            ^ (worldPush.frameIndex * 2654435761u);
    MediumStack cameraMedium = makeMediumStack((worldPush.flags & 1u) != 0u
            ? makeDielectricMedium(true, WATER_IOR, worldPush.waterParams.xyz, 1.0)
            : airMedium());
    uint spp = max(worldPush.spp, 1u);
    uint pixelIndex = dispatchIndex.y * dimensions.x + dispatchIndex.x;
    DevicePtr<PackedPathSegment> queue = DevicePtr<PackedPathSegment>(pc.pathQueueAddr);

    for (uint s = 0u; s < spp; ++s) {
        seed = pcg(seed);
        PathSegment camera = makePathSegment(origin, dir, float3(1.0, 1.0, 1.0),
                cameraMedium, 0.0, rayConeSpread, seed, 0, 0, true, false, false);
        bool alive;
        PathSegment terminal = tracePrimary(camera, s == 0u, alive);
        // Store immediately: no continuation remains live across writeGuides or another primary trace.
        queue[pixelIndex * spp + s] = packPathSegment(terminal, pixelIndex, s, alive);
        if (s == 0u) {
            writeGuides(pix, dir, jndc, size, rayConeSpread);
        }
    }

    if (pc.debugView != 0u) {
        writeDebugView(pix);
    }
}
