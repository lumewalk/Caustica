// Indirect pass (pass B of the wavefront split — see docs/WAVEFRONT_PLAN.md).
//
// Resumes the continuations the primary pass queued and runs the bounce loop: NEE toward the sun, RIS
// emitter lighting, thin-surface SSS, BSDF continuation and Russian roulette. It captures no guides and
// owns no motion vectors — rt_guides and rt_primary are deliberately absent from the include list
// below, so guide state cannot leak into this pass by accident.
//
// Lighting is iterative here rather than recursive from closest-hit, so maxRayRecursionDepth stays 1.
// At each surface vertex:
//   * NEE toward the sun — a single analytic directional light, gated by one hit-object visibility
//     query (TerminateOnFirstHit | SkipClosestHit). Lambertian BRDF = albedo/PI.
//   * a cosine-weighted Lambertian continuation ray (throughput *= albedo; the PI and cos cancel).
// Sky / ambient light needs no separate AO pass: continuation rays that escape land in world.rmiss,
// which returns sky radiance — so soft sky fill and contact AO both fall out of the path integral.
//
// Each frame emits a single SPP-averaged (still noisy) estimate; temporal convergence is the denoiser's
// job. With RR disabled the output is the raw noisy path trace — a reference view, not a converged
// image. Output is HDR (R16G16B16A16_SFLOAT); the HDR -> LDR tonemap happens at the display.comp seam.
//
// SER: the build emits EXT and NV shader-invocation-reorder variants from this source.
import world_common;

#include "rt_core.slanginc"
#include "rt_math.slanginc"
#include "rt_medium.slanginc"
#include "rt_segment.slanginc"
#include "rt_water.slanginc"
#include "rt_trace.slanginc"
#include "rt_lighting.slanginc"
#include "rt_indirect.slanginc"

[shader("raygeneration")]
void main() {
    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    if (pc.debugView != 0u) {
        return; // pass A owns guide visualization output
    }
    uint2 dispatchIndex = DispatchRaysIndex().xy;
    uint2 dimensions = DispatchRaysDimensions().xy;
    int2 pix = int2(dispatchIndex);
    uint spp = max(worldPush.spp, 1u);
    uint pixelIndex = dispatchIndex.y * dimensions.x + dispatchIndex.x;
    ConstPtr<PackedPathSegment> queue = ConstPtr<PackedPathSegment>(pc.pathQueueAddr);
    float3 frameRadiance = float3(0.0, 0.0, 0.0);
    for (uint s = 0u; s < spp; ++s) {
        bool valid;
        PathSegment segment = unpackPathSegment(queue[pixelIndex * spp + s], valid);
        if (valid) {
            frameRadiance += tracePath(segment, uint2(pix), s);
        }
    }
    outImage[pix] = float4(frameRadiance / float(spp), 1.0);
}
