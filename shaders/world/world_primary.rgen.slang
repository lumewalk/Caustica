// Primary/guide pass (pass A of the wavefront split — see docs/WAVEFRONT_PLAN.md).
//
// Traces the camera ray through the dielectric chain to the first surface worth shading, captures the
// DLSS-RR guide buffers, and writes one resumable continuation per sample to the path queue. It shades
// nothing: no NEE, no RIS, no BSDF sampling. That is why lighting and indirect are absent below,
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
import world_core;
import math;
import medium;
import segment;
import water;
import trace;
import guides;

public PathSegment tracePrimary(PathSegment seg, bool captureGuides, out bool alive) {
    float3 ro = seg.ro;
    float3 rd = seg.rd;
    float3 throughput = seg.throughput;
    MediumStack medium = seg.medium;
    float rayConeWidth = seg.rayConeWidth;
    float rayConeSpread = max(seg.rayConeSpread, RAY_CONE_MIN_SPREAD);
    uint seed = seg.seed;
    bool showCelestial = seg.showCelestial;
    int diffuseDepth = seg.diffuseDepth;
    int maxBounces = int(worldPush.maxBounces);
    int rrStart = maxBounces <= 3 ? 1 : 2;
    bool waterWaves = (worldPush.flags & 16u) != 0u;
    alive = false;

    for (int bounce = seg.bounce; bounce <= maxBounces; ++bounce) {
        PathSegment terminal = makePathSegment(ro, rd, throughput, medium,
                                               rayConeWidth, rayConeSpread, seed, bounce, diffuseDepth,
                                               showCelestial, false, false);
        traceRadianceReordered(bounce == 0 ? CULL_PRIMARY : CULL_SECONDARY,
                               ro, 0.0, rd, 10000.0, showCelestial, rayConeWidth, rayConeSpread);

        if (payload.hitT < 0.0) {
            if (bounce == 0 && captureGuides) {
                gv_normal = float3(0.0, 0.0, 0.0);
                gv_albedo = SKY_DIFF_ALBEDO;
                gv_rough = 1.0;
                gv_hitCamRel = rd * 1.0e6;
                gv_motionHitCamRel = gv_hitCamRel;
                gv_motionUseRefracted = false;
                gv_motionObjDisp = float3(0.0, 0.0, 0.0);
                gv_spec = makeSpecSurface(gv_hitCamRel, gv_normal, 1.0, SKY_SPEC_ALBEDO);
            }
            alive = true;
            return terminal;
        }

        float3 n = payload.normal;
        float3 hitPos = ro + rd * payload.hitT;
        uint material = payloadMaterial();
        if (material != MATERIAL_WATER && material != MATERIAL_DIELECTRIC) {
            if (bounce == 0 && captureGuides) {
                gv_normal = n;
                gv_hitCamRel = hitPos - worldPush.camOffset;
                gv_motionHitCamRel = gv_hitCamRel;
                gv_motionUseRefracted = false;
                gv_motionObjDisp = payload.motionPrev;
                if (material == MATERIAL_PARTICLE) {
                    gv_albedo = payload.albedo;
                    gv_rough = 1.0;
                    gv_spec = makeSpecSurface(gv_hitCamRel, n, 1.0,
                                              float3(0.0, 0.0, 0.0));
                } else {
                    float rough = clamp(payloadRoughness(), 0.0, 1.0);
                    rough = rough <= MIRROR_ALPHA_MAX ? 0.0 : rough;
                    float metal = clamp(payloadMetalness(), 0.0, 1.0);
                    float3 diffAlb = payload.albedo * (1.0 - metal);
                    float3 v = -rd;
                    gv_albedo = diffAlb;
                    gv_rough = rough;
                    gv_spec = makeSpecSurface(gv_hitCamRel, n, rough,
                                              rrSpecularAlbedo(payload.f0, rough, dot(n, v)));
                }
            }
            alive = true;
            return terminal;
        }

        // This hit is consumed in pass A, so carry its travelled-medium absorption and cone growth into
        // the record for the next trace. The terminal hit above is deliberately not consumed here.
        if (any(medium.current.extinction > 0.0)) {
            throughput *= exp(-medium.current.extinction * payload.hitT);
        }
        rayConeWidth = max(rayConeWidth + rayConeSpread * max(payload.hitT, 0.0),
                           RAY_CONE_MIN_WIDTH);

        bool isWater = material == MATERIAL_WATER;
        bool entering = payloadDielectricEntering();
        float3 geometricNormal = n;
        float3 previousNormal = n;
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

        float transmission = clamp(payloadTransmission(), 0.0, 1.0);
        Medium entered = makeDielectricMedium(isWater, max(payloadIor(), 1.0),
                                              payload.albedo, transmission);
        float etaI = medium.current.ior;
        float etaT = entering ? entered.ior : medium.outer.ior;
        float F = fresnelDielectric(clamp(dot(-rd, n), 0.0, 1.0), etaI, etaT);
        float3 transmittedDir = refract(rd, n, etaI / etaT);
        float transmitBias = isWater ? SURF_BIAS : INSET_TRANSMIT_BIAS;

        if (bounce == 0 && captureGuides) {
            gv_normal = n;
            gv_rough = 0.0;
            gv_albedo = float3(0.0, 0.0, 0.0);
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

        if (rndf(seed) < F) {
            rd = reflect(rd, n);
            ro = offsetSurfaceOrigin(hitPos, geometricNormal, rd, SURF_BIAS);
        } else {
            if (dot(transmittedDir, transmittedDir) <= 0.0) {
                return terminal;
            }
            rd = normalize(transmittedDir);
            ro = offsetSurfaceOrigin(hitPos, geometricNormal, rd, transmitBias);
            if (entering) {
                mediumPush(medium, entered);
            } else {
                mediumPop(medium);
            }
        }
        showCelestial = true;
        if (bounce >= rrStart) {
            float q = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.02, 1.0);
            if (rndf(seed) > q) {
                return terminal;
            }
            throughput /= q;
        }
    }
    return seg;
}

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
