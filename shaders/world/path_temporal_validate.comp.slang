// Conservative temporal admission for future ReSTIR PT reuse. This pass intentionally does not merge
// weights: a historical path may only enter GRIS after replay/reconnection can evaluate its shifted
// target at the current receiver. The pass proves reprojection and strict compatibility independently.

import world_common;
import path_reservoir_common;

[[vk::push_constant]] WorldPushConstants pc;

[[vk::binding(0, 0)]] [format("rg16f")] RWTexture2D<float2> receiverMotion;
[[vk::binding(1, 0)]] [format("r32ui")] RWTexture2D<uint> validationMetadata;
[[vk::binding(2, 0)]] [format("rgba16f")] RWTexture2D<float4> debugColor;

static const uint HISTORY_VALID = 1u << 7u;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID) {
    uint width;
    uint height;
    receiverMotion.GetDimensions(width, height);
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= width || pixel.y >= height) return;

    uint pixelIndex = pixel.y * width + pixel.x;
    PathReservoir current = ConstPtr<PathReservoir>(pc.pathReservoirAddr)[pixelIndex];

    // 0 current empty, 1 first frame/reset, 2 surface reprojection reject, 3 previous empty,
    // 4 generation mismatch, 5 depth mismatch, 6 transport mismatch, 7 topology mismatch,
    // 8 footprint mismatch, 9 strictly compatible and ready for future replay/shift evaluation,
    // 10 replay ABI mismatch.
    uint debugState = current.weights.y > 0.0 && current.sampleRadianceTarget.w > 0.0 ? 1u : 0u;
    bool historyAvailable = (pc.historyFlags & 8u) != 0u
            && pc.pathReservoirPreviousAddr != 0u;
    if (debugState != 0u && historyAvailable) {
        if ((validationMetadata[pixel] & HISTORY_VALID) == 0u) {
            debugState = 2u;
        } else {
            float2 previousPixel = float2(pixel) + 0.5 + receiverMotion[pixel];
            int2 previousTexel = int2(floor(previousPixel));
            bool inBounds = all(previousTexel >= int2(0, 0))
                    && all(previousTexel < int2(width, height));
            if (!inBounds) {
                debugState = 2u;
            } else {
                uint previousIndex = uint(previousTexel.y) * width + uint(previousTexel.x);
                PathReservoir previous =
                        ConstPtr<PathReservoir>(pc.pathReservoirPreviousAddr)[previousIndex];
                if (previous.weights.y <= 0.0 || previous.weights.z <= 0.0
                        || previous.sampleRadianceTarget.w <= 0.0) {
                    debugState = 3u;
                } else {
                    uint currentMetadata = current.identity.w;
                    uint previousMetadata = previous.identity.w;
                    uint expectedGeneration = pc.historyGeneration & PATH_GENERATION_MASK;
                    uint currentGeneration = currentMetadata & PATH_GENERATION_MASK;
                    uint previousGeneration = previousMetadata & PATH_GENERATION_MASK;
                    uint currentDepth = (currentMetadata >> 24u) & 0xFu;
                    uint previousDepth = (previousMetadata >> 24u) & 0xFu;
                    uint currentTransport = (currentMetadata >> 28u) & 0xFu;
                    uint previousTransport = (previousMetadata >> 28u) & 0xFu;
                    float currentFootprint = max(current.proposalShift.z, 1.0e-5);
                    float previousFootprint = max(previous.proposalShift.z, 1.0e-5);
                    float footprintRatio = max(currentFootprint, previousFootprint)
                            / min(currentFootprint, previousFootprint);

                    if (current.replay.z != PATH_REPLAY_VERSION
                            || previous.replay.z != PATH_REPLAY_VERSION) {
                        debugState = 10u;
                    } else if (currentGeneration != expectedGeneration
                            || previousGeneration != expectedGeneration) {
                        debugState = 4u;
                    } else if (currentDepth != previousDepth) {
                        debugState = 5u;
                    } else if (currentTransport != previousTransport) {
                        debugState = 6u;
                    } else if (current.identity.z != previous.identity.z) {
                        debugState = 7u;
                    } else if (footprintRatio > PATH_MAX_FOOTPRINT_RATIO) {
                        debugState = 8u;
                    } else {
                        debugState = 9u;
                    }
                }
            }
        }
    }

    if (pc.debugView == PATH_TEMPORAL_DEBUG_VIEW) {
        float3 color;
        if (debugState == 0u) color = float3(0.0, 0.0, 0.0);
        else if (debugState == 1u) color = float3(0.03, 0.08, 0.20); // first frame / reset
        else if (debugState == 2u) color = float3(0.90, 0.06, 0.05); // surface validation reject
        else if (debugState == 3u) color = float3(0.10, 0.10, 0.10); // previous reservoir empty
        else if (debugState == 4u) color = float3(0.95, 0.72, 0.04); // generation mismatch
        else if (debugState == 5u) color = float3(0.85, 0.05, 0.80); // depth mismatch
        else if (debugState == 6u) color = float3(0.95, 0.30, 0.04); // transport mismatch
        else if (debugState == 7u) color = float3(0.15, 0.35, 0.95); // topology mismatch
        else if (debugState == 8u) color = float3(0.55, 0.10, 0.95); // footprint mismatch
        else if (debugState == 10u) color = float3(0.05, 0.85, 0.95); // replay ABI mismatch
        else color = float3(0.05, 0.95, 0.20);                       // strict compatibility accepted
        debugColor[pixel] = float4(color, 1.0);
    }
}
