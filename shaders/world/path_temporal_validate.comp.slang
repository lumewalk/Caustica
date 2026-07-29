// Conservative temporal admission for future ReSTIR PT reuse. This pass intentionally does not merge
// weights: a historical path may only enter GRIS after replay/reconnection can evaluate its shifted
// target at the current receiver. The pass proves reprojection and strict compatibility independently.

import world_common;
import path_reservoir_common;

[[vk::push_constant]] WorldPushConstants pc;

[[vk::binding(0, 0)]] [format("rg16f")] RWTexture2D<float2> receiverMotion;
[[vk::binding(1, 0)]] [format("r32ui")] RWTexture2D<uint> validationMetadata;
[[vk::binding(2, 0)]] [format("rgba16f")] RWTexture2D<float4> debugColor;
[[vk::binding(3, 0)]] [format("rgba32f")] RWTexture2D<float4> receiverPositionMaterial;
[[vk::binding(4, 0)]] [format("rgba16f")] RWTexture2D<float4> receiverNormalRoughness;
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> spatialDiagnosticCounters;
[[vk::binding(6, 0)]] RWStructuredBuffer<float2> spatialDiagnosticPairs;

static const uint HISTORY_VALID = 1u << 7u;
static const uint SPATIAL_DIAGNOSTIC_PAIR_MASK = 3u;

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
    // 10 replay ABI mismatch or inconsistent replay lanes, 11 canonical endpoint unavailable.
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

                    bool currentReplayConsistent = current.replayControl.x == PATH_REPLAY_VERSION
                            && current.replayControl.y >= 1u
                            && current.replayControl.y <= MAX_PATH_SEGMENTS
                            && asuint(current.proposalShift.w) == current.replayControl.z
                            && (current.replayControl.y == MAX_PATH_SEGMENTS
                                    || all(current.replay.zw == uint2(0u)))
                            && all(current.proposalComponents.xyz > float3(0.0))
                            && (asuint(current.proposalComponents.w)
                                    & PATH_PROPOSAL_RESERVED_MASK) == 0u;
                    bool previousReplayConsistent = previous.replayControl.x == PATH_REPLAY_VERSION
                            && previous.replayControl.y >= 1u
                            && previous.replayControl.y <= MAX_PATH_SEGMENTS
                            && asuint(previous.proposalShift.w) == previous.replayControl.z
                            && (previous.replayControl.y == MAX_PATH_SEGMENTS
                                    || all(previous.replay.zw == uint2(0u)))
                            && all(previous.proposalComponents.xyz > float3(0.0))
                            && (asuint(previous.proposalComponents.w)
                                    & PATH_PROPOSAL_RESERVED_MASK) == 0u;
                    if (!currentReplayConsistent || !previousReplayConsistent) {
                        debugState = 10u;
                    } else if (!pathCanonicalEndpointConsistent(current)
                            || !pathCanonicalEndpointConsistent(previous)) {
                        debugState = 11u;
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
        else if (debugState == 11u) color = float3(1.00, 0.12, 0.55); // no canonical endpoint
        else color = float3(0.05, 0.95, 0.20);                       // strict compatibility accepted
        debugColor[pixel] = float4(color, 1.0);
    }
    bool spatialPolicyDebug = pc.debugView == PATH_SPATIAL_POLICY_DEBUG_VIEW;
    if (pc.debugView == PATH_SPATIAL_DEBUG_VIEW || spatialPolicyDebug) {
        float4 receiverPosition = receiverPositionMaterial[pixel];
        float4 receiverNormal = receiverNormalRoughness[pixel];
        float receiverNormalLength = length(receiverNormal.xyz);
        uint receiverMaterial = uint(max(receiverPosition.w, 0.0));
        bool receiverPresent = receiverNormalLength >= 0.5
                && receiverMaterial <= MATERIAL_PARTICLE
                && current.weights.y > 0.0
                && current.sampleRadianceTarget.w > 0.0;
        uint compatibleCount = 0u;
        uint validCount = 0u;
        uint reconnectionCount = 0u;
        uint limitedReconnectionCount = 0u;
        bool depthRejected = false;
        bool topologyRejected = false;
        bool transportRejected = false;
        bool footprintRejected = false;
        bool topologyRescued = false;
        bool diagnosticPairValid = false;
        float2 diagnosticPair = float2(0.0);
        bool rescuedDiagnosticPairValid = false;
        float2 rescuedDiagnosticPair = float2(0.0);
        if (receiverPresent) {
            float3 receiverNormalUnit = receiverNormal.xyz / receiverNormalLength;
            float receiverDistance = length(receiverPosition.xyz);
            uint seed = pathHash(pixel.x * 2246822519u
                    ^ pixel.y * 3266489917u ^ pc.frameIndex * 668265263u);
            float rotation = float(seed) * (6.28318530718 / 4294967296.0);
            for (uint sample = 0u; sample < PATH_SPATIAL_SAMPLE_COUNT; ++sample) {
                float radiusFraction = sqrt((float(sample) + 0.5)
                        / float(PATH_SPATIAL_SAMPLE_COUNT));
                float angle = rotation + float(sample) * 2.39996322973;
                float radius = 2.0 + radiusFraction * (PATH_SPATIAL_RADIUS - 2.0);
                int2 offset = int2(round(float2(cos(angle), sin(angle)) * radius));
                int2 neighborTexel = int2(pixel) + offset;
                if (any(neighborTexel < int2(0, 0))
                        || any(neighborTexel >= int2(width, height))) continue;
                float4 neighborPosition = receiverPositionMaterial[neighborTexel];
                float4 neighborNormal = receiverNormalRoughness[neighborTexel];
                float neighborNormalLength = length(neighborNormal.xyz);
                uint neighborMaterial = uint(max(neighborPosition.w, 0.0));
                if (neighborNormalLength < 0.5 || neighborMaterial != receiverMaterial) continue;
                float3 neighborNormalUnit = neighborNormal.xyz / neighborNormalLength;
                float relativeDepth = abs(length(neighborPosition.xyz) - receiverDistance)
                        / max(receiverDistance, 1.0);
                if (dot(receiverNormalUnit, neighborNormalUnit) < PATH_SPATIAL_NORMAL_COS
                        || relativeDepth > PATH_SPATIAL_RELATIVE_DEPTH) continue;
                compatibleCount++;
                uint neighborIndex = uint(neighborTexel.y) * width + uint(neighborTexel.x);
                PathReservoir source =
                        ConstPtr<PathReservoir>(pc.pathReservoirAddr)[neighborIndex];
                if (source.weights.y <= 0.0 || source.sampleRadianceTarget.w <= 0.0) continue;
                validCount++;
                uint sourceMetadata = source.identity.w;
                if (pathMetadataDepth(sourceMetadata)
                        != pathMetadataDepth(current.identity.w)) {
                    depthRejected = true;
                    continue;
                }
                bool topologyMismatch = source.identity.z != current.identity.z;
                if (topologyMismatch) {
                    topologyRejected = true;
                    if (!spatialPolicyDebug) continue;
                }
                if (pathMetadataTransport(sourceMetadata)
                        != pathMetadataTransport(current.identity.w)) {
                    transportRejected = true;
                    continue;
                }
                float sourceFootprint = max(source.proposalShift.z, 1.0e-5);
                float receiverFootprint = max(current.proposalShift.z, 1.0e-5);
                float footprintRatio = max(sourceFootprint, receiverFootprint)
                        / min(sourceFootprint, receiverFootprint);
                if (footprintRatio > PATH_MAX_FOOTPRINT_RATIO) {
                    // Preserve the strict baseline category: under exact topology this is a
                    // footprint reject; a relaxed-topology candidate remains a topology reject
                    // unless it passes every later limited-policy condition.
                    if (!topologyMismatch) footprintRejected = true;
                    continue;
                }
                uint reconnectionMetadata = asuint(source.reconnectionVertex.w);
                bool reconnectionValid = (reconnectionMetadata & PATH_RECONNECTION_VALID) != 0u;
                if (reconnectionValid && ((reconnectionMetadata >> 1u) & 0xFu) == 1u) {
                    limitedReconnectionCount++;
                    if (topologyMismatch) {
                        topologyRescued = true;
                        if (!rescuedDiagnosticPairValid
                                && current.canonicalRadianceTarget.w > 0.0
                                && source.canonicalRadianceTarget.w > 0.0) {
                            rescuedDiagnosticPair = float2(current.canonicalRadianceTarget.w,
                                    source.canonicalRadianceTarget.w);
                            rescuedDiagnosticPairValid = true;
                        }
                    } else {
                        reconnectionCount++;
                        if (!diagnosticPairValid
                                && current.canonicalRadianceTarget.w > 0.0
                                && source.canonicalRadianceTarget.w > 0.0) {
                            diagnosticPair = float2(current.canonicalRadianceTarget.w,
                                    source.canonicalRadianceTarget.w);
                            diagnosticPairValid = true;
                        }
                    }
                }
            }
        }
        float3 spatialColor;
        if (!receiverPresent) spatialColor = float3(0.0);
        else if (reconnectionCount > 0u) spatialColor = float3(0.05, 0.95, 0.20);
        else if (spatialPolicyDebug && topologyRescued) spatialColor = float3(0.98, 0.82, 0.05);
        else if (footprintRejected) spatialColor = float3(0.55, 0.10, 0.95);
        else if (depthRejected) spatialColor = float3(0.85, 0.05, 0.80);
        else if (topologyRejected) spatialColor = float3(0.15, 0.35, 0.95);
        else if (transportRejected) spatialColor = float3(0.95, 0.72, 0.04);
        else if (validCount > 0u) spatialColor = float3(0.10, 0.10, 0.10);
        else if (compatibleCount > 0u) spatialColor = float3(0.95, 0.30, 0.04);
        else spatialColor = float3(0.90, 0.06, 0.05);
        uint category = !receiverPresent ? 0u
                : reconnectionCount > 0u ? 1u
                : footprintRejected ? 2u
                : depthRejected ? 3u
                : topologyRejected ? 4u
                : transportRejected ? 5u
                : validCount > 0u ? 6u
                : compatibleCount > 0u ? 7u
                : 8u;
        InterlockedAdd(spatialDiagnosticCounters[category], 1u);
        if (spatialPolicyDebug) {
            if (limitedReconnectionCount > 0u) {
                InterlockedAdd(
                        spatialDiagnosticCounters[PATH_SPATIAL_DIAGNOSTIC_LIMITED_ADMITTED], 1u);
            }
            if (topologyRescued && reconnectionCount == 0u) {
                InterlockedAdd(
                        spatialDiagnosticCounters[PATH_SPATIAL_DIAGNOSTIC_TOPOLOGY_RESCUED], 1u);
            }
        }
        // Keep readback bounded: hash-subsample roughly one quarter of admitted pixels and retain
        // raw canonical targets only. These are correlation diagnostics, not shifted GRIS targets.
        uint pairHash = pathHash(pixelIndex ^ pc.frameIndex * 668265263u);
        if (category == 1u && diagnosticPairValid
                && (pairHash & SPATIAL_DIAGNOSTIC_PAIR_MASK) == 0u) {
            uint pairIndex;
            InterlockedAdd(
                    spatialDiagnosticCounters[PATH_SPATIAL_DIAGNOSTIC_STRICT_PAIR_CURSOR],
                    1u, pairIndex);
            if (pairIndex < PATH_SPATIAL_DIAGNOSTIC_PAIR_CAPACITY) {
                spatialDiagnosticPairs[pairIndex] = diagnosticPair;
            }
        }
        if (spatialPolicyDebug && rescuedDiagnosticPairValid
                && (pairHash & SPATIAL_DIAGNOSTIC_PAIR_MASK) == 0u) {
            uint pairIndex;
            InterlockedAdd(
                    spatialDiagnosticCounters[PATH_SPATIAL_DIAGNOSTIC_RESCUED_PAIR_CURSOR],
                    1u, pairIndex);
            if (pairIndex < PATH_SPATIAL_DIAGNOSTIC_PAIR_CAPACITY) {
                spatialDiagnosticPairs[PATH_SPATIAL_DIAGNOSTIC_PAIR_CAPACITY + pairIndex] =
                        rescuedDiagnosticPair;
            }
        }
        debugColor[pixel] = float4(spatialColor, 1.0);
    }
}
