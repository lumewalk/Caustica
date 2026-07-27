// Conservative temporal reuse for persistent ReSTIR DI reservoirs. This pass runs after same-frame
// candidates and merges at most one reprojected previous reservoir. It remains diagnostic-only: the
// finalized reservoir is not consumed by final lighting yet.

import world_common;
import sampling_core;
import direct_reservoir_common;

[[vk::push_constant]] WorldPushConstants pc;

[[vk::binding(0, 0)]] [format("rgba32f")] RWTexture2D<float4> receiverPositionMaterial;
[[vk::binding(1, 0)]] [format("rgba16f")] RWTexture2D<float4> receiverNormalRoughness;
[[vk::binding(2, 0)]] [format("rgba16f")] RWTexture2D<float4> receiverAlbedoSss;
[[vk::binding(3, 0)]] [format("rg16f")] RWTexture2D<float2> receiverMotion;
[[vk::binding(4, 0)]] [format("r32ui")] RWTexture2D<uint> validationMetadata;
[[vk::binding(5, 0)]] [format("rgba16f")] RWTexture2D<float4> debugColor;
[[vk::binding(6, 0)]] RWStructuredBuffer<DirectReservoir, Std430DataLayout> currentReservoirs;
[[vk::binding(7, 0)]] StructuredBuffer<DirectReservoir, Std430DataLayout> previousReservoirs;

static const uint HISTORY_VALID = 1u << 7u;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID) {
    uint width;
    uint height;
    receiverPositionMaterial.GetDimensions(width, height);
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= width || pixel.y >= height) return;

    uint pixelIndex = pixel.y * width + pixel.x;
    float4 positionMaterial = receiverPositionMaterial[pixel];
    float4 normalRoughness = receiverNormalRoughness[pixel];
    float4 albedoSss = receiverAlbedoSss[pixel];
    float normalLength = length(normalRoughness.xyz);
    uint material = uint(max(positionMaterial.w, 0.0));
    bool receiverPresent = normalLength >= 0.5
            && material <= MATERIAL_PARTICLE
            && max(albedoSss.x, max(albedoSss.y, albedoSss.z)) > 0.0;

    // Diagnostic state: 0 no receiver, 1 no history, 2 validation reject, 3 empty history,
    // 4 epoch reject, 5 age reject, 6 zero target, 7 merged/current selected, 8 merged/history selected.
    uint debugState = receiverPresent ? 1u : 0u;
    DirectReservoir current = currentReservoirs[pixelIndex];
    WorldPush push = ConstPtr<WorldPush>(pc.worldPushAddr)[0];

    if (receiverPresent && (pc.historyFlags & 1u) != 0u) {
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
                DirectReservoir previous = previousReservoirs[previousIndex];
                uint previousFlags = reservoirFlags(previous);
                if ((previousFlags & RESERVOIR_FLAG_VALID) == 0u
                        || previous.weights.y <= 0.0 || previous.weights.z <= 0.0) {
                    debugState = 3u;
                } else if (previous.metadata.y != push.lightEpoch) {
                    debugState = 4u;
                } else if (reservoirAge(previous) >= RESERVOIR_MAX_AGE) {
                    debugState = 5u;
                } else {
                    float3 hitPosition = positionMaterial.xyz;
                    float3 normal = normalRoughness.xyz / normalLength;
                    float2 size = float2(width, height);
                    float2 uv = (float2(pixel) + 0.5) / size;
                    float2 jitteredNdc = (uv + push.jitter / size) * 2.0 - 1.0;
                    float4 nearH = mul(push.invViewProj, float4(jitteredNdc, 1.0, 1.0));
                    float3 rayOrigin = nearH.xyz / nearH.w + push.camOffset;
                    float3 rayDirection = normalize(hitPosition - rayOrigin);
                    bool twoSided = material == MATERIAL_PARTICLE;
                    float sss = twoSided ? 0.0 : max(0.0, albedoSss.w);
                    float3 samplePosition = previous.samplePositionArea.xyz + push.lightRebase.xyz;
                    float currentTarget = directCandidateTarget(
                            samplePosition,
                            previous.lightNormalTargetPdf.xyz,
                            previous.radianceProposalPdf.xyz,
                            previous.samplePositionArea.w,
                            hitPosition, normal, rayDirection, albedoSss.xyz, twoSided, sss);
                    if (currentTarget <= 0.0) {
                        debugState = 6u;
                    } else {
                        float currentM = max(0.0, current.weights.y);
                        float historyM = min(previous.weights.y,
                                max(1.0, currentM) * RESERVOIR_HISTORY_M_MULTIPLIER);
                        float historyWeight = currentTarget * previous.weights.z * historyM;
                        float currentWeight = max(0.0, current.weights.x);
                        float totalWeight = currentWeight + historyWeight;
                        float totalM = currentM + historyM;
                        uint seed = (pixel.x * 1597334677u + pixel.y * 3812015801u)
                                ^ (push.frameIndex * 2798796415u) ^ 0xa511e9b3u;
                        seed = pcg(seed);
                        bool selectHistory = historyWeight > 0.0
                                && (currentWeight <= 0.0 || rndf(seed) * totalWeight < historyWeight);
                        uint mergedFlags;
                        if (selectHistory) {
                            current.samplePositionArea = previous.samplePositionArea;
                            current.lightNormalTargetPdf = float4(
                                    previous.lightNormalTargetPdf.xyz, currentTarget);
                            current.radianceProposalPdf = previous.radianceProposalPdf;
                            mergedFlags = previousFlags | RESERVOIR_FLAG_VALID
                                    | RESERVOIR_FLAG_TEMPORAL_MERGED
                                    | RESERVOIR_FLAG_TEMPORAL_SELECTED;
                            uint age = min(reservoirAge(previous) + 1u, 0xffffu);
                            current.metadata = uint4(previous.metadata.x, previous.metadata.y,
                                    previous.metadata.z, age | (mergedFlags << 16u));
                            debugState = 8u;
                        } else {
                            mergedFlags = reservoirFlags(current) | RESERVOIR_FLAG_TEMPORAL_MERGED;
                            current.metadata.w = (current.metadata.w & 0xffffu) | (mergedFlags << 16u);
                            debugState = 7u;
                        }
                        float selectedTarget = current.lightNormalTargetPdf.w;
                        current.weights = float4(totalWeight, totalM,
                                selectedTarget > 0.0 && totalM > 0.0
                                        ? totalWeight / (totalM * selectedTarget) : 0.0,
                                totalM);
                        currentReservoirs[pixelIndex] = current;
                    }
                }
            }
        }
    }

    if (pc.debugView == 10u) {
        float3 color;
        if (debugState == 0u) color = float3(0.0, 0.0, 0.0);
        else if (debugState == 1u) color = float3(0.03, 0.08, 0.20); // first frame / reset
        else if (debugState == 2u) color = float3(0.90, 0.06, 0.05); // surface validation reject
        else if (debugState == 3u) color = float3(0.10, 0.10, 0.10); // previous reservoir empty
        else if (debugState == 4u) color = float3(0.95, 0.72, 0.04); // light hierarchy changed
        else if (debugState == 5u) color = float3(0.85, 0.05, 0.80); // history aged out
        else if (debugState == 6u) color = float3(0.95, 0.30, 0.04); // old sample has zero current target
        else if (debugState == 7u) color = float3(0.05, 0.75, 0.95); // merged, current won
        else color = float3(0.05, 0.95, 0.20);                       // merged, history won
        debugColor[pixel] = float4(color, 1.0);
    }
}
