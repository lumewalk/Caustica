// Same-frame spatial reuse for persistent ReSTIR DI reservoirs. The temporal result is immutable
// input while this pass writes the other slot, avoiding read/write races without a third reservoir
// buffer. Final lighting remains unchanged until the reservoir estimator is validated end to end.

import world_common;
import sampling_core;
import direct_reservoir_common;

[[vk::push_constant]] WorldPushConstants pc;

[[vk::binding(0, 0)]] [format("rgba32f")] RWTexture2D<float4> receiverPositionMaterial;
[[vk::binding(1, 0)]] [format("rgba16f")] RWTexture2D<float4> receiverNormalRoughness;
[[vk::binding(2, 0)]] [format("rgba16f")] RWTexture2D<float4> receiverAlbedoSss;
[[vk::binding(3, 0)]] [format("rgba16f")] RWTexture2D<float4> debugColor;
[[vk::binding(4, 0)]] StructuredBuffer<DirectReservoir, Std430DataLayout> inputReservoirs;
[[vk::binding(5, 0)]] RWStructuredBuffer<DirectReservoir, Std430DataLayout> outputReservoirs;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID) {
    uint width;
    uint height;
    receiverPositionMaterial.GetDimensions(width, height);
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= width || pixel.y >= height) return;

    uint pixelIndex = pixel.y * width + pixel.x;
    DirectReservoir result = inputReservoirs[pixelIndex];
    float4 positionMaterial = receiverPositionMaterial[pixel];
    float4 normalRoughness = receiverNormalRoughness[pixel];
    float4 albedoSss = receiverAlbedoSss[pixel];
    float normalLength = length(normalRoughness.xyz);
    uint material = uint(max(positionMaterial.w, 0.0));
    bool receiverPresent = normalLength >= 0.5
            && material <= MATERIAL_PARTICLE
            && max(albedoSss.x, max(albedoSss.y, albedoSss.z)) > 0.0;

    uint compatibleCount = 0u;
    uint validCount = 0u;
    uint positiveTargetCount = 0u;
    bool selectedNeighbor = false;
    bool epochRejected = false;
    if (receiverPresent) {
        WorldPush push = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
        float3 hitPosition = positionMaterial.xyz;
        float3 normal = normalRoughness.xyz / normalLength;
        float2 size = float2(width, height);
        float2 uv = (float2(pixel) + 0.5) / size;
        float2 jitteredNdc = (uv + push.jitter / size) * 2.0 - 1.0;
        float4 nearH = mul(push.invViewProj, float4(jitteredNdc, 1.0, 1.0));
        float3 rayOrigin = nearH.xyz / nearH.w + push.camOffset;
        float3 rayDirection = normalize(hitPosition - rayOrigin);
        float receiverDistance = length(hitPosition - rayOrigin);
        bool twoSided = material == MATERIAL_PARTICLE;
        float sss = twoSided ? 0.0 : max(0.0, albedoSss.w);
        float baseM = max(1.0, result.weights.y);
        float weightSum = max(0.0, result.weights.x);
        float effectiveM = max(0.0, result.weights.y);

        uint seed = (pixel.x * 2246822519u + pixel.y * 3266489917u)
                ^ (push.frameIndex * 668265263u) ^ 0xb5297a4du;
        seed = pcg(seed);
        float rotation = rndf(seed) * 6.28318530718;
        for (uint sample = 0u; sample < RESERVOIR_SPATIAL_SAMPLE_COUNT; ++sample) {
            float radiusFraction = sqrt((float(sample) + 0.5)
                    / float(RESERVOIR_SPATIAL_SAMPLE_COUNT));
            float angle = rotation + float(sample) * 2.39996322973;
            float radius = 2.0 + radiusFraction * (RESERVOIR_SPATIAL_RADIUS - 2.0);
            int2 offset = int2(round(float2(cos(angle), sin(angle)) * radius));
            int2 neighborTexel = int2(pixel) + offset;
            if (any(neighborTexel < int2(0, 0))
                    || any(neighborTexel >= int2(width, height))) continue;

            float4 neighborPositionMaterial = receiverPositionMaterial[neighborTexel];
            float4 neighborNormalRoughness = receiverNormalRoughness[neighborTexel];
            float neighborNormalLength = length(neighborNormalRoughness.xyz);
            uint neighborMaterial = uint(max(neighborPositionMaterial.w, 0.0));
            if (neighborNormalLength < 0.5 || neighborMaterial != material) continue;
            float3 neighborNormal = neighborNormalRoughness.xyz / neighborNormalLength;
            float neighborDistance = length(neighborPositionMaterial.xyz - rayOrigin);
            float relativeDepth = abs(neighborDistance - receiverDistance)
                    / max(receiverDistance, 1.0);
            if (dot(normal, neighborNormal) < RESERVOIR_SPATIAL_NORMAL_COS
                    || relativeDepth > RESERVOIR_SPATIAL_RELATIVE_DEPTH) continue;
            compatibleCount++;

            uint neighborIndex = uint(neighborTexel.y) * width + uint(neighborTexel.x);
            DirectReservoir source = inputReservoirs[neighborIndex];
            uint sourceFlags = reservoirFlags(source);
            if ((sourceFlags & RESERVOIR_FLAG_VALID) == 0u
                    || source.weights.y <= 0.0 || source.weights.z <= 0.0) continue;
            if (source.metadata.y != push.lightEpoch) {
                epochRejected = true;
                continue;
            }
            validCount++;

            float3 samplePosition = source.samplePositionArea.xyz + push.lightRebase.xyz;
            float currentTarget = directCandidateTarget(
                    samplePosition,
                    source.lightNormalTargetPdf.xyz,
                    source.radianceProposalPdf.xyz,
                    source.samplePositionArea.w,
                    hitPosition, normal, rayDirection, albedoSss.xyz, twoSided, sss);
            if (currentTarget <= 0.0) continue;
            positiveTargetCount++;

            // Each neighbor contributes at most one current-reservoir equivalent. This bounds spatial
            // correlation while retaining the source reservoir's normalized estimator weight.
            float sourceM = min(source.weights.y, baseM);
            float sourceWeight = currentTarget * source.weights.z * sourceM;
            float nextWeightSum = weightSum + sourceWeight;
            effectiveM += sourceM;
            seed ^= sample * 747796405u + 2891336453u;
            seed = pcg(seed);
            bool selectSource = sourceWeight > 0.0
                    && (weightSum <= 0.0 || rndf(seed) * nextWeightSum < sourceWeight);
            weightSum = nextWeightSum;
            if (selectSource) {
                result.samplePositionArea = source.samplePositionArea;
                result.lightNormalTargetPdf = float4(source.lightNormalTargetPdf.xyz, currentTarget);
                result.radianceProposalPdf = source.radianceProposalPdf;
                uint receiverFlags = RESERVOIR_FLAG_VALID | RESERVOIR_FLAG_DIFFUSE_TARGET
                        | (sss > 0.0 ? RESERVOIR_FLAG_SSS_TARGET : 0u);
                uint inheritedFlags = sourceFlags
                        & (RESERVOIR_FLAG_TEMPORAL_MERGED | RESERVOIR_FLAG_TEMPORAL_SELECTED);
                uint mergedFlags = receiverFlags | inheritedFlags
                        | RESERVOIR_FLAG_SPATIAL_MERGED | RESERVOIR_FLAG_SPATIAL_SELECTED;
                result.metadata = uint4(source.metadata.x, source.metadata.y, source.metadata.z,
                        reservoirAge(source) | (mergedFlags << 16u));
                selectedNeighbor = true;
            }
        }

        uint resultFlags = reservoirFlags(result);
        if (positiveTargetCount > 0u) {
            resultFlags |= RESERVOIR_FLAG_SPATIAL_MERGED;
            result.metadata.w = reservoirAge(result) | (resultFlags << 16u);
            float selectedTarget = result.lightNormalTargetPdf.w;
            result.weights = float4(weightSum, effectiveM,
                    selectedTarget > 0.0 && effectiveM > 0.0
                            ? weightSum / (effectiveM * selectedTarget) : 0.0,
                    effectiveM);
        }
    }

    outputReservoirs[pixelIndex] = result;
    if (pc.debugView == 11u) {
        float3 color;
        if (!receiverPresent) color = float3(0.0, 0.0, 0.0);
        else if (selectedNeighbor) color = float3(0.05, 0.95, 0.20); // neighbor selected
        else if (positiveTargetCount > 0u) color = float3(0.05, 0.75, 0.95); // current retained
        else if (epochRejected) color = float3(0.95, 0.72, 0.04); // light hierarchy changed
        else if (validCount > 0u) color = float3(0.95, 0.30, 0.04); // zero target at receiver
        else if (compatibleCount > 0u) color = float3(0.10, 0.10, 0.10); // empty neighbors
        else color = float3(0.90, 0.06, 0.05); // surface compatibility rejection
        debugColor[pixel] = float4(color, 1.0);
    }
}
