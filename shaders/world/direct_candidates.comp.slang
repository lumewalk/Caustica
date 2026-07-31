// Initial ReSTIR DI candidate generation. This pass fills the current persistent reservoir but does
// not contribute to final lighting yet. The target is deliberately diffuse + thin-surface SSS only;
// specular reuse will be enabled after a raw-F0 receiver guide exists.

import world_common;
import sampling_core;
import light_sampling;
import direct_reservoir_common;

[[vk::push_constant]] WorldPushConstants pc;

[[vk::binding(0, 0)]] [format("rgba32f")] RWTexture2D<float4> receiverPositionMaterial;
[[vk::binding(1, 0)]] [format("rgba16f")] RWTexture2D<float4> receiverNormalRoughness;
[[vk::binding(2, 0)]] [format("rgba16f")] RWTexture2D<float4> receiverAlbedoSss;
[[vk::binding(3, 0)]] [format("rgba16f")] RWTexture2D<float4> debugColor;
[[vk::binding(4, 0)]] RWStructuredBuffer<DirectReservoir, Std430DataLayout> reservoirs;

float3 hashLightColor(uint lightId) {
    uint h = lightId * 747796405u + 2891336453u;
    h ^= h >> 16u;
    float3 color = float3(float(h & 255u), float((h >> 8u) & 255u),
            float((h >> 16u) & 255u)) / 255.0;
    return 0.18 + 0.82 * color;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID) {
    uint width;
    uint height;
    receiverPositionMaterial.GetDimensions(width, height);
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= width || pixel.y >= height) return;

    uint pixelIndex = pixel.y * width + pixel.x;
    DirectReservoir reservoir = {};
    float4 positionMaterial = receiverPositionMaterial[pixel];
    float4 normalRoughness = receiverNormalRoughness[pixel];
    float4 albedoSss = receiverAlbedoSss[pixel];
    uint material = receiverMaterialModel(uint(max(positionMaterial.w, 0.0)));
    float normalLength = length(normalRoughness.xyz);

    WorldPush push = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    bool canSample = normalLength >= 0.5
            && material <= MATERIAL_PARTICLE
            && pc.lightBufAddr != 0
            && push.lightCount > 0u
            && push.risCandidates > 0u
            && max(albedoSss.x, max(albedoSss.y, albedoSss.z)) > 0.0;

    if (canSample) {
        float3 hitPos = positionMaterial.xyz;
        float3 n = normalRoughness.xyz / normalLength;
        float2 size = float2(width, height);
        float2 uv = (float2(pixel) + 0.5) / size;
        float2 jitteredNdc = (uv + push.jitter / size) * 2.0 - 1.0;
        float4 nearH = mul(push.invViewProj, float4(jitteredNdc, 1.0, 1.0));
        float3 rayOrigin = nearH.xyz / nearH.w + push.camOffset;
        float3 rayDirection = normalize(hitPos - rayOrigin);
        bool twoSided = material == MATERIAL_PARTICLE;
        float sss = twoSided ? 0.0 : max(0.0, albedoSss.w);

        uint seed = (pixel.x * 1973u + pixel.y * 9277u + 26699u)
                ^ (push.frameIndex * 2654435761u) ^ 0x68bc21ebu;
        uint proposalSeed = seed ^ 0x9e3779b9u;
        proposalSeed = pcg(proposalSeed);
        seed = pcg(seed);
        LightProposalState proposal = prepareLightProposal(pc, push, hitPos, proposalSeed);
        float weightSum = 0.0;
        float selectedTarget = 0.0;
        uint selectedLight = 0u;

        for (uint candidate = 0u; candidate < proposal.candidateCount; ++candidate) {
            uint lightIndex = selectLightProposal(pc, push, proposal, candidate, proposalSeed);
            Light light = ConstPtr<Light>(pc.lightBufAddr)[lightIndex];
            float s = rndf(seed) * 2.0 - 1.0;
            float t = rndf(seed) * 2.0 - 1.0;
            float3 hierarchySamplePosition = light.pos
                    + s * lightHalfU(light) + t * lightHalfV(light);
            float3 samplePosition = hierarchySamplePosition + push.lightRebase.xyz;
            float3 radiance = lightRadiance(light);
            float3 lightNormal = lightGeometricNormal(light);
            float area = lightArea(light);
            float target = directCandidateTarget(samplePosition, lightNormal, radiance, area,
                    hitPos, n, rayDirection, albedoSss.xyz, twoSided, sss);
            if (target <= 0.0) continue;
            float sourcePdf = proposalPdf(pc, push, light, radiance,
                    proposal.cell, proposal.cellCoord, proposal.localProbability);
            float weight = target / max(sourcePdf, 1.0e-20);
            weightSum += weight;
            if (rndf(seed) * weightSum < weight) {
                selectedLight = lightIndex;
                selectedTarget = target;
                reservoir.samplePositionArea = float4(hierarchySamplePosition, area);
                reservoir.lightNormalTargetPdf = float4(lightNormal, target);
                reservoir.radianceProposalPdf = float4(radiance, sourcePdf);
            }
        }

        if (selectedTarget > 0.0) {
            float candidateCount = float(proposal.candidateCount);
            float finalWeight = weightSum / (candidateCount * selectedTarget);
            reservoir.weights = float4(weightSum, candidateCount, finalWeight, candidateCount);
            uint flags = RESERVOIR_FLAG_VALID | RESERVOIR_FLAG_DIFFUSE_TARGET
                    | (sss > 0.0 ? RESERVOIR_FLAG_SSS_TARGET : 0u);
            reservoir.metadata = uint4(selectedLight, push.lightEpoch, push.frameIndex, flags << 16u);
        }
    }

    reservoirs[pixelIndex] = reservoir;
    if (pc.debugView == 9u) {
        float3 debug = reservoir.lightNormalTargetPdf.w > 0.0
                ? hashLightColor(reservoir.metadata.x)
                    * clamp(0.25 + 0.18 * log2(1.0 + reservoir.weights.x), 0.25, 1.0)
                : float3(0.0, 0.0, 0.0);
        debugColor[pixel] = float4(debug, 1.0);
    }
}
