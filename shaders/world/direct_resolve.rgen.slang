// Diagnostic end-to-end ReSTIR DI resolve. It reads the finalized spatial reservoir, re-evaluates its
// diffuse/SSS contribution at the current first interface, and traces one visibility ray through the
// production shadow SBT. It deliberately writes only debug view 12; normal lighting stays unchanged.

import world_common;
import world_core;
import trace;
import direct_reservoir_common;

[shader("raygeneration")]
void main() {
    if (pc.debugView != 12u) return;

    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 dimensions = DispatchRaysDimensions().xy;
    uint pixelIndex = pixel.y * dimensions.x + pixel.x;
    int2 texel = int2(pixel);

    float4 positionMaterial = gRestirPositionMaterial[texel];
    float4 normalRoughness = gRestirNormalRoughness[texel];
    float4 albedoSss = gRestirAlbedoSss[texel];
    float normalLength = length(normalRoughness.xyz);
    uint material = uint(max(positionMaterial.w, 0.0));
    DirectReservoir reservoir = ConstPtr<DirectReservoir>(pc.directReservoirAddr)[pixelIndex];
    uint flags = reservoirFlags(reservoir);

    bool receiverPresent = normalLength >= 0.5
            && material <= MATERIAL_PARTICLE
            && max(albedoSss.x, max(albedoSss.y, albedoSss.z)) > 0.0;
    bool reservoirValid = (flags & RESERVOIR_FLAG_VALID) != 0u
            && reservoir.metadata.y == worldPush.lightEpoch
            && reservoir.weights.y > 0.0
            && reservoir.weights.z > 0.0;
    if (!receiverPresent || !reservoirValid) {
        outImage[texel] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    float3 hitPosition = positionMaterial.xyz;
    float3 normal = normalRoughness.xyz / normalLength;
    float2 size = float2(dimensions);
    float2 uv = (float2(pixel) + 0.5) / size;
    float2 jitteredNdc = (uv + worldPush.jitter / size) * 2.0 - 1.0;
    float4 nearH = mul(worldPush.invViewProj, float4(jitteredNdc, 1.0, 1.0));
    float3 rayOrigin = nearH.xyz / nearH.w + worldPush.camOffset;
    float3 rayDirection = normalize(hitPosition - rayOrigin);
    bool twoSided = material == MATERIAL_PARTICLE;
    float sss = twoSided ? 0.0 : max(0.0, albedoSss.w);
    float3 samplePosition = reservoir.samplePositionArea.xyz + worldPush.lightRebase.xyz;

    float3 contribution = directCandidateContribution(
            samplePosition,
            reservoir.lightNormalTargetPdf.xyz,
            reservoir.radianceProposalPdf.xyz,
            reservoir.samplePositionArea.w,
            hitPosition, normal, rayDirection, albedoSss.xyz, twoSided, sss);
    if (max(contribution.x, max(contribution.y, contribution.z)) <= 0.0) {
        outImage[texel] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    float side = dot(normal, samplePosition - hitPosition) >= 0.0 ? 1.0 : -1.0;
    float3 shadowOrigin = hitPosition + side * normal * SURF_BIAS;
    float3 toLight = samplePosition - shadowOrigin;
    float distanceToLight = length(toLight);
    VisibilityResult shadow = visibility(shadowOrigin, toLight / distanceToLight,
            distanceToLight * 0.999);
    float3 resolved = contribution * shadow.transmittance * reservoir.weights.z;
    outImage[texel] = float4(resolved, 1.0);
}
