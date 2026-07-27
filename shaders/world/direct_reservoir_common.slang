// Binding-free persistent direct-reservoir contract shared by candidate and reuse passes.

import world_common;
import sampling_core;

public static const uint RESERVOIR_FLAG_VALID = 1u;
public static const uint RESERVOIR_FLAG_DIFFUSE_TARGET = 2u;
public static const uint RESERVOIR_FLAG_SSS_TARGET = 4u;
public static const uint RESERVOIR_FLAG_TEMPORAL_MERGED = 8u;
public static const uint RESERVOIR_FLAG_TEMPORAL_SELECTED = 16u;
public static const uint RESERVOIR_FLAG_SPATIAL_MERGED = 32u;
public static const uint RESERVOIR_FLAG_SPATIAL_SELECTED = 64u;
public static const uint RESERVOIR_MAX_AGE = 30u;
public static const float RESERVOIR_HISTORY_M_MULTIPLIER = 20.0;
public static const uint RESERVOIR_SPATIAL_SAMPLE_COUNT = 4u;
public static const float RESERVOIR_SPATIAL_RADIUS = 16.0;
public static const float RESERVOIR_SPATIAL_NORMAL_COS = 0.85;
public static const float RESERVOIR_SPATIAL_RELATIVE_DEPTH = 0.10;

static const float DIRECT_INV_PI = 0.31830988618;
static const float DIRECT_SSS_G = 0.6;
static const float DIRECT_SSS_STRENGTH = 1.0;

public uint reservoirAge(DirectReservoir reservoir) {
    return reservoir.metadata.w & 0xffffu;
}

public uint reservoirFlags(DirectReservoir reservoir) {
    return reservoir.metadata.w >> 16u;
}

public float directHgPhase(float cosT) {
    float g2 = DIRECT_SSS_G * DIRECT_SSS_G;
    return (1.0 - g2) / (4.0 * 3.14159265359
            * pow(max(0.0, 1.0 + g2 - 2.0 * DIRECT_SSS_G * cosT), 1.5));
}

// Unshadowed diffuse/SSS contribution for the proposal target. samplePosition is in the receiver's
// current rebased world space. Keeping the RGB estimator and scalar target on one implementation avoids
// a silent mismatch when the diagnostic resolve becomes the production direct-light path later.
public float3 directCandidateContribution(float3 samplePosition, float3 lightNormal, float3 radiance,
                            float area, float3 hitPosition, float3 normal, float3 rayDirection,
                            float3 diffuseAlbedo, bool twoSided, float sss) {
    float3 toLight = samplePosition - hitPosition;
    float distanceSquared = dot(toLight, toLight);
    if (distanceSquared < 1.0e-6) return float3(0.0, 0.0, 0.0);
    float3 wi = toLight * rsqrt(distanceSquared);
    float cosLight = max(0.0, dot(lightNormal, -wi));
    if (cosLight <= 0.0) return float3(0.0, 0.0, 0.0);
    float ndl = dot(normal, wi);
    if (twoSided) ndl = abs(ndl);
    float3 contribution = float3(0.0, 0.0, 0.0);
    if (ndl > 0.0) {
        float geometry = ndl * cosLight / distanceSquared;
        contribution = diffuseAlbedo * DIRECT_INV_PI * radiance * (geometry * area);
    } else if (sss > 0.0) {
        float geometry = (-ndl) * cosLight / distanceSquared;
        contribution = diffuseAlbedo * sss * DIRECT_SSS_STRENGTH
                * directHgPhase(dot(wi, rayDirection)) * radiance * (geometry * area);
    }
    return contribution;
}

public float directCandidateTarget(float3 samplePosition, float3 lightNormal, float3 radiance,
                            float area, float3 hitPosition, float3 normal, float3 rayDirection,
                            float3 diffuseAlbedo, bool twoSided, float sss) {
    return luminance(directCandidateContribution(samplePosition, lightNormal, radiance, area,
            hitPosition, normal, rayDirection, diffuseAlbedo, twoSided, sss));
}
