// Binding-free sampling primitives shared by ray tracing and compute passes.
// Keep this module independent of world_core: ReSTIR compute passes must not inherit ray-tracing
// descriptors merely to use the same RNG and target luminance as the existing RIS estimator.

public float luminance(float3 c) {
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

public uint pcg(inout uint s) {
    s = s * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

public float rndf(inout uint s) {
    // The high 24 bits are exactly representable as float and therefore never round up to 1.0.
    return float(pcg(s) >> 8u) * (1.0 / 16777216.0);
}
