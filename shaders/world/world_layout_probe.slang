// Build-time reflection probe only. This file is not packaged as a runtime shader.
// Exposing WorldPush as a structured-buffer element makes Slang's JSON reflection include the
// complete Std430DataLayout: field types, byte offsets, total size, matrix mode, and array counts.
import world_common;

struct WorldPushLayoutProbe {
    WorldPush values[2]; // array stride is the complete, tail-padded WorldPush byte size
};

struct MaterialHeaderLayoutProbe {
    MaterialHeader values[2]; // array stride is the complete, tail-padded MaterialHeader byte size
};

struct DirectReservoirLayoutProbe {
    DirectReservoir values[2]; // clarity-first persistent reservoir stride
};

struct PathReservoirLayoutProbe {
    PathReservoir values[2]; // future ReSTIR PT path-reservoir stride
};

struct PathSourceRootLayoutProbe {
    PathSourceRoot values[2]; // retained two-segment source replay root + source guides
};

[[vk::binding(0, 0)]] StructuredBuffer<WorldPushLayoutProbe, Std430DataLayout> worldPushLayoutProbe;
[[vk::binding(1, 0)]] StructuredBuffer<MaterialHeaderLayoutProbe, Std430DataLayout> materialHeaderLayoutProbe;
[[vk::binding(2, 0)]] StructuredBuffer<DirectReservoirLayoutProbe, Std430DataLayout> directReservoirLayoutProbe;
[[vk::binding(3, 0)]] StructuredBuffer<PathReservoirLayoutProbe, Std430DataLayout> pathReservoirLayoutProbe;
[[vk::binding(4, 0)]] StructuredBuffer<PathSourceRootLayoutProbe, Std430DataLayout> pathSourceRootLayoutProbe;
[[vk::push_constant]] WorldPushConstants pushConstantsLayoutProbe;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    // Keep the parameter reachable without requiring a writable output resource.
    if (id.x == 0xFFFFFFFFu) {
        float sink = worldPushLayoutProbe[0].values[0].invViewProj[0][0]
                + materialHeaderLayoutProbe[0].values[0].params.x
                + directReservoirLayoutProbe[0].values[0].weights.x
                + pathReservoirLayoutProbe[0].values[0].weights.x
                + pathSourceRootLayoutProbe[0].values[0].sourcePositionMaterial.x
                + float(pushConstantsLayoutProbe.frameIndex);
    }
}
