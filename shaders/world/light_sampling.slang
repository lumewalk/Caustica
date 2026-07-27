// Shared emitter proposal distribution. Both the legacy inline RIS estimator and the persistent
// ReSTIR candidate pass call these exact functions, so selection and reconstructed PDFs cannot drift.
// All frame state is explicit to keep this module binding-free and usable from compute shaders.

import world_common;
import sampling_core;

public struct LightProposalState {
    public LightGridCell cell;
    public int3 cellCoord;
    public uint candidateCount;
    public uint globalCandidateCount;
    public float localProbability;
    public bool hasGridCell;
};

public void selectGlobalLight(WorldPushConstants constants, WorldPush push,
                       inout uint proposalSeed, out uint lightIndex) {
    float aliasSample = rndf(proposalSeed) * float(push.lightCount);
    uint column = min(uint(aliasSample), push.lightCount - 1u);
    if (constants.lightAliasAddr != 0) {
        LightAlias a = ConstPtr<LightAlias>(constants.lightAliasAddr)[column];
        bool self = aliasSample - float(column) < a.accept;
        lightIndex = self ? column : a.aliasIndex;
    } else {
        lightIndex = column;
    }
}

public bool findLightGridCell(WorldPushConstants constants, WorldPush push, float3 p,
                       out LightGridCell cell, out int3 cellCoord) {
    cell.spanOffset = 0u;
    cell.spanCount = 0u;
    cell.invWeightSum = 0.0;
    cellCoord = int3(0, 0, 0);
    if (constants.lightGridCellAddr == 0 || constants.lightGridSpanAddr == 0) return false;
    int3 coord = int3(floor((p - push.lightGridOrigin.xyz) / push.lightGridOrigin.w));
    if (coord.x < 0 || coord.y < 0 || coord.z < 0
            || coord.x >= push.lightGridDims.x
            || coord.y >= push.lightGridDims.y
            || coord.z >= push.lightGridDims.z) return false;
    uint linear = (uint(coord.z) * uint(push.lightGridDims.y) + uint(coord.y))
            * uint(push.lightGridDims.x) + uint(coord.x);
    cell = ConstPtr<LightGridCell>(constants.lightGridCellAddr)[linear];
    cellCoord = coord;
    return cell.spanCount > 0u;
}

public void selectSectionLight(WorldPushConstants constants, uint firstLight, uint lightCount,
                        inout uint proposalSeed, out uint lightIndex) {
    float aliasSample = rndf(proposalSeed) * float(lightCount);
    uint column = min(uint(aliasSample), lightCount - 1u);
    LightAlias alias = ConstPtr<LightAlias>(constants.lightLocalAliasAddr)[firstLight + column];
    bool self = aliasSample - float(column) < alias.accept;
    uint localIndex = self ? column : alias.aliasIndex;
    lightIndex = firstLight + localIndex;
}

public float unpackUnsignedFloat(uint bits, uint mantissaBits) {
    uint mantissaMask = (1u << mantissaBits) - 1u;
    uint mantissa = bits & mantissaMask;
    uint exponent = (bits >> mantissaBits) & 31u;
    if (exponent == 0u) {
        return float(mantissa) * exp2(float(1 - 15 - int(mantissaBits)));
    }
    return (1.0 + float(mantissa) / float(1u << mantissaBits))
            * exp2(float(int(exponent) - 15));
}

public float3 lightRadiance(Light light) {
    uint packed = light.le;
    return float3(unpackUnsignedFloat(packed & 0x7ffu, 6u),
            unpackUnsignedFloat((packed >> 11u) & 0x7ffu, 6u),
            unpackUnsignedFloat((packed >> 22u) & 0x3ffu, 5u));
}

public int3 lightSectionCoord(Light light) {
    uint packed = light.section;
    return int3(int(packed & 0x3ffu), int((packed >> 10u) & 0x3ffu),
            int((packed >> 20u) & 0x3ffu));
}

public float3 lightHalfU(Light light) {
    return float3(unpackHalf2(light.halfUxy), unpackHalf2(light.halfUzVx).x);
}

public float3 lightHalfV(Light light) {
    return float3(unpackHalf2(light.halfUzVx).y, unpackHalf2(light.halfVyz));
}

public float3 lightCrossUV(Light light) {
    return cross(lightHalfU(light), lightHalfV(light));
}

public float lightArea(Light light) {
    return 4.0 * length(lightCrossUV(light));
}

public float3 lightGeometricNormal(Light light) {
    float3 normal = normalize(lightCrossUV(light));
    return (light.section & 0x40000000u) != 0u ? -normal : normal;
}

public float proposalPdf(WorldPushConstants constants, WorldPush push, Light light, float3 le,
                  LightGridCell cell, int3 cellCoord, float localProbability) {
    float power = max(0.0, lightArea(light) * luminance(le));
    float globalPdf = constants.lightAliasAddr != 0
            ? power * push.lightRebase.w : 1.0 / float(push.lightCount);
    float localPdf = 0.0;
    if (localProbability > 0.0 && power > 0.0) {
        int3 delta = lightSectionCoord(light) - cellCoord;
        if (all(abs(delta) <= int3(2, 2, 2))) {
            float3 deltaF = float3(delta);
            localPdf = power * cell.invWeightSum / max(1.0, dot(deltaF, deltaF));
        }
    }
    return localProbability * localPdf + (1.0 - localProbability) * globalPdf;
}

public void selectLightGridSpanLight(WorldPushConstants constants, LightGridCell cell,
                              inout uint proposalSeed, out uint lightIndex) {
    ConstPtr<LightGridSpan> spans = ConstPtr<LightGridSpan>(constants.lightGridSpanAddr);
    float aliasSample = rndf(proposalSeed) * float(cell.spanCount);
    uint column = min(uint(aliasSample), cell.spanCount - 1u);
    LightGridSpan span = spans[cell.spanOffset + column];
    bool self = aliasSample - float(column) < span.accept;
    uint firstLight = self ? span.firstLight : span.aliasFirstLight;
    uint lightCount = self ? (span.packedLightCounts & 0xffffu)
            : (span.packedLightCounts >> 16u);
    selectSectionLight(constants, firstLight, lightCount, proposalSeed, lightIndex);
}

public void selectLightGridLight(WorldPushConstants constants, WorldPush push, LightGridCell cell,
                          bool useLocal, inout uint proposalSeed, out uint lightIndex) {
    if (useLocal) {
        selectLightGridSpanLight(constants, cell, proposalSeed, lightIndex);
    } else {
        selectGlobalLight(constants, push, proposalSeed, lightIndex);
    }
}

public LightProposalState prepareLightProposal(WorldPushConstants constants, WorldPush push,
                                        float3 hitPos, inout uint proposalSeed) {
    LightProposalState state;
    state.cell.spanOffset = 0u;
    state.cell.spanCount = 0u;
    state.cell.invWeightSum = 0.0;
    state.cellCoord = int3(0, 0, 0);
    state.candidateCount = push.risCandidates;
    float3 gridLookup = hitPos;
    if (constants.lightGridCellAddr != 0) {
        gridLookup += (float3(rndf(proposalSeed), rndf(proposalSeed), rndf(proposalSeed)) - 0.5)
                * push.lightGridOrigin.w;
    }
    state.hasGridCell = findLightGridCell(constants, push, gridLookup, state.cell, state.cellCoord);
    state.globalCandidateCount = state.hasGridCell
            ? max(1u, (state.candidateCount + 2u) / 4u) : state.candidateCount;
    uint localCandidateCount = state.candidateCount - state.globalCandidateCount;
    state.localProbability = float(localCandidateCount) / float(state.candidateCount);
    return state;
}

public uint selectLightProposal(WorldPushConstants constants, WorldPush push,
                         LightProposalState state, uint candidateIndex, inout uint proposalSeed) {
    uint globalsBefore = (candidateIndex * state.globalCandidateCount) / state.candidateCount;
    uint globalsAfter = ((candidateIndex + 1u) * state.globalCandidateCount) / state.candidateCount;
    bool useLocal = state.hasGridCell && globalsAfter == globalsBefore;
    uint lightIndex;
    selectLightGridLight(constants, push, state.cell, useLocal, proposalSeed, lightIndex);
    return lightIndex;
}
