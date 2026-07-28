import world_common;
import sampling_core;

public static const uint PATH_DEBUG_VIEW = 13u;
public static const uint PATH_TEMPORAL_DEBUG_VIEW = 14u;
public static const uint PATH_GENERATION_MASK = 0x00FFFFFFu;
public static const float PATH_MAX_FOOTPRINT_RATIO = 4.0;

public static const uint PATH_TRANSPORT_DIFFUSE = 1u;
public static const uint PATH_TRANSPORT_GLOSSY = 2u;
public static const uint PATH_TRANSPORT_DELTA = 4u;
public static const uint PATH_TRANSPORT_TRANSMISSION = 8u;
// Increment whenever replay metadata or RNG interpretation changes. Temporal reuse must reject
// reservoirs produced by a different contract instead of trying to replay an incompatible path.
public static const uint PATH_REPLAY_VERSION = 1u;

public struct PathTraceResult {
    public float3 radiance;
    public uint depth;
    public uint topologyKey;
    public uint replaySeed;
    public uint replayState;
    public uint transportMask;
    public float footprint;
};

public float pathLuminance(float3 color) {
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

public PathReservoir pathReservoirEmpty() {
    PathReservoir reservoir;
    reservoir.sampleRadianceTarget = float4(0.0);
    reservoir.proposalShift = float4(1.0, 1.0, 1.0, 0.0);
    reservoir.weights = float4(0.0);
    reservoir.identity = uint4(0u);
    reservoir.replay = uint4(0u);
    return reservoir;
}

public uint pathHash(uint value) {
    uint state = value;
    return pcg(state);
}

public uint pathTopologyStep(uint topology, uint eventToken) {
    return pathHash(topology ^ eventToken * 2246822519u + 3266489917u);
}

public uint pathPackMetadata(uint generation, uint depth, uint transportMask) {
    return (generation & PATH_GENERATION_MASK)
            | ((min(depth, 15u) & 0xFu) << 24u)
            | ((transportMask & 0xFu) << 28u);
}

// Initial per-pixel RIS update. The traced path is already an importance-weighted Monte-Carlo
// estimator, so this bootstrap stage uses an effective proposal density of one. A later replay pass
// will replace it with decomposed path PDFs before any temporal or spatial path reuse is allowed.
public void pathReservoirUpdate(inout PathReservoir reservoir,
                                PathTraceResult candidate,
                                uint sampleId,
                                uint pathKey,
                                uint historyGeneration,
                                inout uint selectionSeed) {
    float3 radiance = max(candidate.radiance, float3(0.0));
    float target = max(pathLuminance(radiance), 0.0);
    float proposalDensity = 1.0;
    float candidateWeight = target / proposalDensity;
    float previousWeightSum = reservoir.weights.x;
    float nextWeightSum = min(previousWeightSum + candidateWeight, 1.0e30);
    float nextM = min(reservoir.weights.y + 1.0, 16777216.0);
    bool selectCandidate = candidateWeight > 0.0
            && (previousWeightSum <= 0.0 || rndf(selectionSeed) < candidateWeight / nextWeightSum);

    if (selectCandidate) {
        reservoir.sampleRadianceTarget = float4(radiance, target);
        reservoir.proposalShift = float4(proposalDensity, 1.0,
                max(candidate.footprint, 1.0e-5), asfloat(candidate.replayState));
        reservoir.identity = uint4(sampleId, pathKey, candidate.topologyKey,
                pathPackMetadata(historyGeneration, candidate.depth, candidate.transportMask));
        // Storage-only replay contract. The current estimator deliberately does not consume this lane;
        // a later path-replay stage will validate the version before reconstructing the selected path.
        reservoir.replay = uint4(candidate.replaySeed, candidate.replayState,
                PATH_REPLAY_VERSION, candidate.depth);
    }

    reservoir.weights.x = nextWeightSum;
    reservoir.weights.y = nextM;
    float selectedTarget = reservoir.sampleRadianceTarget.w;
    reservoir.weights.z = selectedTarget > 0.0 && nextM > 0.0
            ? nextWeightSum / (nextM * selectedTarget) : 0.0;
    reservoir.weights.w = selectedTarget > 0.0 ? min(nextM * 0.25, 1.0) : 0.0;
}

// Diagnostic channels: red = normalized traced depth, green = selected target luminance,
// blue = transport-class mask. Black means no positive-contribution path was selected.
public float3 pathReservoirDebug(PathReservoir reservoir, uint maxDepth) {
    if (reservoir.weights.y <= 0.0 || reservoir.sampleRadianceTarget.w <= 0.0) {
        return float3(0.0);
    }
    uint metadata = reservoir.identity.w;
    float depth = float((metadata >> 24u) & 0xFu) / float(max(maxDepth, 1u));
    float target = reservoir.sampleRadianceTarget.w;
    float transport = float((metadata >> 28u) & 0xFu) / 15.0;
    return float3(clamp(depth, 0.0, 1.0), target / (1.0 + target), transport);
}
