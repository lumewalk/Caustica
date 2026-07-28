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
public static const uint PATH_REPLAY_VERSION = 5u;
public static const uint PATH_CANONICAL_MODE_FLAG = 16u;
public static const uint PATH_PROPOSAL_CANONICAL_VALID = 1u << 30u;
public static const uint PATH_PROPOSAL_RESERVED_MASK = 1u << 31u;
// Debug-view 13 admission states. These numeric values are mirrored by
// RtPathCanonicalReference so shader and CPU diagnostics stay deterministic.
public static const uint PATH_CANONICAL_ACCEPTED = 0u;
public static const uint PATH_CANONICAL_NO_ENDPOINT = 1u;
public static const uint PATH_CANONICAL_INVALID_RADIANCE = 2u;
public static const uint PATH_CANONICAL_ZERO_TARGET = 3u;
public static const uint PATH_CANONICAL_INVALID_CONTINUATION_PDF = 4u;
public static const uint PATH_CANONICAL_INVALID_ROULETTE_PDF = 5u;
public static const uint PATH_CANONICAL_INVALID_PROPOSAL = 6u;

public struct PathTraceResult {
    public float3 radiance;
    public uint depth;
    public uint topologyKey;
    public uint replaySeed;
    public uint replayState;
    public uint proposalReplaySeed;
    public uint proposalReplayState;
    public uint transportMask;
    public float footprint;
    // Proposal components are diagnostic until the radiance candidate is split into canonical
    // contribution paths; the current candidate still sums NEE and continuation radiance.
    public float lightProposalPdf;
    public uint lightProposalEvents;
    public float continuationProposalPdf;
    public float rouletteProposalPdf;
    public uint continuationProposalEvents;
    public uint rouletteProposalEvents;
    public float3 canonicalRadiance;
    public uint canonicalValid;
    public float canonicalContinuationProposalPdf;
    public float canonicalRouletteProposalPdf;
    public uint canonicalContinuationProposalEvents;
    public uint canonicalRouletteProposalEvents;
};

// A complete screen-owned sample can contain both continuation records emitted by Pass A. Keep its
// persistent replay inputs separate from the per-segment trace result so later replay/shift code has an
// explicit contract and does not need to reinterpret a transient queue record.
public struct PathCandidate {
    public float3 radiance;
    public uint depth;
    public uint topologyKey;
    public uint4 replaySeeds;
    public uint replayState;
    public uint proposalReplayState;
    public uint replaySegmentCount;
    public uint transportMask;
    public float footprint;
    public float lightProposalPdf;
    public uint lightProposalEvents;
    public float continuationProposalPdf;
    public float rouletteProposalPdf;
    public uint continuationProposalEvents;
    public uint rouletteProposalEvents;
    public float3 canonicalRadiance;
    public uint canonicalValid;
    public float canonicalContinuationProposalPdf;
    public float canonicalRouletteProposalPdf;
    public uint canonicalContinuationProposalEvents;
    public uint canonicalRouletteProposalEvents;
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
    reservoir.replayControl = uint4(0u);
    reservoir.proposalComponents = float4(0.0, 0.0, 0.0, 0.0);
    reservoir.canonicalRadianceTarget = float4(0.0);
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

public float pathCombineProposal(float left, float right) {
    if (left <= 0.0 || right <= 0.0) return 0.0;
    float combined = left * right;
    return combined > 0.0 ? clamp(combined, 1.0e-30, 1.0e30) : 0.0;
}

public void pathAccumulateProposal(inout float product, inout uint events, float factor) {
    product = pathCombineProposal(product, factor);
    events++;
}

public uint pathPackProposalEvents(uint lightEvents, uint continuationEvents, uint rouletteEvents,
                                   bool canonicalValid) {
    return min(lightEvents, 1023u)
            | (min(continuationEvents, 1023u) << 10u)
            | (min(rouletteEvents, 1023u) << 20u)
            | (canonicalValid ? PATH_PROPOSAL_CANONICAL_VALID : 0u);
}

public bool pathCanonicalEndpointConsistent(PathReservoir reservoir) {
    uint packedEvents = asuint(reservoir.proposalComponents.w);
    if ((packedEvents & PATH_PROPOSAL_CANONICAL_VALID) == 0u
            || (packedEvents & PATH_PROPOSAL_RESERVED_MASK) != 0u) {
        return false;
    }
    float3 radiance = reservoir.canonicalRadianceTarget.xyz;
    float storedTarget = reservoir.canonicalRadianceTarget.w;
    if (!all(radiance >= float3(0.0)) || !(storedTarget >= 0.0)) {
        return false;
    }
    float expectedTarget = pathLuminance(radiance);
    float tolerance = max(1.0e-5, expectedTarget * 1.0e-4);
    return abs(expectedTarget - storedTarget) <= tolerance;
}

public uint pathCanonicalCandidateStatus(PathCandidate candidate) {
    if (candidate.canonicalValid == 0u) {
        return PATH_CANONICAL_NO_ENDPOINT;
    }
    if (!all(isfinite(candidate.canonicalRadiance))
            || !all(candidate.canonicalRadiance >= float3(0.0))) {
        return PATH_CANONICAL_INVALID_RADIANCE;
    }
    float target = pathLuminance(candidate.canonicalRadiance);
    if (!isfinite(target) || target < 0.0) {
        return PATH_CANONICAL_INVALID_RADIANCE;
    }
    if (target == 0.0) {
        return PATH_CANONICAL_ZERO_TARGET;
    }
    if (!isfinite(candidate.canonicalContinuationProposalPdf)
            || candidate.canonicalContinuationProposalPdf <= 0.0) {
        return PATH_CANONICAL_INVALID_CONTINUATION_PDF;
    }
    if (!isfinite(candidate.canonicalRouletteProposalPdf)
            || candidate.canonicalRouletteProposalPdf <= 0.0) {
        return PATH_CANONICAL_INVALID_ROULETTE_PDF;
    }
    float proposalDensity = pathCombineProposal(candidate.canonicalContinuationProposalPdf,
            candidate.canonicalRouletteProposalPdf);
    float candidateWeight = target / proposalDensity;
    if (!isfinite(proposalDensity) || proposalDensity <= 0.0
            || !isfinite(candidateWeight) || candidateWeight <= 0.0) {
        return PATH_CANONICAL_INVALID_PROPOSAL;
    }
    return PATH_CANONICAL_ACCEPTED;
}

public float3 pathCanonicalCandidateDebugColor(uint status) {
    if (status == PATH_CANONICAL_ACCEPTED) {
        return float3(0.05, 0.95, 0.20); // admitted
    }
    if (status == PATH_CANONICAL_NO_ENDPOINT) {
        return float3(1.00, 0.12, 0.55); // no sky/emissive endpoint
    }
    if (status == PATH_CANONICAL_INVALID_RADIANCE) {
        return float3(0.95, 0.05, 0.05); // non-finite or negative radiance/target
    }
    if (status == PATH_CANONICAL_ZERO_TARGET) {
        return float3(0.10, 0.25, 1.00); // valid black/zero-target endpoint
    }
    if (status == PATH_CANONICAL_INVALID_CONTINUATION_PDF) {
        return float3(0.05, 0.85, 0.95); // continuation PDF
    }
    if (status == PATH_CANONICAL_INVALID_ROULETTE_PDF) {
        return float3(0.95, 0.30, 0.04); // roulette PDF
    }
    return float3(0.95, 0.72, 0.04); // combined proposal or importance weight
}

// Initial per-pixel RIS update. The traced path is already an importance-weighted Monte-Carlo
// estimator, so this bootstrap stage uses an effective proposal density of one. A later replay pass
// will replace it with decomposed path PDFs before any temporal or spatial path reuse is allowed.
public uint pathReservoirUpdate(inout PathReservoir reservoir,
                                PathCandidate candidate,
                                uint sampleId,
                                uint pathKey,
                                uint historyGeneration,
                                inout uint selectionSeed,
                                bool canonicalMode) {
    uint canonicalStatus = PATH_CANONICAL_ACCEPTED;
    if (canonicalMode) {
        canonicalStatus = pathCanonicalCandidateStatus(candidate);
        if (canonicalStatus != PATH_CANONICAL_ACCEPTED) {
            return canonicalStatus;
        }
    }
    float3 radiance = canonicalMode
            ? max(candidate.canonicalRadiance, float3(0.0))
            : max(candidate.radiance, float3(0.0));
    float target = max(pathLuminance(radiance), 0.0);
    // The regular capture remains bootstrap-weighted because its candidate still sums multiple
    // estimators. The opt-in canonical debug mode has one endpoint and uses its continuation and
    // roulette products as the first path-only proposal-density contract.
    float proposalDensity = canonicalMode
            ? pathCombineProposal(candidate.canonicalContinuationProposalPdf,
                    candidate.canonicalRouletteProposalPdf)
            : 1.0;
    float candidateWeight = target / proposalDensity;
    float previousWeightSum = reservoir.weights.x;
    float nextWeightSum = min(previousWeightSum + candidateWeight, 1.0e30);
    float nextM = min(reservoir.weights.y + 1.0, 16777216.0);
    bool selectCandidate = candidateWeight > 0.0
            && (previousWeightSum <= 0.0 || rndf(selectionSeed) < candidateWeight / nextWeightSum);

    if (selectCandidate) {
        reservoir.sampleRadianceTarget = float4(radiance, target);
        reservoir.proposalShift = float4(1.0, 1.0,
                max(candidate.footprint, 1.0e-5), asfloat(candidate.replayState));
        reservoir.proposalComponents = float4(
                candidate.lightProposalPdf,
                candidate.canonicalContinuationProposalPdf,
                candidate.canonicalRouletteProposalPdf,
                asfloat(pathPackProposalEvents(candidate.lightProposalEvents,
                        candidate.canonicalContinuationProposalEvents,
                        candidate.canonicalRouletteProposalEvents,
                        candidate.canonicalValid != 0u)));
        float3 canonicalRadiance = max(candidate.canonicalRadiance, float3(0.0));
        float canonicalTarget = candidate.canonicalValid != 0u
                ? max(pathLuminance(canonicalRadiance), 0.0) : 0.0;
        reservoir.canonicalRadianceTarget = float4(canonicalRadiance, canonicalTarget);
        reservoir.identity = uint4(sampleId, pathKey, candidate.topologyKey,
                pathPackMetadata(historyGeneration, candidate.depth, candidate.transportMask));
        // Storage-only replay contract. The current estimator deliberately does not consume this lane;
        // a later path-replay stage will validate the version before reconstructing the selected path.
        reservoir.replay = candidate.replaySeeds;
        reservoir.replayControl = uint4(PATH_REPLAY_VERSION,
                min(candidate.replaySegmentCount, MAX_PATH_SEGMENTS),
                candidate.replayState, candidate.proposalReplayState);
    }

    reservoir.weights.x = nextWeightSum;
    reservoir.weights.y = nextM;
    float selectedTarget = reservoir.sampleRadianceTarget.w;
    reservoir.weights.z = selectedTarget > 0.0 && nextM > 0.0
            ? nextWeightSum / (nextM * selectedTarget) : 0.0;
    reservoir.weights.w = selectedTarget > 0.0 ? min(nextM * 0.25, 1.0) : 0.0;
    return canonicalStatus;
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
