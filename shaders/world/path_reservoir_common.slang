import world_common;
import sampling_core;

public static const uint PATH_DEBUG_VIEW = 13u;
public static const uint PATH_TEMPORAL_DEBUG_VIEW = 14u;
public static const uint PATH_REPLAY_DEBUG_VIEW = 15u;
public static const uint PATH_TEMPORAL_MERGE_DEBUG_VIEW = 16u;
public static const uint PATH_GENERATION_MASK = 0x00FFFFFFu;
public static const float PATH_MAX_FOOTPRINT_RATIO = 4.0;
public static const float PATH_TEMPORAL_MAX_SOURCE_COUNT = 8.0;
public static const float PATH_IDENTITY_SHIFT_JACOBIAN = 1.0;

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

// Replay diagnostic mismatch bits. A pixel may expose more than one independent mismatch; the
// debug color adds one channel per category instead of hiding the root cause behind one enum.
public static const uint PATH_REPLAY_EMPTY = 1u << 0u;
public static const uint PATH_REPLAY_ABI_MISMATCH = 1u << 1u;
public static const uint PATH_REPLAY_SEGMENT_MISMATCH = 1u << 2u;
public static const uint PATH_REPLAY_PATH_STATE_MISMATCH = 1u << 3u;
public static const uint PATH_REPLAY_PROPOSAL_STATE_MISMATCH = 1u << 4u;
public static const uint PATH_REPLAY_TOPOLOGY_MISMATCH = 1u << 5u;
public static const uint PATH_REPLAY_ENDPOINT_MISMATCH = 1u << 6u;
public static const uint PATH_REPLAY_PROPOSAL_MISMATCH = 1u << 7u;
public static const uint PATH_REPLAY_METADATA_MISMATCH = 1u << 8u;

// Temporal merge diagnostic bits. The first five bits explain a rejected history sample;
// accepted merges add MERGED and optionally HISTORY_SELECTED.
public static const uint PATH_TEMPORAL_CURRENT_EMPTY = 1u << 0u;
public static const uint PATH_TEMPORAL_HISTORY_EMPTY = 1u << 1u;
public static const uint PATH_TEMPORAL_REPLAY_REJECT = 1u << 2u;
public static const uint PATH_TEMPORAL_COMPATIBILITY_REJECT = 1u << 3u;
public static const uint PATH_TEMPORAL_FOOTPRINT_REJECT = 1u << 4u;
public static const uint PATH_TEMPORAL_MERGED = 1u << 5u;
public static const uint PATH_TEMPORAL_HISTORY_SELECTED = 1u << 6u;

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

// Aggregate returned by a seeded replay of the same queue chain used by the selected candidate.
// Keeping this shape separate from PathCandidate makes the comparison explicit and prevents the
// replay diagnostic from accidentally feeding a second reservoir update.
public struct PathReplayAggregate {
    public float3 radiance;
    public uint depth;
    public uint topologyKey;
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

public PathReplayAggregate pathReplayAggregateEmpty() {
    PathReplayAggregate aggregate;
    aggregate.radiance = float3(0.0);
    aggregate.depth = 0u;
    aggregate.topologyKey = pathHash(0x7F4A7C15u);
    aggregate.replayState = 0u;
    aggregate.proposalReplayState = 0u;
    aggregate.replaySegmentCount = 0u;
    aggregate.transportMask = 0u;
    aggregate.footprint = 1.0e-5;
    aggregate.lightProposalPdf = 1.0;
    aggregate.lightProposalEvents = 0u;
    aggregate.continuationProposalPdf = 1.0;
    aggregate.rouletteProposalPdf = 1.0;
    aggregate.continuationProposalEvents = 0u;
    aggregate.rouletteProposalEvents = 0u;
    aggregate.canonicalRadiance = float3(0.0);
    aggregate.canonicalValid = 0u;
    aggregate.canonicalContinuationProposalPdf = 1.0;
    aggregate.canonicalRouletteProposalPdf = 1.0;
    aggregate.canonicalContinuationProposalEvents = 0u;
    aggregate.canonicalRouletteProposalEvents = 0u;
    return aggregate;
}

public void pathReplayAccumulate(inout PathReplayAggregate aggregate,
                                 PathTraceResult traced, uint leaf) {
    aggregate.replaySegmentCount = leaf + 1u;
    aggregate.radiance += traced.radiance;
    aggregate.depth = max(aggregate.depth, traced.depth);
    aggregate.topologyKey = pathTopologyStep(aggregate.topologyKey,
            traced.topologyKey ^ leaf * 668265263u);
    aggregate.replayState = pathHash(
            aggregate.replayState ^ traced.replayState ^ leaf * 374761393u);
    aggregate.proposalReplayState = pathHash(
            aggregate.proposalReplayState ^ traced.proposalReplayState ^ leaf * 1103515245u);
    aggregate.transportMask |= traced.transportMask;
    aggregate.footprint = max(aggregate.footprint, traced.footprint);
    aggregate.lightProposalPdf = pathCombineProposal(
            aggregate.lightProposalPdf, traced.lightProposalPdf);
    aggregate.lightProposalEvents += traced.lightProposalEvents;
    aggregate.continuationProposalPdf = pathCombineProposal(
            aggregate.continuationProposalPdf, traced.continuationProposalPdf);
    aggregate.continuationProposalEvents += traced.continuationProposalEvents;
    aggregate.rouletteProposalPdf = pathCombineProposal(
            aggregate.rouletteProposalPdf, traced.rouletteProposalPdf);
    aggregate.rouletteProposalEvents += traced.rouletteProposalEvents;
    if (aggregate.canonicalValid == 0u && traced.canonicalValid != 0u) {
        aggregate.canonicalRadiance = traced.canonicalRadiance;
        aggregate.canonicalValid = 1u;
        aggregate.canonicalContinuationProposalPdf = traced.canonicalContinuationProposalPdf;
        aggregate.canonicalContinuationProposalEvents = traced.canonicalContinuationProposalEvents;
        aggregate.canonicalRouletteProposalPdf = traced.canonicalRouletteProposalPdf;
        aggregate.canonicalRouletteProposalEvents = traced.canonicalRouletteProposalEvents;
    }
}

public bool pathReplayFloatMatches(float expected, float actual) {
    float tolerance = max(1.0e-5, max(abs(expected), abs(actual)) * 1.0e-4);
    return isfinite(expected) && isfinite(actual) && abs(expected - actual) <= tolerance;
}

public bool pathReplayColorMatches(float3 expected, float3 actual) {
    return all(pathReplayFloatMatches(expected.r, actual.r)
            && pathReplayFloatMatches(expected.g, actual.g)
            && pathReplayFloatMatches(expected.b, actual.b));
}

public uint pathReplayCompare(PathReservoir reservoir, PathReplayAggregate replay) {
    uint mismatch = 0u;
    uint segmentCount = reservoir.replayControl.y;
    if (reservoir.replayControl.x != PATH_REPLAY_VERSION
            || segmentCount < 1u || segmentCount > MAX_PATH_SEGMENTS
            || (segmentCount < MAX_PATH_SEGMENTS && any(reservoir.replay.zw != uint2(0u)))) {
        mismatch |= PATH_REPLAY_ABI_MISMATCH;
    }
    if (replay.replaySegmentCount != segmentCount) {
        mismatch |= PATH_REPLAY_SEGMENT_MISMATCH;
    }
    if (replay.replayState != reservoir.replayControl.z) {
        mismatch |= PATH_REPLAY_PATH_STATE_MISMATCH;
    }
    if (replay.proposalReplayState != reservoir.replayControl.w) {
        mismatch |= PATH_REPLAY_PROPOSAL_STATE_MISMATCH;
    }
    if (replay.topologyKey != reservoir.identity.z) {
        mismatch |= PATH_REPLAY_TOPOLOGY_MISMATCH;
    }
    if (replay.canonicalValid == 0u
            || !pathReplayColorMatches(replay.canonicalRadiance,
                    reservoir.canonicalRadianceTarget.xyz)
            || !pathReplayFloatMatches(pathLuminance(replay.canonicalRadiance),
                    reservoir.canonicalRadianceTarget.w)) {
        mismatch |= PATH_REPLAY_ENDPOINT_MISMATCH;
    }
    uint packedEvents = asuint(reservoir.proposalComponents.w);
    uint expectedCanonicalValid = (packedEvents & PATH_PROPOSAL_CANONICAL_VALID) != 0u ? 1u : 0u;
    if (!pathReplayFloatMatches(replay.lightProposalPdf, reservoir.proposalComponents.x)
            || !pathReplayFloatMatches(replay.canonicalContinuationProposalPdf,
                    reservoir.proposalComponents.y)
            || !pathReplayFloatMatches(replay.canonicalRouletteProposalPdf,
                    reservoir.proposalComponents.z)
            || ((replay.canonicalValid != 0u ? 1u : 0u) != expectedCanonicalValid)
            || ((packedEvents & 1023u)
                    != min(replay.lightProposalEvents, 1023u))
            || (((packedEvents >> 10u) & 1023u)
                    != min(replay.canonicalContinuationProposalEvents, 1023u))
            || (((packedEvents >> 20u) & 1023u)
                    != min(replay.canonicalRouletteProposalEvents, 1023u))
            || (packedEvents & PATH_PROPOSAL_RESERVED_MASK) != 0u) {
        mismatch |= PATH_REPLAY_PROPOSAL_MISMATCH;
    }
    uint storedMetadata = reservoir.identity.w;
    uint replayMetadata = pathPackMetadata(storedMetadata & PATH_GENERATION_MASK,
            replay.depth, replay.transportMask);
    if (replayMetadata != storedMetadata) {
        mismatch |= PATH_REPLAY_METADATA_MISMATCH;
    }
    return mismatch;
}

public float3 pathReplayDebugColor(uint mismatch) {
    if ((mismatch & PATH_REPLAY_EMPTY) != 0u) {
        return float3(0.0);
    }
    if (mismatch == 0u) {
        return float3(0.05, 0.95, 0.20); // exact seeded replay
    }
    float3 color = float3(0.0);
    if ((mismatch & PATH_REPLAY_ABI_MISMATCH) != 0u) color += float3(0.05, 0.85, 0.95);
    if ((mismatch & PATH_REPLAY_SEGMENT_MISMATCH) != 0u) color += float3(0.55, 0.10, 0.95);
    if ((mismatch & PATH_REPLAY_PATH_STATE_MISMATCH) != 0u) color += float3(0.95, 0.05, 0.05);
    if ((mismatch & PATH_REPLAY_PROPOSAL_STATE_MISMATCH) != 0u) color += float3(0.95, 0.30, 0.04);
    if ((mismatch & PATH_REPLAY_TOPOLOGY_MISMATCH) != 0u) color += float3(0.15, 0.35, 0.95);
    if ((mismatch & PATH_REPLAY_ENDPOINT_MISMATCH) != 0u) color += float3(1.00, 0.12, 0.55);
    if ((mismatch & PATH_REPLAY_PROPOSAL_MISMATCH) != 0u) color += float3(0.95, 0.72, 0.04);
    if ((mismatch & PATH_REPLAY_METADATA_MISMATCH) != 0u) color += float3(0.80, 0.80, 0.80);
    return min(color, float3(1.0));
}

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

public uint pathMetadataDepth(uint metadata) {
    return (metadata >> 24u) & 0xFu;
}

public uint pathMetadataTransport(uint metadata) {
    return (metadata >> 28u) & 0xFu;
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

public PathCandidate pathCandidateFromReplay(PathReplayAggregate replay,
                                              PathReservoir source) {
    PathCandidate candidate;
    candidate.radiance = replay.radiance;
    candidate.depth = replay.depth;
    candidate.topologyKey = replay.topologyKey;
    candidate.replaySeeds = source.replay;
    candidate.replayState = replay.replayState;
    candidate.proposalReplayState = replay.proposalReplayState;
    candidate.replaySegmentCount = replay.replaySegmentCount;
    candidate.transportMask = replay.transportMask;
    candidate.footprint = replay.footprint;
    candidate.lightProposalPdf = replay.lightProposalPdf;
    candidate.lightProposalEvents = replay.lightProposalEvents;
    candidate.continuationProposalPdf = replay.continuationProposalPdf;
    candidate.rouletteProposalPdf = replay.rouletteProposalPdf;
    candidate.continuationProposalEvents = replay.continuationProposalEvents;
    candidate.rouletteProposalEvents = replay.rouletteProposalEvents;
    candidate.canonicalRadiance = replay.canonicalRadiance;
    candidate.canonicalValid = replay.canonicalValid;
    candidate.canonicalContinuationProposalPdf = replay.canonicalContinuationProposalPdf;
    candidate.canonicalRouletteProposalPdf = replay.canonicalRouletteProposalPdf;
    candidate.canonicalContinuationProposalEvents = replay.canonicalContinuationProposalEvents;
    candidate.canonicalRouletteProposalEvents = replay.canonicalRouletteProposalEvents;
    return candidate;
}

public uint pathReservoirTemporalMerge(inout PathReservoir receiver,
                                       PathReservoir source,
                                       PathCandidate shifted,
                                       uint pathKey,
                                       uint historyGeneration,
                                       inout uint selectionSeed) {
    uint status = 0u;
    if (receiver.weights.y <= 0.0 || receiver.sampleRadianceTarget.w <= 0.0) {
        status |= PATH_TEMPORAL_CURRENT_EMPTY;
    }
    if (source.weights.y <= 0.0 || source.sampleRadianceTarget.w <= 0.0
            || source.weights.z <= 0.0 || shifted.canonicalValid == 0u) {
        return status | PATH_TEMPORAL_HISTORY_EMPTY;
    }

    uint sourceMetadata = source.identity.w;
    uint shiftedMetadata = pathPackMetadata(sourceMetadata & PATH_GENERATION_MASK,
            shifted.depth, shifted.transportMask);
    if (shifted.topologyKey != source.identity.z
            || shiftedMetadata != sourceMetadata) {
        return status | PATH_TEMPORAL_COMPATIBILITY_REJECT;
    }
    float sourceFootprint = max(source.proposalShift.z, 1.0e-5);
    float shiftedFootprint = max(shifted.footprint, 1.0e-5);
    float sourceFootprintRatio = max(sourceFootprint, shiftedFootprint)
            / min(sourceFootprint, shiftedFootprint);
    if (sourceFootprintRatio > PATH_MAX_FOOTPRINT_RATIO) {
        return status | PATH_TEMPORAL_FOOTPRINT_REJECT;
    }
    if (receiver.weights.y > 0.0) {
        uint receiverMetadata = receiver.identity.w;
        if (receiver.identity.z != shifted.topologyKey
                || pathMetadataDepth(receiverMetadata) != shifted.depth
                || pathMetadataTransport(receiverMetadata) != (shifted.transportMask & 0xFu)) {
            return status | PATH_TEMPORAL_COMPATIBILITY_REJECT;
        }
        float receiverFootprint = max(receiver.proposalShift.z, 1.0e-5);
        float receiverFootprintRatio = max(receiverFootprint, shiftedFootprint)
                / min(receiverFootprint, shiftedFootprint);
        if (receiverFootprintRatio > PATH_MAX_FOOTPRINT_RATIO) {
            return status | PATH_TEMPORAL_FOOTPRINT_REJECT;
        }
    }

    float sourceCount = min(source.weights.y, PATH_TEMPORAL_MAX_SOURCE_COUNT);
    float mergeTarget = max(pathLuminance(shifted.canonicalRadiance), 0.0);
    float mergeWeight = mergeTarget * source.weights.z * sourceCount
            * PATH_IDENTITY_SHIFT_JACOBIAN;
    float nextWeightSum = min(receiver.weights.x + mergeWeight, 1.0e30);
    float nextM = min(receiver.weights.y + sourceCount, 16777216.0);
    bool selectHistory = mergeWeight > 0.0
            && (receiver.weights.x <= 0.0
                    || rndf(selectionSeed) < mergeWeight / nextWeightSum);
    if (selectHistory) {
        receiver.sampleRadianceTarget = float4(max(shifted.canonicalRadiance, float3(0.0)),
                mergeTarget);
        receiver.proposalShift = float4(1.0, 1.0, shiftedFootprint,
                asfloat(shifted.replayState));
        receiver.proposalComponents = float4(
                shifted.lightProposalPdf,
                shifted.canonicalContinuationProposalPdf,
                shifted.canonicalRouletteProposalPdf,
                asfloat(pathPackProposalEvents(shifted.lightProposalEvents,
                        shifted.canonicalContinuationProposalEvents,
                        shifted.canonicalRouletteProposalEvents,
                        shifted.canonicalValid != 0u)));
        receiver.canonicalRadianceTarget = float4(max(shifted.canonicalRadiance, float3(0.0)),
                mergeTarget);
        receiver.identity = uint4(source.identity.x, pathKey, shifted.topologyKey,
                pathPackMetadata(historyGeneration, shifted.depth, shifted.transportMask));
        receiver.replay = shifted.replaySeeds;
        receiver.replayControl = uint4(PATH_REPLAY_VERSION,
                min(shifted.replaySegmentCount, MAX_PATH_SEGMENTS),
                shifted.replayState, shifted.proposalReplayState);
        status |= PATH_TEMPORAL_HISTORY_SELECTED;
    }
    receiver.weights.x = nextWeightSum;
    receiver.weights.y = nextM;
    float selectedTarget = receiver.sampleRadianceTarget.w;
    receiver.weights.z = selectedTarget > 0.0 && nextM > 0.0
            ? nextWeightSum / (nextM * selectedTarget) : 0.0;
    receiver.weights.w = selectedTarget > 0.0 ? min(nextM * 0.25, 1.0) : 0.0;
    return status | PATH_TEMPORAL_MERGED;
}

public float3 pathTemporalMergeDebugColor(uint status) {
    if ((status & PATH_TEMPORAL_HISTORY_EMPTY) != 0u
            && (status & PATH_TEMPORAL_CURRENT_EMPTY) != 0u) {
        return float3(0.0);
    }
    if ((status & PATH_TEMPORAL_REPLAY_REJECT) != 0u) {
        return float3(0.05, 0.85, 0.95);
    }
    if ((status & PATH_TEMPORAL_FOOTPRINT_REJECT) != 0u) {
        return float3(0.55, 0.10, 0.95);
    }
    if ((status & PATH_TEMPORAL_COMPATIBILITY_REJECT) != 0u) {
        return float3(0.15, 0.35, 0.95);
    }
    if ((status & PATH_TEMPORAL_HISTORY_EMPTY) != 0u) {
        return float3(0.10, 0.10, 0.10);
    }
    if ((status & PATH_TEMPORAL_HISTORY_SELECTED) != 0u) {
        return float3(0.05, 0.95, 0.20);
    }
    if ((status & PATH_TEMPORAL_MERGED) != 0u) {
        return float3(0.20, 0.70, 0.35);
    }
    return float3(0.80, 0.65, 0.05);
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
