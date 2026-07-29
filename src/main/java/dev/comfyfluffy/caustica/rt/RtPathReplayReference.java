package dev.comfyfluffy.caustica.rt;

/**
 * Shader-independent contract for replaying a wavefront path sample.
 *
 * <p>The GPU path carries two independent random streams: one for transport decisions and one
 * for light proposals. Keeping them separate is required before a shifted path can be evaluated
 * with a correct proposal density. This class intentionally contains no estimator code; it only
 * defines the validity, compatibility, hashing, and deterministic aggregation rules shared by
 * the seeded replay diagnostic and the future shift/reconnection stages.</p>
 */
final class RtPathReplayReference {
    static final int REPLAY_VERSION = 7;
    static final int MAX_SEGMENTS = 2;
    static final int MISMATCH_EMPTY = 1 << 0;
    static final int MISMATCH_ABI = 1 << 1;
    static final int MISMATCH_SEGMENT = 1 << 2;
    static final int MISMATCH_PATH_STATE = 1 << 3;
    static final int MISMATCH_PROPOSAL_STATE = 1 << 4;
    static final int MISMATCH_TOPOLOGY = 1 << 5;
    static final int MISMATCH_ENDPOINT = 1 << 6;
    static final int MISMATCH_PROPOSAL = 1 << 7;
    static final int MISMATCH_METADATA = 1 << 8;

    private static final int PCG_MULTIPLIER = 747796405;
    private static final int PCG_INCREMENT = (int) 2891336453L;
    private static final int PCG_OUTPUT_MULTIPLIER = 277803737;
    private static final int TOPOLOGY_MULTIPLIER = (int) 2246822519L;
    private static final int TOPOLOGY_INCREMENT = (int) 3266489917L;
    private static final int SEGMENT_TOPOLOGY_MULTIPLIER = 668265263;
    private static final int SEGMENT_PATH_MULTIPLIER = 374761393;
    private static final int SEGMENT_PROPOSAL_MULTIPLIER = 1103515245;

    record Segment(int replaySeed, int proposalReplaySeed, int topologyKey,
                   int transportDraws, int proposalDraws) {
        Segment {
            if (transportDraws < 0 || proposalDraws < 0) {
                throw new IllegalArgumentException("replay draw counts must be non-negative");
            }
        }
    }

    record SegmentState(int replayState, int proposalReplayState) {
    }

    record Aggregate(int topologyKey, int replayState, int proposalReplayState,
                     int segmentCount) {
    }

    record State(int version, int segmentCount, long transportState, long proposalState) {
        State {
            if (version != REPLAY_VERSION) {
                throw new IllegalArgumentException("unsupported path replay version");
            }
            if (segmentCount < 1 || segmentCount > MAX_SEGMENTS) {
                throw new IllegalArgumentException("path replay segment count out of range");
            }
            if (transportState < 0L || proposalState < 0L) {
                throw new IllegalArgumentException("path replay states must be non-negative");
            }
        }
    }

    static boolean compatible(State receiver, State shifted) {
        return receiver != null
                && shifted != null
                && receiver.version() == shifted.version()
                && receiver.segmentCount() == shifted.segmentCount()
                && receiver.transportState() == shifted.transportState()
                && receiver.proposalState() == shifted.proposalState();
    }

    static int pcg(int state) {
        state = state * PCG_MULTIPLIER + PCG_INCREMENT;
        int word = ((state >>> ((state >>> 28) + 4)) ^ state) * PCG_OUTPUT_MULTIPLIER;
        return (word >>> 22) ^ word;
    }

    static int pathHash(int value) {
        return pcg(value);
    }

    static int pathTopologyStep(int topology, int eventToken) {
        return pathHash(topology ^ eventToken * TOPOLOGY_MULTIPLIER + TOPOLOGY_INCREMENT);
    }

    static SegmentState replaySegment(Segment segment) {
        int replayState = segment.replaySeed();
        for (int draw = 0; draw < segment.transportDraws(); draw++) {
            replayState = pcg(replayState);
        }
        int proposalReplayState = segment.proposalReplaySeed();
        for (int draw = 0; draw < segment.proposalDraws(); draw++) {
            proposalReplayState = pcg(proposalReplayState);
        }
        return new SegmentState(replayState, proposalReplayState);
    }

    static Aggregate aggregate(int initialTopology, Segment... segments) {
        if (segments == null || segments.length < 1 || segments.length > MAX_SEGMENTS) {
            throw new IllegalArgumentException("replay must contain one or two segments");
        }
        int topology = initialTopology;
        int replayState = 0;
        int proposalReplayState = 0;
        for (int leaf = 0; leaf < segments.length; leaf++) {
            Segment segment = segments[leaf];
            if (segment == null) {
                throw new IllegalArgumentException("replay segment must not be null");
            }
            SegmentState state = replaySegment(segment);
            topology = pathTopologyStep(topology,
                    segment.topologyKey() ^ leaf * SEGMENT_TOPOLOGY_MULTIPLIER);
            replayState = pathHash(
                    replayState ^ state.replayState() ^ leaf * SEGMENT_PATH_MULTIPLIER);
            proposalReplayState = pathHash(
                    proposalReplayState ^ state.proposalReplayState()
                            ^ leaf * SEGMENT_PROPOSAL_MULTIPLIER);
        }
        return new Aggregate(topology, replayState, proposalReplayState, segments.length);
    }

    private RtPathReplayReference() {
    }
}
