package dev.comfyfluffy.caustica.rt;

/**
 * Shader-independent contract for replaying a wavefront path sample.
 *
 * <p>The GPU path carries two independent random streams: one for transport decisions and one
 * for light proposals. Keeping them separate is required before a shifted path can be evaluated
 * with a correct proposal density. This class intentionally contains no estimator code; it only
 * defines the validity and compatibility rules that the future replay pass must obey.</p>
 */
final class RtPathReplayReference {
    static final int REPLAY_VERSION = 2;
    static final int MAX_SEGMENTS = 2;

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

    private RtPathReplayReference() {
    }
}
