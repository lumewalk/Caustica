package dev.comfyfluffy.caustica.rt;

/**
 * Small CPU contract for the first path-temporal GRIS merge.
 *
 * <p>T-049 deliberately models only identity temporal mapping: the historical path is replayed
 * on the current queue, and its shift Jacobian is one. Reprojection and reconnection mappings
 * remain separate future contracts.</p>
 */
final class RtPathTemporalMergeReference {
    static final double IDENTITY_SHIFT_JACOBIAN = 1.0;
    static final double MAX_SOURCE_COUNT = 8.0;

    enum Decision {
        ACCEPTED,
        HISTORY_EMPTY,
        REPLAY_MISMATCH,
        COMPATIBILITY_MISMATCH,
        FOOTPRINT_MISMATCH,
        INVALID_JACOBIAN
    }

    record Mapping(boolean replayCompatible, double shiftJacobian) {
        Mapping {
            if (!Double.isFinite(shiftJacobian) || shiftJacobian <= 0.0) {
                throw new IllegalArgumentException("shift Jacobian must be finite and positive");
            }
        }

        static Mapping identity(boolean replayCompatible) {
            return new Mapping(replayCompatible, IDENTITY_SHIFT_JACOBIAN);
        }
    }

    static Decision merge(RtPathReservoirReference.Reservoir receiver,
                          RtPathReservoirReference.Reservoir source,
                          RtPathReservoirReference.PathSample sourceSample,
                          RtPathReservoirReference.PathSample shiftedSample,
                          Mapping mapping,
                          double random01) {
        if (source == null || sourceSample == null) {
            return Decision.HISTORY_EMPTY;
        }
        if (!mapping.replayCompatible()) {
            return Decision.REPLAY_MISMATCH;
        }
        if (Math.abs(shiftedSample.shiftJacobian() - mapping.shiftJacobian()) > 1.0e-12) {
            return Decision.INVALID_JACOBIAN;
        }
        double footprintRatio = Math.max(sourceSample.footprint(), shiftedSample.footprint())
                / Math.min(sourceSample.footprint(), shiftedSample.footprint());
        if (footprintRatio > RtPathReservoirReference.MAX_FOOTPRINT_RATIO) {
            return Decision.FOOTPRINT_MISMATCH;
        }
        if (!RtPathReservoirReference.compatible(sourceSample, shiftedSample)) {
            return Decision.COMPATIBILITY_MISMATCH;
        }
        receiver.merge(source, shiftedSample, random01, MAX_SOURCE_COUNT);
        return Decision.ACCEPTED;
    }

    private RtPathTemporalMergeReference() {
    }
}
