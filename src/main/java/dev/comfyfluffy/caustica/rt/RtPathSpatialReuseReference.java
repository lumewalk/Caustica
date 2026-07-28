package dev.comfyfluffy.caustica.rt;

/**
 * Admission and Jacobian contract for the first path spatial-reuse experiment.
 *
 * <p>This is intentionally reference-only. A GPU shift must store a reconnection vertex (and the
 * associated geometric/PDF terms) before it can consume this contract.</p>
 */
final class RtPathSpatialReuseReference {
    static final double NORMAL_COSINE_THRESHOLD = 0.85;
    static final double RELATIVE_DEPTH_THRESHOLD = 0.10;
    static final double MAX_FOOTPRINT_RATIO = 4.0;
    static final int RECONNECTION_HIT_DEPTH = 1;
    static final int RECONNECTION_VALID = 1;

    enum Decision {
        ACCEPTED,
        MATERIAL_MISMATCH,
        SURFACE_NORMAL_MISMATCH,
        RELATIVE_DEPTH_MISMATCH,
        DEPTH_MISMATCH,
        TOPOLOGY_MISMATCH,
        TRANSPORT_MISMATCH,
        FOOTPRINT_MISMATCH
    }

    record Surface(long materialKey, double normalCosine, double relativeDepth,
                   int depth, long topologyKey, int transportClass, double footprint) {
        Surface {
            if (!Double.isFinite(normalCosine) || normalCosine < -1.0 || normalCosine > 1.0
                    || !Double.isFinite(relativeDepth) || relativeDepth < 0.0
                    || depth < 0 || topologyKey < 0L || transportClass < 0
                    || !Double.isFinite(footprint) || footprint <= 0.0) {
                throw new IllegalArgumentException("invalid spatial compatibility surface");
            }
        }
    }

    /**
     * Distances are |x_(i+1)-x_i| and |x_(i+1)-y_i|. Cosines are measured at the shared next
     * vertex against the source and receiver directions respectively.
     */
    record ReconnectionGeometry(double sourceDistance, double receiverDistance,
                                double sourceCosine, double receiverCosine,
                                double sourceDirectionalPdf, double receiverDirectionalPdf) {
        ReconnectionGeometry {
            if (!positiveFinite(sourceDistance) || !positiveFinite(receiverDistance)
                    || !positiveFinite(sourceCosine) || !positiveFinite(receiverCosine)
                    || !positiveFinite(sourceDirectionalPdf)
                    || !positiveFinite(receiverDirectionalPdf)) {
                throw new IllegalArgumentException("invalid reconnection geometry/PDF");
            }
        }

        double solidAngleJacobian() {
            return (receiverCosine * sourceDistance * sourceDistance)
                    / (sourceCosine * receiverDistance * receiverDistance);
        }

        /**
         * Primary-sample-space form: the solid-angle reconnection Jacobian multiplied by the
         * receiver/source directional-PDF ratio. Random replay itself contributes unit Jacobian.
         */
        double primarySampleJacobian() {
            return solidAngleJacobian() * receiverDirectionalPdf / sourceDirectionalPdf;
        }
    }

    static Decision admit(Surface receiver, Surface source) {
        if (receiver.materialKey() != source.materialKey()) {
            return Decision.MATERIAL_MISMATCH;
        }
        if (receiver.normalCosine() < NORMAL_COSINE_THRESHOLD) {
            return Decision.SURFACE_NORMAL_MISMATCH;
        }
        if (receiver.relativeDepth() > RELATIVE_DEPTH_THRESHOLD) {
            return Decision.RELATIVE_DEPTH_MISMATCH;
        }
        if (receiver.depth() != source.depth()) {
            return Decision.DEPTH_MISMATCH;
        }
        if (receiver.topologyKey() != source.topologyKey()) {
            return Decision.TOPOLOGY_MISMATCH;
        }
        if (receiver.transportClass() != source.transportClass()) {
            return Decision.TRANSPORT_MISMATCH;
        }
        double footprintRatio = Math.max(receiver.footprint(), source.footprint())
                / Math.min(receiver.footprint(), source.footprint());
        return footprintRatio <= MAX_FOOTPRINT_RATIO
                ? Decision.ACCEPTED : Decision.FOOTPRINT_MISMATCH;
    }

    static boolean captureReconnectionAtDepth(int hitDepth) {
        return hitDepth == RECONNECTION_HIT_DEPTH;
    }

    static int packReconnectionMetadata(int depth, boolean valid) {
        if (depth < 0) {
            throw new IllegalArgumentException("reconnection depth must be non-negative");
        }
        return (valid ? RECONNECTION_VALID : 0) | ((Math.min(depth, 15) & 0xF) << 1);
    }

    static boolean reconnectionValid(int packedMetadata) {
        return (packedMetadata & RECONNECTION_VALID) != 0;
    }

    static int reconnectionDepth(int packedMetadata) {
        return (packedMetadata >>> 1) & 0xF;
    }

    private static boolean positiveFinite(double value) {
        return Double.isFinite(value) && value > 0.0;
    }

    private RtPathSpatialReuseReference() {
    }
}
