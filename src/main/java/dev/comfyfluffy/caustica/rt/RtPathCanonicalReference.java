package dev.comfyfluffy.caustica.rt;

/**
 * Shader-independent contract for the canonical sky/emissive endpoint stored with a path sample.
 *
 * <p>The endpoint-valid bit is kept beside the packed proposal event counters so a valid black
 * endpoint remains distinguishable from a path that never reached an endpoint.</p>
 */
final class RtPathCanonicalReference {
    static final int CANONICAL_VALID_BIT = 1 << 30;
    static final int RESERVED_MASK = 1 << 31;
    private static final double LUMA_R = 0.2126;
    private static final double LUMA_G = 0.7152;
    private static final double LUMA_B = 0.0722;

    record Endpoint(double red, double green, double blue, double target, int packedEvents) {
    }

    static int packProposalEvents(int lightEvents, int continuationEvents, int rouletteEvents,
                                  boolean canonicalValid) {
        requireCounter(lightEvents, "light events");
        requireCounter(continuationEvents, "continuation events");
        requireCounter(rouletteEvents, "roulette events");
        return Math.min(lightEvents, 1023)
                | (Math.min(continuationEvents, 1023) << 10)
                | (Math.min(rouletteEvents, 1023) << 20)
                | (canonicalValid ? CANONICAL_VALID_BIT : 0);
    }

    static boolean isConsistent(Endpoint endpoint) {
        if (endpoint == null || (endpoint.packedEvents() & CANONICAL_VALID_BIT) == 0
                || (endpoint.packedEvents() & RESERVED_MASK) != 0) {
            return false;
        }
        if (!finiteNonNegative(endpoint.red()) || !finiteNonNegative(endpoint.green())
                || !finiteNonNegative(endpoint.blue()) || !finiteNonNegative(endpoint.target())) {
            return false;
        }
        double expected = luminance(endpoint.red(), endpoint.green(), endpoint.blue());
        return Math.abs(expected - endpoint.target())
                <= Math.max(1.0e-5, expected * 1.0e-4);
    }

    static double luminance(double red, double green, double blue) {
        return red * LUMA_R + green * LUMA_G + blue * LUMA_B;
    }

    private static boolean finiteNonNegative(double value) {
        return Double.isFinite(value) && value >= 0.0;
    }

    private static void requireCounter(int value, String label) {
        if (value < 0) {
            throw new IllegalArgumentException(label + " must be non-negative");
        }
    }

    private RtPathCanonicalReference() {
    }
}
