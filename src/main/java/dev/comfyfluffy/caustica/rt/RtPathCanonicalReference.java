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
    static final int CANDIDATE_ACCEPTED = 0;
    static final int CANDIDATE_NO_ENDPOINT = 1;
    static final int CANDIDATE_INVALID_RADIANCE = 2;
    static final int CANDIDATE_ZERO_TARGET = 3;
    static final int CANDIDATE_INVALID_CONTINUATION_PDF = 4;
    static final int CANDIDATE_INVALID_ROULETTE_PDF = 5;
    static final int CANDIDATE_INVALID_PROPOSAL = 6;
    private static final double LUMA_R = 0.2126;
    private static final double LUMA_G = 0.7152;
    private static final double LUMA_B = 0.0722;

    record Endpoint(double red, double green, double blue, double target, int packedEvents) {
    }

    record Candidate(boolean canonicalValid, double red, double green, double blue,
                     double continuationProposalPdf, double rouletteProposalPdf) {
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

    static int candidateStatus(Candidate candidate) {
        if (candidate == null || !candidate.canonicalValid()) {
            return CANDIDATE_NO_ENDPOINT;
        }
        if (!finiteNonNegative(candidate.red()) || !finiteNonNegative(candidate.green())
                || !finiteNonNegative(candidate.blue())) {
            return CANDIDATE_INVALID_RADIANCE;
        }
        double target = luminance(candidate.red(), candidate.green(), candidate.blue());
        if (!finiteNonNegative(target)) {
            return CANDIDATE_INVALID_RADIANCE;
        }
        if (target == 0.0) {
            return CANDIDATE_ZERO_TARGET;
        }
        if (!finitePositive(candidate.continuationProposalPdf())) {
            return CANDIDATE_INVALID_CONTINUATION_PDF;
        }
        if (!finitePositive(candidate.rouletteProposalPdf())) {
            return CANDIDATE_INVALID_ROULETTE_PDF;
        }
        double proposalDensity = candidate.continuationProposalPdf()
                * candidate.rouletteProposalPdf();
        proposalDensity = proposalDensity > 0.0
                ? Math.min(Math.max(proposalDensity, 1.0e-30), 1.0e30) : 0.0;
        double candidateWeight = target / proposalDensity;
        if (!finitePositive(proposalDensity) || !finitePositive(candidateWeight)) {
            return CANDIDATE_INVALID_PROPOSAL;
        }
        return CANDIDATE_ACCEPTED;
    }

    private static boolean finiteNonNegative(double value) {
        return Double.isFinite(value) && value >= 0.0;
    }

    private static boolean finitePositive(double value) {
        return Double.isFinite(value) && value > 0.0;
    }

    private static void requireCounter(int value, String label) {
        if (value < 0) {
            throw new IllegalArgumentException(label + " must be non-negative");
        }
    }

    private RtPathCanonicalReference() {
    }
}
