package dev.comfyfluffy.caustica.rt;

/**
 * Shader-independent proposal-density contract for a path sample.
 *
 * <p>Each event records the probability of selecting the technique and, for continuous events,
 * the directional density in one agreed measure. Delta events deliberately do not multiply a
 * directional density: their mass is represented by the technique-selection probability.
 * {@link RtPathCanonicalProposalReference} composes these event measures into the same canonical
 * continuation and roulette products that the GPU stores.</p>
 */
final class RtPathPdfReference {
    private static final double PI = 3.14159265359;

    record Event(double techniqueProbability, double directionalDensity, boolean delta) {
        Event {
            requireFinitePositive(techniqueProbability, "technique probability");
            requireFinitePositive(directionalDensity, "directional density");
        }
    }

    static double combinedProposalDensity(Event... events) {
        if (events == null || events.length == 0) {
            throw new IllegalArgumentException("a path must contain at least one proposal event");
        }
        double result = 1.0;
        for (Event event : events) {
            if (event == null) {
                throw new IllegalArgumentException("proposal event must not be null");
            }
            result *= event.techniqueProbability();
            if (!event.delta()) {
                result *= event.directionalDensity();
            }
            if (!Double.isFinite(result) || result <= 0.0) {
                throw new IllegalArgumentException("combined proposal density must remain finite and positive");
            }
        }
        return result;
    }

    static double importanceWeight(double targetDensity, double proposalDensity) {
        if (!Double.isFinite(targetDensity) || targetDensity < 0.0) {
            throw new IllegalArgumentException("target density must be finite and non-negative");
        }
        requireFinitePositive(proposalDensity, "proposal density");
        double weight = targetDensity / proposalDensity;
        if (!Double.isFinite(weight)) {
            throw new IllegalArgumentException("importance weight must be finite");
        }
        return weight;
    }

    static double cosineHemisphereDensity(double cosine) {
        if (!Double.isFinite(cosine) || cosine < -1.0 || cosine > 1.0) {
            throw new IllegalArgumentException("cosine must be finite and in [-1, 1]");
        }
        return Math.max(cosine, 0.0) / PI;
    }

    static double ggxVndfReflectionDensity(double normalDotView, double normalDotHalf,
                                           double alpha) {
        if (!Double.isFinite(normalDotView) || !Double.isFinite(normalDotHalf)
                || normalDotView < 0.0 || normalDotView > 1.0
                || normalDotHalf < 0.0 || normalDotHalf > 1.0) {
            throw new IllegalArgumentException("GGX cosines must be finite and in [0, 1]");
        }
        if (!Double.isFinite(alpha) || alpha <= 0.0 || alpha > 1.0) {
            throw new IllegalArgumentException("GGX alpha must be finite and in (0, 1]");
        }
        if (normalDotView == 0.0 || normalDotHalf == 0.0) {
            return 0.0;
        }
        double alpha2 = alpha * alpha;
        double dDenominator = normalDotHalf * normalDotHalf * (alpha2 - 1.0) + 1.0;
        double distribution = alpha2 / (PI * dDenominator * dDenominator + 1.0e-7);
        double masking = 2.0 * normalDotView
                / (normalDotView
                + Math.sqrt(alpha2 + (1.0 - alpha2) * normalDotView * normalDotView)
                + 1.0e-7);
        return distribution * masking / (4.0 * normalDotView);
    }

    private RtPathPdfReference() {
    }

    private static void requireFinitePositive(double value, String label) {
        if (!Double.isFinite(value) || value <= 0.0) {
            throw new IllegalArgumentException(label + " must be finite and positive");
        }
    }
}
