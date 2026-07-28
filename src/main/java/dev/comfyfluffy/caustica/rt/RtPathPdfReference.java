package dev.comfyfluffy.caustica.rt;

/**
 * Shader-independent proposal-density contract for a path sample.
 *
 * <p>Each event records the probability of selecting the technique and, for continuous events,
 * the directional density in one agreed measure. Delta events deliberately do not multiply a
 * directional density: their mass is represented by the technique-selection probability. The
 * current GPU path does not yet emit all of these factors, so this model is a correctness gate for
 * the future PDF capture rather than a change to the active estimator.</p>
 */
final class RtPathPdfReference {
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

    private RtPathPdfReference() {
    }

    private static void requireFinitePositive(double value, String label) {
        if (!Double.isFinite(value) || value <= 0.0) {
            throw new IllegalArgumentException(label + " must be finite and positive");
        }
    }
}
