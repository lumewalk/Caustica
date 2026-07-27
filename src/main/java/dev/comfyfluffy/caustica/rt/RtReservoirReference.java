package dev.comfyfluffy.caustica.rt;

/**
 * Shader-independent reference implementation of weighted reservoir update and merge.
 *
 * <p>This is intentionally double precision and allocation-light, not GPU-shaped. It defines the
 * estimator contract that later shader passes must match: zero-weight candidates still contribute to
 * the effective count, and merging a source reservoir uses {@code pHatCurrent * WSource * MSource}.</p>
 */
final class RtReservoirReference {
    static int globalCandidateCount(int candidateCount, boolean hasGridCell) {
        if (candidateCount <= 0) {
            throw new IllegalArgumentException("candidate count must be positive");
        }
        return hasGridCell ? Math.max(1, (candidateCount + 2) / 4) : candidateCount;
    }

    static boolean usesLocalCandidate(int candidateIndex, int candidateCount, boolean hasGridCell) {
        if (candidateIndex < 0 || candidateIndex >= candidateCount) {
            throw new IllegalArgumentException("candidate index out of range");
        }
        int globals = globalCandidateCount(candidateCount, hasGridCell);
        int globalsBefore = candidateIndex * globals / candidateCount;
        int globalsAfter = (candidateIndex + 1) * globals / candidateCount;
        return hasGridCell && globalsAfter == globalsBefore;
    }

    record Candidate(long sampleId, double targetDensity, double proposalDensity) {
        Candidate {
            if (sampleId < 0L) {
                throw new IllegalArgumentException("sample id must be non-negative");
            }
            requireFiniteNonNegative(targetDensity, "target density");
            if (!Double.isFinite(proposalDensity) || proposalDensity <= 0.0) {
                throw new IllegalArgumentException("proposal density must be finite and positive");
            }
        }
    }

    record Snapshot(long selectedSampleId, boolean selected, double weightSum,
                    double effectiveCount, double selectedTargetDensity,
                    double finalWeight, int age) {
    }

    static final class Reservoir {
        private Candidate selected;
        private double selectedTargetDensity;
        private double weightSum;
        private double effectiveCount;
        private int age;

        void update(Candidate candidate, double random01) {
            requireRandom(random01);
            double candidateWeight = candidate.targetDensity() / candidate.proposalDensity();
            effectiveCount = saturatedAdd(effectiveCount, 1.0);
            if (select(candidateWeight, random01)) {
                selected = candidate;
                selectedTargetDensity = candidate.targetDensity();
                age = 0;
            }
        }

        void merge(Reservoir source, double targetDensityAtReceiver, double random01) {
            merge(source, targetDensityAtReceiver, random01, Double.MAX_VALUE);
        }

        void merge(Reservoir source, double targetDensityAtReceiver, double random01,
                   double maxSourceCount) {
            if (source == this) {
                throw new IllegalArgumentException("reservoir cannot merge itself");
            }
            requireFiniteNonNegative(targetDensityAtReceiver, "receiver target density");
            requireRandom(random01);
            if (!Double.isFinite(maxSourceCount) || maxSourceCount <= 0.0) {
                throw new IllegalArgumentException("maximum source count must be finite and positive");
            }
            double sourceCount = Math.min(source.effectiveCount, maxSourceCount);
            effectiveCount = saturatedAdd(effectiveCount, sourceCount);
            if (source.selected == null || sourceCount <= 0.0) {
                return;
            }
            double mergeWeight = targetDensityAtReceiver * source.finalWeight() * sourceCount;
            if (select(mergeWeight, random01)) {
                selected = new Candidate(source.selected.sampleId(), targetDensityAtReceiver,
                        source.selected.proposalDensity());
                selectedTargetDensity = targetDensityAtReceiver;
                age = source.age == Integer.MAX_VALUE ? Integer.MAX_VALUE : source.age + 1;
            }
        }

        double finalWeight() {
            if (selected == null || selectedTargetDensity <= 0.0 || effectiveCount <= 0.0) {
                return 0.0;
            }
            return weightSum / (effectiveCount * selectedTargetDensity);
        }

        Snapshot snapshot() {
            return new Snapshot(
                    selected == null ? -1L : selected.sampleId(),
                    selected != null,
                    weightSum,
                    effectiveCount,
                    selectedTargetDensity,
                    finalWeight(),
                    age);
        }

        private boolean select(double weight, double random01) {
            if (!Double.isFinite(weight) || weight < 0.0) {
                throw new IllegalArgumentException("candidate weight must be finite and non-negative");
            }
            double nextWeightSum = saturatedAdd(weightSum, weight);
            boolean choose = weight > 0.0
                    && (weightSum == 0.0 || random01 < weight / nextWeightSum);
            weightSum = nextWeightSum;
            return choose;
        }
    }

    private RtReservoirReference() {
    }

    private static double saturatedAdd(double left, double right) {
        double sum = left + right;
        return Double.isFinite(sum) ? sum : Double.MAX_VALUE;
    }

    private static void requireRandom(double random01) {
        if (!Double.isFinite(random01) || random01 < 0.0 || random01 >= 1.0) {
            throw new IllegalArgumentException("random value must be in [0, 1)");
        }
    }

    private static void requireFiniteNonNegative(double value, String label) {
        if (!Double.isFinite(value) || value < 0.0) {
            throw new IllegalArgumentException(label + " must be finite and non-negative");
        }
    }
}
