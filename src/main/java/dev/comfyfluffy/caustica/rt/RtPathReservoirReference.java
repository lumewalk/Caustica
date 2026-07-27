package dev.comfyfluffy.caustica.rt;

/**
 * Shader-independent reference contract for a GRIS/ReSTIR PT path reservoir.
 *
 * <p>The direct-light reference stores a light sample whose target can be evaluated at the
 * receiver. A path reservoir additionally carries the compact path identity needed to decide
 * whether a neighbor or historical sample can be shifted to the current path vertex. This class
 * deliberately uses strict compatibility until a concrete replay/reconnection mapping is wired
 * into the GPU path.</p>
 */
final class RtPathReservoirReference {
    static final double MAX_FOOTPRINT_RATIO = 4.0;

    record PathSample(long sampleId, int depth, long topologyKey,
                      long geometryEpoch, long materialEpoch, int transportClass,
                      long randomReplayState, double unshadowedContribution,
                      double targetDensity, double proposalDensity,
                      double shiftJacobian, double footprint) {
        PathSample {
            if (sampleId < 0L) {
                throw new IllegalArgumentException("path sample id must be non-negative");
            }
            if (depth < 0) {
                throw new IllegalArgumentException("path depth must be non-negative");
            }
            if (topologyKey < 0L || geometryEpoch < 0L || materialEpoch < 0L
                    || randomReplayState < 0L) {
                throw new IllegalArgumentException("path identity values must be non-negative");
            }
            requireFiniteNonNegative(unshadowedContribution, "unshadowed contribution");
            requireFiniteNonNegative(targetDensity, "target density");
            if (!Double.isFinite(proposalDensity) || proposalDensity <= 0.0) {
                throw new IllegalArgumentException("proposal density must be finite and positive");
            }
            if (!Double.isFinite(shiftJacobian) || shiftJacobian <= 0.0) {
                throw new IllegalArgumentException("shift Jacobian must be finite and positive");
            }
            if (!Double.isFinite(footprint) || footprint <= 0.0) {
                throw new IllegalArgumentException("path footprint must be finite and positive");
            }
        }

        static PathSample candidate(long sampleId, int depth, long topologyKey,
                                    long geometryEpoch, long materialEpoch, int transportClass,
                                    long randomReplayState, double unshadowedContribution,
                                    double targetDensity, double proposalDensity,
                                    double footprint) {
            return new PathSample(sampleId, depth, topologyKey, geometryEpoch, materialEpoch,
                    transportClass, randomReplayState, unshadowedContribution, targetDensity,
                    proposalDensity, 1.0, footprint);
        }
    }

    record Snapshot(long selectedSampleId, boolean selected, int depth, long topologyKey,
                    double weightSum, double effectiveCount, double finalWeight,
                    double estimatedContribution, int age) {
    }

    static boolean compatible(PathSample receiver, PathSample shifted) {
        if (receiver == null || shifted == null) {
            return false;
        }
        double footprintRatio = Math.max(receiver.footprint(), shifted.footprint())
                / Math.min(receiver.footprint(), shifted.footprint());
        return receiver.depth() == shifted.depth()
                && receiver.topologyKey() == shifted.topologyKey()
                && receiver.geometryEpoch() == shifted.geometryEpoch()
                && receiver.materialEpoch() == shifted.materialEpoch()
                && receiver.transportClass() == shifted.transportClass()
                && footprintRatio <= MAX_FOOTPRINT_RATIO;
    }

    static final class Reservoir {
        private PathSample selected;
        private double weightSum;
        private double effectiveCount;
        private int age;

        void update(PathSample sample, double random01) {
            requireRandom(random01);
            double candidateWeight = sample.targetDensity() / sample.proposalDensity();
            effectiveCount = saturatedAdd(effectiveCount, 1.0);
            if (select(candidateWeight, random01)) {
                selected = sample;
                age = 0;
            }
        }

        /**
         * Merge a source path after shifting its selected sample to the receiver.
         * The shift Jacobian is carried by {@code shiftedSample}; it is part of the GRIS weight,
         * never a post-hoc color multiplier.
         */
        void merge(Reservoir source, PathSample shiftedSample, double random01,
                   double maxSourceCount) {
            if (source == this) {
                throw new IllegalArgumentException("path reservoir cannot merge itself");
            }
            requireRandom(random01);
            if (!Double.isFinite(maxSourceCount) || maxSourceCount <= 0.0) {
                throw new IllegalArgumentException("maximum source count must be finite and positive");
            }
            if (source.selected == null || !compatible(source.selected, shiftedSample)) {
                return;
            }
            double sourceCount = Math.min(source.effectiveCount, maxSourceCount);
            if (sourceCount <= 0.0) {
                return;
            }
            double mergeWeight = shiftedSample.targetDensity()
                    * source.finalWeight() * sourceCount * shiftedSample.shiftJacobian();
            effectiveCount = saturatedAdd(effectiveCount, sourceCount);
            if (select(mergeWeight, random01)) {
                selected = shiftedSample;
                age = source.age == Integer.MAX_VALUE ? source.age : source.age + 1;
            }
        }

        double finalWeight() {
            if (selected == null || selected.targetDensity() <= 0.0
                    || effectiveCount <= 0.0) {
                return 0.0;
            }
            return weightSum / (effectiveCount * selected.targetDensity());
        }

        Snapshot snapshot() {
            return new Snapshot(
                    selected == null ? -1L : selected.sampleId(),
                    selected != null,
                    selected == null ? -1 : selected.depth(),
                    selected == null ? -1L : selected.topologyKey(),
                    weightSum,
                    effectiveCount,
                    finalWeight(),
                    selected == null ? 0.0 : selected.unshadowedContribution() * finalWeight(),
                    age);
        }

        private boolean select(double weight, double random01) {
            if (!Double.isFinite(weight) || weight < 0.0) {
                throw new IllegalArgumentException("path candidate weight must be finite and non-negative");
            }
            double nextWeightSum = saturatedAdd(weightSum, weight);
            boolean choose = weight > 0.0
                    && (weightSum == 0.0 || random01 < weight / nextWeightSum);
            weightSum = nextWeightSum;
            return choose;
        }
    }

    private RtPathReservoirReference() {
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
