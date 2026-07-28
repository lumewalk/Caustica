package dev.comfyfluffy.caustica.rt;

import java.util.List;
import java.util.Objects;

/**
 * Shader-independent composition contract for a canonical path proposal.
 *
 * <p>Continuation events use a mixed measure: delta reflection/transmission contribute discrete
 * technique mass, while diffuse and glossy events contribute technique probability multiplied by
 * directional density. Roulette contributes the discrete mass of the outcome that actually
 * occurred. A sky or emissive endpoint terminates the canonical path with unit measure, so it sets
 * the canonical-valid bit but does not increment a proposal-event counter.</p>
 */
final class RtPathCanonicalProposalReference {
    static final int TRANSPORT_DIFFUSE = 1;
    static final int TRANSPORT_GLOSSY = 2;
    static final int TRANSPORT_DELTA = 4;
    static final int TRANSPORT_TRANSMISSION = 8;
    private static final float GPU_MIN_PROPOSAL = 1.0e-30f;
    private static final float GPU_MAX_PROPOSAL = 1.0e30f;

    enum ContinuationKind {
        DELTA_REFLECTION(TRANSPORT_DELTA, true),
        DELTA_TRANSMISSION(TRANSPORT_TRANSMISSION, true),
        DIFFUSE(TRANSPORT_DIFFUSE, false),
        GLOSSY(TRANSPORT_GLOSSY, false);

        private final int transportMask;
        private final boolean delta;

        ContinuationKind(int transportMask, boolean delta) {
            this.transportMask = transportMask;
            this.delta = delta;
        }
    }

    enum EndpointKind {
        SKY,
        EMISSIVE
    }

    record Continuation(ContinuationKind kind, double techniqueProbability,
                        double directionalDensity) {
        Continuation {
            Objects.requireNonNull(kind, "continuation kind");
            requireFinitePositive(techniqueProbability, "technique probability");
            requireFinitePositive(directionalDensity, "directional density");
            if (kind.delta && directionalDensity != 1.0) {
                throw new IllegalArgumentException(
                        "delta continuation must use unit directional density");
            }
        }

        static Continuation deltaReflection(double reflectionMass) {
            return new Continuation(ContinuationKind.DELTA_REFLECTION, reflectionMass, 1.0);
        }

        static Continuation deltaTransmission(double transmissionMass) {
            return new Continuation(ContinuationKind.DELTA_TRANSMISSION, transmissionMass, 1.0);
        }

        static Continuation diffuse(double techniqueProbability, double normalDotDirection) {
            return new Continuation(ContinuationKind.DIFFUSE, techniqueProbability,
                    RtPathPdfReference.cosineHemisphereDensity(normalDotDirection));
        }

        static Continuation glossy(double techniqueProbability, double normalDotView,
                                   double normalDotHalf, double alpha) {
            return new Continuation(ContinuationKind.GLOSSY, techniqueProbability,
                    RtPathPdfReference.ggxVndfReflectionDensity(
                            normalDotView, normalDotHalf, alpha));
        }

        double proposalFactor() {
            return techniqueProbability * (kind.delta ? 1.0 : directionalDensity);
        }

        int transportMask() {
            return kind.transportMask;
        }

        RtPathPdfReference.Event pdfEvent() {
            return new RtPathPdfReference.Event(
                    techniqueProbability, directionalDensity, kind.delta);
        }
    }

    record Roulette(double survivalProbability, boolean survived) {
        Roulette {
            if (!Double.isFinite(survivalProbability)
                    || survivalProbability <= 0.0 || survivalProbability > 1.0) {
                throw new IllegalArgumentException(
                        "roulette survival probability must be finite and in (0, 1]");
            }
            if (!survived && survivalProbability == 1.0) {
                throw new IllegalArgumentException(
                        "roulette termination must have positive probability");
            }
        }

        double proposalFactor() {
            return survived ? survivalProbability : 1.0 - survivalProbability;
        }
    }

    record Endpoint(EndpointKind kind) {
        Endpoint {
            Objects.requireNonNull(kind, "endpoint kind");
        }
    }

    record GpuProposalLane(float lightProposalPdf, float continuationProposalPdf,
                           float rouletteProposalPdf, int packedEvents) {
        float packedEventsAsFloat() {
            return RtPathCanonicalReference.packedEventsLane(packedEvents);
        }
    }

    record Proposal(List<Continuation> continuations, List<Roulette> roulettes,
                    Endpoint endpoint) {
        Proposal {
            continuations = List.copyOf(
                    Objects.requireNonNull(continuations, "continuations"));
            roulettes = List.copyOf(Objects.requireNonNull(roulettes, "roulettes"));
            Objects.requireNonNull(endpoint, "canonical endpoint");
        }

        double continuationProposalDensity() {
            if (continuations.isEmpty()) {
                return 1.0;
            }
            return RtPathPdfReference.combinedProposalDensity(
                    continuations.stream()
                            .map(Continuation::pdfEvent)
                            .toArray(RtPathPdfReference.Event[]::new));
        }

        double rouletteProposalMass() {
            double product = 1.0;
            for (Roulette roulette : roulettes) {
                product = strictProduct(product, roulette.proposalFactor());
            }
            return product;
        }

        double canonicalProposalDensity() {
            return strictProduct(continuationProposalDensity(), rouletteProposalMass());
        }

        float gpuContinuationProposalPdf() {
            float product = 1.0f;
            for (Continuation continuation : continuations) {
                product = gpuCombineProposal(product, (float) continuation.proposalFactor());
            }
            return product;
        }

        float gpuRouletteProposalPdf() {
            float product = 1.0f;
            for (Roulette roulette : roulettes) {
                product = gpuCombineProposal(product, (float) roulette.proposalFactor());
            }
            return product;
        }

        int transportMask() {
            int mask = 0;
            for (Continuation continuation : continuations) {
                mask |= continuation.transportMask();
            }
            return mask;
        }

        GpuProposalLane gpuLane(float lightProposalPdf, int lightEvents) {
            requireFinitePositive(lightProposalPdf, "light proposal PDF");
            int packedEvents = RtPathCanonicalReference.packProposalEvents(
                    lightEvents, continuations.size(), roulettes.size(), true);
            return new GpuProposalLane(lightProposalPdf, gpuContinuationProposalPdf(),
                    gpuRouletteProposalPdf(), packedEvents);
        }
    }

    static float gpuCombineProposal(float left, float right) {
        if (!(left > 0.0f) || !(right > 0.0f)) {
            return 0.0f;
        }
        float combined = left * right;
        return combined > 0.0f
                ? Math.min(Math.max(combined, GPU_MIN_PROPOSAL), GPU_MAX_PROPOSAL)
                : 0.0f;
    }

    private static double strictProduct(double left, double right) {
        double product = left * right;
        if (!Double.isFinite(product) || product <= 0.0) {
            throw new IllegalArgumentException(
                    "combined canonical proposal must remain finite and positive");
        }
        return product;
    }

    private static void requireFinitePositive(double value, String label) {
        if (!Double.isFinite(value) || value <= 0.0) {
            throw new IllegalArgumentException(label + " must be finite and positive");
        }
    }

    private RtPathCanonicalProposalReference() {
    }
}
