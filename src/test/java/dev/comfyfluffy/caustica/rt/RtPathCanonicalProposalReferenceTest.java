package dev.comfyfluffy.caustica.rt;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

final class RtPathCanonicalProposalReferenceTest {
    @Test
    void allTransportClassesUseTheShaderSamplingMeasures() {
        var deltaReflection =
                RtPathCanonicalProposalReference.Continuation.deltaReflection(0.25);
        var deltaTransmission =
                RtPathCanonicalProposalReference.Continuation.deltaTransmission(0.75);
        var diffuse = RtPathCanonicalProposalReference.Continuation.diffuse(0.6, 0.5);
        var glossy = RtPathCanonicalProposalReference.Continuation.glossy(
                0.4, 0.8, 0.9, 0.35);
        var proposal = proposal(List.of(
                deltaReflection, deltaTransmission, diffuse, glossy),
                List.of(new RtPathCanonicalProposalReference.Roulette(0.8, true)),
                RtPathCanonicalProposalReference.EndpointKind.EMISSIVE);

        double expectedDiffuse = 0.6
                * RtPathPdfReference.cosineHemisphereDensity(0.5);
        double expectedGlossy = 0.4
                * RtPathPdfReference.ggxVndfReflectionDensity(0.8, 0.9, 0.35);
        double expectedContinuation =
                0.25 * 0.75 * expectedDiffuse * expectedGlossy;

        assertEquals(0.25, deltaReflection.proposalFactor(), 1.0e-12);
        assertEquals(0.75, deltaTransmission.proposalFactor(), 1.0e-12);
        assertEquals(expectedDiffuse, diffuse.proposalFactor(), 1.0e-12);
        assertEquals(expectedGlossy, glossy.proposalFactor(), 1.0e-12);
        assertEquals(expectedContinuation, proposal.continuationProposalDensity(), 1.0e-15);
        assertEquals(expectedContinuation * 0.8, proposal.canonicalProposalDensity(), 1.0e-15);
        assertEquals(15, proposal.transportMask());
    }

    @Test
    void rouletteUsesTheMassOfTheObservedOutcome() {
        var survived = proposal(List.of(),
                List.of(new RtPathCanonicalProposalReference.Roulette(0.8, true)),
                RtPathCanonicalProposalReference.EndpointKind.SKY);
        var terminated = proposal(List.of(),
                List.of(new RtPathCanonicalProposalReference.Roulette(0.8, false)),
                RtPathCanonicalProposalReference.EndpointKind.SKY);

        assertEquals(0.8, survived.rouletteProposalMass(), 1.0e-12);
        assertEquals(0.2, terminated.rouletteProposalMass(), 1.0e-12);
        assertThrows(IllegalArgumentException.class,
                () -> new RtPathCanonicalProposalReference.Roulette(1.0, false));
    }

    @Test
    void endpointHasUnitMeasureAndOnlySetsCanonicalValidBit() {
        for (RtPathCanonicalProposalReference.EndpointKind endpointKind
                : RtPathCanonicalProposalReference.EndpointKind.values()) {
            var proposal = proposal(List.of(), List.of(), endpointKind);
            var lane = proposal.gpuLane(1.0f, 0);

            assertEquals(1.0, proposal.continuationProposalDensity());
            assertEquals(1.0, proposal.rouletteProposalMass());
            assertEquals(1.0, proposal.canonicalProposalDensity());
            assertEquals(0, RtPathCanonicalReference.continuationEventCount(
                    lane.packedEvents()));
            assertEquals(0, RtPathCanonicalReference.rouletteEventCount(
                    lane.packedEvents()));
            assertTrue(RtPathCanonicalReference.canonicalValid(lane.packedEvents()));
        }
    }

    @Test
    void gpuLaneMatchesShaderProductsCountersAndBitcast() {
        var proposal = proposal(
                List.of(
                        RtPathCanonicalProposalReference.Continuation.deltaReflection(0.3),
                        RtPathCanonicalProposalReference.Continuation.diffuse(0.7, 0.6),
                        RtPathCanonicalProposalReference.Continuation.glossy(
                                0.2, 0.9, 0.8, 0.4)),
                List.of(
                        new RtPathCanonicalProposalReference.Roulette(0.9, true),
                        new RtPathCanonicalProposalReference.Roulette(0.75, true)),
                RtPathCanonicalProposalReference.EndpointKind.EMISSIVE);
        var lane = proposal.gpuLane(0.125f, 4);

        assertEquals(0.125f, lane.lightProposalPdf());
        assertEquals(proposal.gpuContinuationProposalPdf(),
                lane.continuationProposalPdf());
        assertEquals(proposal.gpuRouletteProposalPdf(), lane.rouletteProposalPdf());
        assertEquals(4, RtPathCanonicalReference.lightEventCount(lane.packedEvents()));
        assertEquals(3, RtPathCanonicalReference.continuationEventCount(
                lane.packedEvents()));
        assertEquals(2, RtPathCanonicalReference.rouletteEventCount(lane.packedEvents()));
        assertTrue(RtPathCanonicalReference.canonicalValid(lane.packedEvents()));
        assertTrue(RtPathCanonicalReference.reservedBitsClear(lane.packedEvents()));
        assertEquals(lane.packedEvents(),
                RtPathCanonicalReference.packedEventsFromLane(
                        lane.packedEventsAsFloat()));
    }

    @Test
    void gpuProductUsesTheSameClampAndInvalidRulesAsShader() {
        assertEquals(1.0e-30f,
                RtPathCanonicalProposalReference.gpuCombineProposal(1.0e-20f, 1.0e-20f));
        assertEquals(0.0f,
                RtPathCanonicalProposalReference.gpuCombineProposal(1.0e-25f, 1.0e-25f));
        assertEquals(1.0e30f,
                RtPathCanonicalProposalReference.gpuCombineProposal(1.0e25f, 1.0e25f));
        assertEquals(0.0f,
                RtPathCanonicalProposalReference.gpuCombineProposal(1.0f, 0.0f));
        assertEquals(0.0f,
                RtPathCanonicalProposalReference.gpuCombineProposal(Float.NaN, 1.0f));
    }

    @Test
    void invalidMeasuresCannotEnterTheReferencePath() {
        assertThrows(IllegalArgumentException.class,
                () -> RtPathCanonicalProposalReference.Continuation.deltaReflection(0.0));
        assertThrows(IllegalArgumentException.class,
                () -> new RtPathCanonicalProposalReference.Continuation(
                        RtPathCanonicalProposalReference.ContinuationKind.DELTA_REFLECTION,
                        0.5, 0.25));
        assertThrows(IllegalArgumentException.class,
                () -> RtPathCanonicalProposalReference.Continuation.diffuse(0.5, -0.25));
        assertThrows(NullPointerException.class,
                () -> new RtPathCanonicalProposalReference.Proposal(
                        List.of(), List.of(), null));
    }

    private static RtPathCanonicalProposalReference.Proposal proposal(
            List<RtPathCanonicalProposalReference.Continuation> continuations,
            List<RtPathCanonicalProposalReference.Roulette> roulettes,
            RtPathCanonicalProposalReference.EndpointKind endpointKind) {
        return new RtPathCanonicalProposalReference.Proposal(
                continuations, roulettes,
                new RtPathCanonicalProposalReference.Endpoint(endpointKind));
    }
}
