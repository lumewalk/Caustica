package dev.comfyfluffy.caustica.rt;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

final class RtPathPdfReferenceTest {
    @Test
    void proposalDensityMultipliesTechniqueAndContinuousDirectionTerms() {
        double density = RtPathPdfReference.combinedProposalDensity(
                new RtPathPdfReference.Event(0.25, 1.0, true),
                new RtPathPdfReference.Event(0.5, 0.4, false));

        assertEquals(0.05, density, 1.0e-12);
    }

    @Test
    void deltaEventUsesDiscreteMassInsteadOfDirectionalDensity() {
        double density = RtPathPdfReference.combinedProposalDensity(
                new RtPathPdfReference.Event(0.25, 100.0, true),
                new RtPathPdfReference.Event(0.5, 0.4, false));

        assertEquals(0.05, density, 1.0e-12);
    }

    @Test
    void importanceWeightKeepsZeroTargetButRejectsInvalidProposal() {
        assertEquals(0.0, RtPathPdfReference.importanceWeight(0.0, 0.5));
        assertEquals(4.0, RtPathPdfReference.importanceWeight(2.0, 0.5), 1.0e-12);
        assertThrows(IllegalArgumentException.class,
                () -> RtPathPdfReference.importanceWeight(1.0, 0.0));
        assertThrows(IllegalArgumentException.class,
                () -> RtPathPdfReference.importanceWeight(Double.NaN, 1.0));
    }

    @Test
    void invalidOrOverflowingProposalIsRejected() {
        assertThrows(IllegalArgumentException.class,
                () -> new RtPathPdfReference.Event(0.0, 1.0, false));
        assertThrows(IllegalArgumentException.class,
                () -> RtPathPdfReference.combinedProposalDensity());
        assertThrows(IllegalArgumentException.class,
                () -> RtPathPdfReference.combinedProposalDensity(
                        new RtPathPdfReference.Event(Double.MAX_VALUE, 1.0, true),
                        new RtPathPdfReference.Event(2.0, 1.0, true)));
    }

    @Test
    void diffuseAndGgxDensitiesMatchShaderSamplingMeasures() {
        assertEquals(1.0 / 3.14159265359,
                RtPathPdfReference.cosineHemisphereDensity(1.0), 1.0e-12);
        assertEquals(0.0, RtPathPdfReference.cosineHemisphereDensity(-0.25));
        assertEquals(1.0 / (4.0 * 3.14159265359),
                RtPathPdfReference.ggxVndfReflectionDensity(1.0, 1.0, 1.0), 1.0e-8);
        assertEquals(0.0,
                RtPathPdfReference.ggxVndfReflectionDensity(1.0, 0.0, 0.5));
    }
}
