package dev.comfyfluffy.caustica.rt;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

final class RtPathCanonicalReferenceTest {
    @Test
    void validBlackEndpointIsDistinctFromMissingEndpoint() {
        int validEvents = RtPathCanonicalReference.packProposalEvents(2, 0, 0, true);
        int missingEvents = RtPathCanonicalReference.packProposalEvents(2, 0, 0, false);

        assertTrue(RtPathCanonicalReference.isConsistent(
                new RtPathCanonicalReference.Endpoint(0.0, 0.0, 0.0, 0.0, validEvents)));
        assertFalse(RtPathCanonicalReference.isConsistent(
                new RtPathCanonicalReference.Endpoint(0.0, 0.0, 0.0, 0.0, missingEvents)));
    }

    @Test
    void targetMustMatchStoredEndpointLuminance() {
        int events = RtPathCanonicalReference.packProposalEvents(1, 2, 3, true);

        assertTrue(RtPathCanonicalReference.isConsistent(
                new RtPathCanonicalReference.Endpoint(1.0, 0.5, 0.25,
                        RtPathCanonicalReference.luminance(1.0, 0.5, 0.25), events)));
        assertFalse(RtPathCanonicalReference.isConsistent(
                new RtPathCanonicalReference.Endpoint(1.0, 0.5, 0.25, 0.0, events)));
        assertFalse(RtPathCanonicalReference.isConsistent(
                new RtPathCanonicalReference.Endpoint(Double.NaN, 0.5, 0.25, 1.0, events)));
    }

    @Test
    void counterPackingSaturatesAndRejectsNegativeValues() {
        int packed = RtPathCanonicalReference.packProposalEvents(5000, 2000, 3000, true);
        assertTrue((packed & RtPathCanonicalReference.CANONICAL_VALID_BIT) != 0);
        assertThrows(IllegalArgumentException.class,
                () -> RtPathCanonicalReference.packProposalEvents(-1, 0, 0, true));
    }

    @Test
    void candidateAdmissionDistinguishesMissingEndpointAndZeroTarget() {
        assertEquals(RtPathCanonicalReference.CANDIDATE_NO_ENDPOINT,
                status(false, 1.0, 0.5, 0.25, 0.25, 0.8));
        assertEquals(RtPathCanonicalReference.CANDIDATE_ZERO_TARGET,
                status(true, 0.0, 0.0, 0.0, 0.25, 0.8));
        assertEquals(RtPathCanonicalReference.CANDIDATE_ACCEPTED,
                status(true, 1.0, 0.5, 0.25, 0.25, 0.8));
    }

    @Test
    void candidateAdmissionSeparatesInvalidRadianceAndPdfComponents() {
        assertEquals(RtPathCanonicalReference.CANDIDATE_INVALID_RADIANCE,
                status(true, Double.NaN, 0.5, 0.25, 0.25, 0.8));
        assertEquals(RtPathCanonicalReference.CANDIDATE_INVALID_CONTINUATION_PDF,
                status(true, 1.0, 0.5, 0.25, 0.0, 0.8));
        assertEquals(RtPathCanonicalReference.CANDIDATE_INVALID_ROULETTE_PDF,
                status(true, 1.0, 0.5, 0.25, 0.25, Double.POSITIVE_INFINITY));
        assertEquals(RtPathCanonicalReference.CANDIDATE_INVALID_PROPOSAL,
                status(true, 1.0e308, 0.0, 0.0, Double.MIN_VALUE, 1.0));
    }

    @Test
    void candidateAdmissionCodesMatchShaderContract() {
        assertEquals(0, RtPathCanonicalReference.CANDIDATE_ACCEPTED);
        assertEquals(1, RtPathCanonicalReference.CANDIDATE_NO_ENDPOINT);
        assertEquals(2, RtPathCanonicalReference.CANDIDATE_INVALID_RADIANCE);
        assertEquals(3, RtPathCanonicalReference.CANDIDATE_ZERO_TARGET);
        assertEquals(4, RtPathCanonicalReference.CANDIDATE_INVALID_CONTINUATION_PDF);
        assertEquals(5, RtPathCanonicalReference.CANDIDATE_INVALID_ROULETTE_PDF);
        assertEquals(6, RtPathCanonicalReference.CANDIDATE_INVALID_PROPOSAL);
    }

    private static int status(boolean canonicalValid, double red, double green, double blue,
                              double continuationProposalPdf, double rouletteProposalPdf) {
        return RtPathCanonicalReference.candidateStatus(new RtPathCanonicalReference.Candidate(
                canonicalValid, red, green, blue,
                continuationProposalPdf, rouletteProposalPdf));
    }
}
