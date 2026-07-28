package dev.comfyfluffy.caustica.rt;

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
}
