package dev.comfyfluffy.caustica.rt;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

final class RtTemporalValidationTest {
    @Test
    void depthToleranceIsConservativeAndBounded() {
        RtTemporalValidation.Policy policy = RtTemporalValidation.DEFAULT_POLICY;
        assertEquals(0.05f, policy.depthTolerance(1.0f));
        assertEquals(0.25f, policy.depthTolerance(50.0f));
        assertEquals(0.50f, policy.depthTolerance(1000.0f));
    }

    @Test
    void validationBitDoesNotOverlapDiagnosticReasons() {
        int reasons = RtTemporalValidation.HISTORY_AVAILABLE
                | RtTemporalValidation.REPROJECTED_IN_BOUNDS
                | RtTemporalValidation.SURFACE_PRESENT
                | RtTemporalValidation.NORMAL_COMPATIBLE
                | RtTemporalValidation.ROUGHNESS_COMPATIBLE
                | RtTemporalValidation.DEPTH_COMPATIBLE;
        assertEquals(0, reasons & RtTemporalValidation.VALID);
        assertEquals(0x80, RtTemporalValidation.VALID);
    }

    @Test
    void policyRejectsInvertedDepthBounds() {
        assertThrows(IllegalArgumentException.class,
                () -> new RtTemporalValidation.Policy(0.9f, 0.2f,
                        0.5f, 0.005f, 0.05f));
    }
}
