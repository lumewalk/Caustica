package dev.comfyfluffy.caustica.rt;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

final class RtPathSpatialReuseReferenceTest {
    private static RtPathSpatialReuseReference.Surface surface(
            long material, double normalCosine, double relativeDepth, double footprint) {
        return new RtPathSpatialReuseReference.Surface(
                material, normalCosine, relativeDepth, 2, 42L, 2, footprint);
    }

    @Test
    void compatibleNeighborRequiresStrictSurfaceAndPathIdentity() {
        var receiver = surface(7L, 0.95, 0.02, 1.0);
        assertEquals(RtPathSpatialReuseReference.Decision.ACCEPTED,
                RtPathSpatialReuseReference.admit(receiver, surface(7L, 0.90, 0.05, 3.9)));
        assertEquals(RtPathSpatialReuseReference.Decision.MATERIAL_MISMATCH,
                RtPathSpatialReuseReference.admit(receiver, surface(8L, 0.95, 0.02, 1.0)));
        assertEquals(RtPathSpatialReuseReference.Decision.SURFACE_NORMAL_MISMATCH,
                RtPathSpatialReuseReference.admit(surface(7L, 0.84, 0.02, 1.0), receiver));
        assertEquals(RtPathSpatialReuseReference.Decision.RELATIVE_DEPTH_MISMATCH,
                RtPathSpatialReuseReference.admit(surface(7L, 0.95, 0.11, 1.0), receiver));
        assertEquals(RtPathSpatialReuseReference.Decision.FOOTPRINT_MISMATCH,
                RtPathSpatialReuseReference.admit(receiver, surface(7L, 0.95, 0.02, 4.1)));
    }

    @Test
    void reconnectionJacobianUsesSolidAngleGeometryAndPssPdfRatio() {
        var geometry = new RtPathSpatialReuseReference.ReconnectionGeometry(
                2.0, 4.0, 0.5, 0.25, 0.2, 0.4);
        assertEquals(0.125, geometry.solidAngleJacobian(), 1.0e-12);
        assertEquals(0.25, geometry.primarySampleJacobian(), 1.0e-12);
    }

    @Test
    void invalidReconnectionTermsAreRejectedBeforeAReuseDecision() {
        assertThrows(IllegalArgumentException.class,
                () -> new RtPathSpatialReuseReference.ReconnectionGeometry(
                        0.0, 1.0, 1.0, 1.0, 1.0, 1.0));
        assertThrows(IllegalArgumentException.class,
                () -> new RtPathSpatialReuseReference.Surface(
                        1L, 0.9, 0.0, 1, 1L, 1, Double.NaN));
    }
}
