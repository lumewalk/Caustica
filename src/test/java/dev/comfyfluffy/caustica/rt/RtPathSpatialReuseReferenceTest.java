package dev.comfyfluffy.caustica.rt;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

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

    @Test
    void persistentReconnectionLaneSelectsAndPacksTheSecondPathHit() {
        assertFalse(RtPathSpatialReuseReference.captureReconnectionAtDepth(0));
        assertTrue(RtPathSpatialReuseReference.captureReconnectionAtDepth(1));
        assertFalse(RtPathSpatialReuseReference.captureReconnectionAtDepth(2));

        int packed = RtPathSpatialReuseReference.packReconnectionMetadata(1, true);
        assertTrue(RtPathSpatialReuseReference.reconnectionValid(packed));
        assertEquals(1, RtPathSpatialReuseReference.reconnectionDepth(packed));
        assertThrows(IllegalArgumentException.class,
                () -> RtPathSpatialReuseReference.packReconnectionMetadata(-1, true));
    }

    @Test
    void pairedMomentsExposeCovarianceAndCorrelationWithoutBatchStorage() {
        var moments = new RtPathSpatialReuseReference.PairMoments();
        moments.add(1.0, 2.0);
        moments.add(2.0, 4.0);
        moments.add(3.0, 6.0);

        assertEquals(3L, moments.count());
        assertEquals(2.0, moments.meanX(), 1.0e-12);
        assertEquals(4.0, moments.meanY(), 1.0e-12);
        assertEquals(1.0, moments.varianceX(), 1.0e-12);
        assertEquals(4.0, moments.varianceY(), 1.0e-12);
        assertEquals(2.0, moments.covariance(), 1.0e-12);
        assertEquals(1.0, moments.correlation(), 1.0e-12);
    }

    @Test
    void covarianceContractRejectsNonFinitePairsAndZeroVarianceCorrelation() {
        var moments = new RtPathSpatialReuseReference.PairMoments();
        assertThrows(IllegalArgumentException.class,
                () -> moments.add(Double.NaN, 1.0));
        moments.add(2.0, 4.0);
        moments.add(2.0, 8.0);
        assertTrue(Double.isNaN(moments.correlation()));
        assertTrue(Double.isNaN(new RtPathSpatialReuseReference.PairMoments().covariance()));
    }

    @Test
    void diagnosticCountersProduceResolutionIndependentCategoryRatios() {
        var counters = new RtPathSpatialReuseReference.DiagnosticCounters();
        counters.add(RtPathSpatialReuseReference.DiagnosticCategory.ACCEPTED_RECONNECTION);
        counters.add(RtPathSpatialReuseReference.DiagnosticCategory.ACCEPTED_RECONNECTION);
        counters.add(RtPathSpatialReuseReference.DiagnosticCategory.PATH_REJECT);
        counters.add(RtPathSpatialReuseReference.DiagnosticCategory.RECEIVER_EMPTY);

        assertEquals(4L, counters.total());
        assertEquals(2L, counters.count(
                RtPathSpatialReuseReference.DiagnosticCategory.ACCEPTED_RECONNECTION));
        assertEquals(0.5, counters.ratio(
                RtPathSpatialReuseReference.DiagnosticCategory.ACCEPTED_RECONNECTION), 1.0e-12);
        assertTrue(Double.isNaN(new RtPathSpatialReuseReference.DiagnosticCounters().ratio(
                RtPathSpatialReuseReference.DiagnosticCategory.SURFACE_REJECT)));
    }
}
